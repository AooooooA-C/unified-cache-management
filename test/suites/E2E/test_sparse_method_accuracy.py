#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UCM Sparse Method Accuracy Test (Offline)

离线测试稀疏化方法对模型准确度的影响，使用 F1 Score 进行评估
基于 vLLM 本地推理，无需在线 API
"""

import contextlib
import datetime
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

# ⚠️ CRITICAL: Set environment variables BEFORE importing vLLM
# 这些环境变量必须在导入 vLLM 之前设置，否则 pytest 运行时会失败
os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_VERSION", "0.9.2")

# 添加项目路径到 PYTHONPATH
vllm_path = "/home/externals/wangwenxin21/caz/vllm"
ucm_path = "/home/externals/wangwenxin21/caz/unified-cache-management"
if vllm_path not in sys.path:
    sys.path.insert(0, vllm_path)
if ucm_path not in sys.path:
    sys.path.insert(0, ucm_path)

import pytest
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs

# 添加测试根目录路径
TEST_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(TEST_ROOT))

from common.capture_utils import export_vars
from common.config_utils import config_utils
from common.ucm_sparse_eval.eval_f1_score import (
    qa_f1_zh_score,
    scorer,
    extract_pred_after_think,
    has_think_tag,
)

from ucm.logger import init_logger

logger = init_logger(__name__)


@contextlib.contextmanager
def build_llm_with_uc(
    module_path: str,
    name: str,
    model: str,
    storage_backends: str,
    ucm_sparse_config: Dict,
    max_model_len: int = 32768,
    gpu_memory_utilization: float = 0.8,
):
    """
    构建带有 UCM 的 LLM 引擎
    
    Args:
        module_path: UCM connector 模块路径
        name: Connector 名称
        model: 模型路径
        storage_backends: KV cache 存储路径
        ucm_sparse_config: UCM 稀疏化配置
        max_model_len: 最大模型长度
        gpu_memory_utilization: GPU 内存利用率
    """
    ktc = KVTransferConfig(
        kv_connector=name,
        kv_connector_module_path=module_path,
        kv_role="kv_both",
        kv_connector_extra_config={
            "ucm_connector_name": "UcmNfsStore",
            "ucm_connector_config": {
                "storage_backends": storage_backends,
                "kv_block_size": 33554432,
            },
            "ucm_sparse_config": ucm_sparse_config,
        },
    )

    llm_args = EngineArgs(
        model=model,
        kv_transfer_config=ktc,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_batched_tokens=30000,
        block_size=128,
        enforce_eager=True,
        trust_remote_code=True,
        distributed_executor_backend="mp",
        tensor_parallel_size=1,
    )

    llm = LLM(**asdict(llm_args))
    try:
        yield llm
    finally:
        logger.info("LLM engine is exiting.")


def load_dataset(dataset_path: str, max_samples: int = None) -> List[Dict]:
    """
    加载 JSONL 格式的数据集
    
    Args:
        dataset_path: 数据集文件路径
        max_samples: 最大样本数，None 表示加载全部
    
    Returns:
        数据列表
    """
    dataset_full_path = TEST_ROOT / dataset_path
    
    data = []
    with open(dataset_full_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            data.append(json.loads(line))
    
    return data


def construct_prompt(item: Dict, tokenizer) -> str:
    """
    构建推理 prompt
    
    Args:
        item: 数据项
        tokenizer: tokenizer
    
    Returns:
        格式化的 prompt
    """
    context = item.get("context", "")
    question = item.get("input", "")
    
    prompt_text = f"""阅读以下文字并用中文简短回答：

{context}

现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。

问题：{question}
回答："""
    
    messages = [
        {
            "role": "system",
            "content": "先读问题，再根据下面的文章内容回答问题，不要进行分析，不要重复问题，用简短的语句给出答案。",
        },
        {"role": "user", "content": prompt_text},
    ]
    
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=True,
    )


def run_inference_batch(
    llm: LLM,
    dataset: List[Dict],
    tokenizer,
    batch_size: int = 20,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    top_p: float = 0.95,
) -> List[Dict]:
    """
    批量推理
    
    Args:
        llm: vLLM 引擎
        dataset: 数据集
        tokenizer: tokenizer
        batch_size: 批大小
        max_tokens: 最大生成 token 数
        temperature: 温度参数
        top_p: top_p 采样
    
    
    Returns:
        预测结果列表
    """
    results = []
    total_data = len(dataset)
    
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        ignore_eos=False
    )
    
    for start_idx in range(0, total_data, batch_size):
        end_idx = min(start_idx + batch_size, total_data)
        current_batch = dataset[start_idx:end_idx]
        
        print(f"\n[Batch {start_idx//batch_size + 1}] Processing samples {start_idx+1}-{end_idx}/{total_data}")
        
        # 构建 prompts
        prompts = []
        for item in current_batch:
            prompt = construct_prompt(item, tokenizer)
            prompts.append(prompt)
        
        # 推理
        start_time = time.time()
        outputs = llm.generate(prompts, sampling_params)
        elapsed = time.time() - start_time
        
        # 收集结果
        for item, output in zip(current_batch, outputs):
            generated_text = output.outputs[0].text
            # 去除多余的换行和空格
            generated_text = "".join(
                [line.strip() for line in generated_text.splitlines() if line.strip()]
            )
            
            result = {
                "_id": item.get("_id"),
                "input": item.get("input"),
                "context_length": len(item.get("context", "")),
                "answers": item.get("answers"),
                "pred": generated_text,
                "length": item.get("length"),
                "dataset": item.get("dataset"),
            }
            results.append(result)
        
        print(f"  Generated {len(current_batch)} predictions in {elapsed:.2f}s")
        print(f"  Avg time: {elapsed/len(current_batch):.2f}s per sample")
    
    return results


def save_predictions(results: List[Dict], output_file: str):
    """保存预测结果"""
    output_path = TEST_ROOT / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"\nPredictions saved to: {output_path}")


def evaluate_f1_score(results: List[Dict], dataset_name: str, strip_think: bool = False) -> Dict:
    """
    评估 F1 Score
    
    Args:
        results: 预测结果列表
        dataset_name: 数据集名称
        strip_think: 是否提取 </think> 标签后的内容
    """
    predictions = []
    answers = []
    valid_count = 0
    
    for r in results:
        pred_raw = r["pred"]
        
        # 如果启用 strip_think，提取 </think> 后的内容
        if strip_think:
            if has_think_tag(pred_raw):
                pred_clean = extract_pred_after_think(pred_raw)
                valid_count += 1
            else:
                # 没有 think 标签，跳过该样本
                continue
        else:
            pred_clean = pred_raw
            valid_count += 1
        
        predictions.append(pred_clean)
        answers.append(r["answers"])
    
    print(f"  Valid samples for evaluation: {valid_count}/{len(results)}")
    if strip_think:
        print(f"  Samples with </think> tag: {valid_count}")
    
    # 使用 eval.py 中的评分函数
    f1_score = scorer(dataset_name, predictions, answers, all_classes=None)
    
    # 也计算前 50 个样本的分数
    if len(predictions) >= 50:
        f1_score_50 = scorer(dataset_name, predictions[:50], answers[:50], all_classes=None)
    else:
        f1_score_50 = f1_score
    
    evaluation_result = {
        "dataset": dataset_name,
        "total_samples": len(results),
        "valid_samples": valid_count,
        "f1_score": f1_score,
        "f1_score_50": f1_score_50,
        "metric": "qa_f1_zh_score",
        "strip_think": strip_think,
    }
    
    return evaluation_result


def save_evaluation(evaluation: Dict, output_file: str):
    """保存评估结果"""
    output_path = TEST_ROOT / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation, f, ensure_ascii=False, indent=2)
    
    print(f"Evaluation saved to: {output_path}")


@pytest.mark.stage(2)  # Regression test
@pytest.mark.feature("ucm_sparse_accuracy_test")
@export_vars
def test_sparse_method_accuracy():
    """
    测试稀疏化方法的准确度（离线）
    
    流程：
    1. 加载 multifieldqa_zh 数据集
    2. 使用 vLLM 本地推理
    3. 收集预测结果
    4. 使用 eval_f1_score.py 评估 F1 Score
    5. 断言 F1 Score >= 60
    """
    # 获取配置（已大幅精简，只保留必需项）
    config = config_utils.get_config("sparse_accuracy_test")
    
    # 从配置读取核心参数
    model_path = config.get("model_path", "/home/models/Qwen2.5-14B-Instruct")
    storage_backends = config.get("storage_backends", "/tmp/ucm_kv_cache_test")
    ucm_sparse_config_path = config.get("ucm_sparse_config", None)
    max_samples = config.get("max_samples", None)
    expected_f1_score = config.get("expected_f1_score", 60.0)
    strip_think = config.get("strip_think", False)
    
    # 数据集配置（固定值）
    dataset_config = {
        "name": "multifieldqa_zh",
        "path": "common/ucm_sparse_eval/data/multifieldqa_zh.jsonl",
        "max_samples": max_samples,
    }
    
    # 推理配置（固定默认值）
    inference_config = {
        "model_path": model_path,
        "storage_backends": storage_backends,
        "batch_size": 20,
        "max_tokens": 2048,
        "temperature": 0.0,
    }
    
    # 评估配置
    eval_config = {
        "expected_f1_score": expected_f1_score,
        "strip_think": strip_think,
    }
    
    # 输出配置（固定值）
    output_config = {
        "predictions_file": "results/sparse_accuracy_predictions.jsonl",
        "evaluation_file": "results/sparse_accuracy_evaluation.json",
    }
    
    print("\n" + "="*70)
    print("UCM Sparse Method Accuracy Test (Offline)")
    print("="*70)
    
    # 1. 加载数据集
    print(f"\n[Step 1] Loading dataset: {dataset_config['name']}")
    dataset = load_dataset(
        dataset_config["path"],
        max_samples=dataset_config.get("max_samples")
    )
    print(f"  Loaded {len(dataset)} samples")
    
    # 2. 准备模型和配置
    print(f"\n[Step 2] Preparing model and UCM config")
    model_path = inference_config.get("model_path", "/home/models/Qwen2.5-14B-Instruct")
    storage_backends = inference_config.get("storage_backends", "/tmp/ucm_kv_cache_test")
    ucm_sparse_config_path = inference_config.get("ucm_sparse_config", None)  # null 使用默认配置
    
    # 加载 UCM 稀疏化配置
    if ucm_sparse_config_path and os.path.exists(ucm_sparse_config_path):
        with open(ucm_sparse_config_path, 'r', encoding='utf-8') as f:
            ucm_sparse_config = json.load(f)
    else:
        # 默认配置
        ucm_sparse_config = {
            "ESA": {
                "init_window_sz": 1,
                "local_window_sz": 2,
                "min_blocks":4,
                "sparse_ratio": 0.2,
                "retrieval_stride": 5,
            }
        }
    
    print(f"  Model: {model_path}")
    print(f"  Storage: {storage_backends}")
    print(f"  UCM Config: {ucm_sparse_config}")
    
    # 创建存储目录
    os.makedirs(storage_backends, exist_ok=True)
    
    # 3. 初始化 tokenizer 和 LLM
    print(f"\n[Step 3] Initializing tokenizer and LLM...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_chat_template=False)
    
    module_path = "ucm.integration.vllm.uc_connector"
    connector_name = "UnifiedCacheConnectorV1"
    
    # 4. 运行推理
    print(f"\n[Step 4] Running offline inference with vLLM...")
    
    with build_llm_with_uc(
        module_path=module_path,
        name=connector_name,
        model=model_path,
        storage_backends=storage_backends,
        ucm_sparse_config=ucm_sparse_config,
        max_model_len=32768,  # 使用默认值
        gpu_memory_utilization=0.8,  # 使用默认值
    ) as llm:
        predictions = run_inference_batch(
            llm=llm,
            dataset=dataset,
            tokenizer=tokenizer,
            batch_size=inference_config.get("batch_size", 20),
            max_tokens=inference_config.get("max_tokens", 2048),
            temperature=inference_config.get("temperature", 0.0),
            top_p=0.95,  # 使用默认值
          
        )
    
    # 5. 保存预测结果
    print(f"\n[Step 5] Saving predictions...")
    save_predictions(predictions, output_config["predictions_file"])
    
    # 6. 评估 F1 Score
    print(f"\n[Step 6] Evaluating F1 Score...")
    strip_think = eval_config.get("strip_think", False)
    print(f"  Strip think mode: {strip_think}")
    evaluation = evaluate_f1_score(predictions, dataset_config["name"], strip_think=strip_think)
    
    print("\n" + "="*70)
    print("Evaluation Results:")
    print("="*70)
    print(f"  Model: {model_path}")
    print(f"  Dataset: {evaluation['dataset']}")
    print(f"  Total Samples: {evaluation['total_samples']}")
    print(f"  F1 Score (first 50): {evaluation['f1_score_50']:.2f}")
    print(f"  F1 Score (all): {evaluation['f1_score']:.2f}")
    print(f"  Metric: {evaluation['metric']}")
    print("="*70)
    
    # 7. 保存评估结果
    save_evaluation(evaluation, output_config["evaluation_file"])
    
    # 7. 准备测试结果（先生成，确保无论是否通过都能保存）
    expected_f1 = eval_config["expected_f1_score"]
    actual_f1 = evaluation["f1_score"]
    test_passed = actual_f1 >= expected_f1
    
    # 生成 test_build_id（与 conftest.py 中的格式一致）
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    test_build_id = f"pytest_{timestamp}_feature=ucm_sparse_accuracy_test"
    
    # 准备返回数据（在断言之前，确保失败时也能保存）
    result_data = {
        '_name': 'sparse_accuracy_test_offline',
        '_data': {
            'dataset': evaluation['dataset'],
            'total_samples': evaluation['total_samples'],
            'f1_score': evaluation['f1_score'],
            'f1_score_50': evaluation['f1_score_50'],
            'expected_f1_score': expected_f1,
            'test_passed': test_passed,
            'model_path': model_path,
            'ucm_sparse_config': str(ucm_sparse_config),
            'test_build_id': test_build_id,
            'strip_think': strip_think,
        }
    }
    
    # 立即保存结果（在断言之前）
    from common.capture_utils import post_process
    post_process(result_data['_name'], **result_data)
    
    # 显示 KV cache 信息
    print(f"\n[Step 8] KV cache preserved for reuse")
    print(f"  Location: {storage_backends}")
    
    try:
        import subprocess
        size_output = subprocess.check_output(
            ['du', '-sh', storage_backends],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        size = size_output.split()[0]
        print(f"  Size: {size}")
    except:
        print(f"  Size: (unable to calculate)")
    
    print(f"\n  💡 To manually remove KV cache:")
    print(f"     rm -rf {storage_backends}")
    
    # 8. 断言 F1 Score（放在最后，失败时前面的结果已保存）
    print(f"\n[Step 9] Asserting F1 Score >= {expected_f1}")
    
    assert test_passed, (
        f"F1 Score {actual_f1:.2f} is below expected threshold {expected_f1:.2f}"
    )
    
    print(f"\n✓ Test passed! F1 Score {actual_f1:.2f} >= {expected_f1:.2f}")
    
    # 返回结果用于 export_vars 装饰器（虽然已经保存，但保留兼容性）
    return result_data


if __name__ == "__main__":
    """
    直接运行测试（不通过 pytest）
    """
    # 设置环境变量
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["PYTHONHASHSEED"] = "123456"
    
    # 添加路径
    code_root = Path(__file__).resolve().parent.parent.parent.parent.parent
    sys.path.insert(0, str(code_root / "vllm"))
    sys.path.insert(0, str(code_root / "unified-cache-management"))
    
    print("Running sparse method accuracy test (offline)...")
    result = test_sparse_method_accuracy()
    print("\nTest completed!")
    print(f"Result: {json.dumps(result, indent=2, ensure_ascii=False)}")
