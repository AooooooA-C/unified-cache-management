import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import requests
from common.capture_utils import export_vars
from transformers import AutoTokenizer

# 添加路径以导入评分函数
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from common.sparse_method_accuracy_eval.run_eval import (
    dataset2metric,
    qa_f1_zh_score,
    extract_pred_after_think,
    has_think_tag
)


def load_dataset(dataset_path):
    """加载数据集"""
    data = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def call_llm_inference(server_url, model, messages, max_tokens=512, temperature=0.3):
    """调用在线LLM进行推理"""
    api_url = f"{server_url}/v1/chat/completions"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False
    }
    
    try:
        response = requests.post(api_url, json=payload, headers=headers, timeout=180)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[ERROR] LLM inference failed: {e}")
        return None


def run_accuracy_evaluation(
    dataset_name,
    dataset_path,
    server_url,
    model,
    max_samples=None,
    temperature=0.3,
    max_tokens=512,
    tokenizer_path=None,
    strip_think=False
):
    """运行精度评测"""
    # 加载tokenizer
    tokenizer = None
    if tokenizer_path:
        try:
            print(f"[INFO] Loading tokenizer from: {tokenizer_path}")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
            print(f"[INFO] Tokenizer loaded successfully")
        except Exception as e:
            print(f"[WARNING] Failed to load tokenizer: {e}")
            print(f"[WARNING] Will not calculate prompt token count")
    else:
        print(f"[WARNING] No tokenizer path provided, will not calculate prompt token count")
    
    # 加载数据集
    print(f"[INFO] Loading dataset from: {dataset_path}")
    dataset = load_dataset(dataset_path)
    
    if max_samples:
        dataset = dataset[:max_samples]
    
    print(f"[INFO] Total samples to evaluate: {len(dataset)}")
    
    # 创建结果目录
    results_dir = Path("results/accuracy")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成答案文件路径
    answer_file = results_dir / f"{dataset_name}_answers.jsonl"
    
    # 对每个样本进行推理
    with open(answer_file, 'w', encoding='utf-8') as f:
        for idx, item in enumerate(dataset):
            print(f"[INFO] Processing sample {idx + 1}/{len(dataset)}")
            
            # 构建prompt - 使用与test_sparse_accuracy.py相同的格式
            if 'context' in item and item['context']:
                user_prompt = f"""阅读以下文字并用中文简短回答：\n\n{item['context']}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{item['input']}\n回答："""
            else:
                user_prompt = f"""问题：{item['input']}\n回答："""
            
            # 构建messages，包含system message
            system_content = """先读问题，再根据下面的文章内容回答问题，不要进行分析，不要重复问题，用简短的语句给出答案。

例如："全国美国文学研究会的第十八届年会在哪所大学举办的？"
回答应该为："xx大学"。

"""
            messages = [
                {
                    "role": "system",
                    "content": system_content
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
            
            # 计算prompt token数（使用chat template获得准确值）
            if tokenizer:
                try:
                    # 方法1: 使用apply_chat_template（最准确）
                    formatted_prompt = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    tokens = tokenizer.encode(formatted_prompt)
                    prompt_token_count = len(tokens)
                except Exception as e:
                    # 降级为简单拼接（作为备选）
                    print(f"[WARNING] Chat template failed, using simple concatenation: {e}")
                    try:
                        full_prompt = f"{system_content}\n\n{user_prompt}"
                        tokens = tokenizer.encode(full_prompt)
                        prompt_token_count = len(tokens)
                    except Exception as e2:
                        print(f"[WARNING] Token counting failed for sample {idx + 1}: {e2}")
                        prompt_token_count = 0
            else:
                prompt_token_count = 0
            
            # 动态调整max_tokens，避免超出模型限制
            # 假设模型最大长度为32k，预留安全边界
            model_max_length = getattr(tokenizer, 'model_max_length', 32768) if tokenizer else 32768
            safe_margin = 100  # 安全边界
            
            if prompt_token_count > 0:
                # 计算可用的输出token数
                available_tokens = model_max_length - prompt_token_count - safe_margin
                adjusted_max_tokens = min(max_tokens, max(available_tokens, 50))
                
                if adjusted_max_tokens < max_tokens:
                    print(f"[INFO]   Adjusted max_tokens: {max_tokens} → {adjusted_max_tokens} (prompt: {prompt_token_count} tokens)")
            else:
                adjusted_max_tokens = max_tokens
            
            # 调用LLM（使用调整后的max_tokens）
            pred_raw = call_llm_inference(server_url, model, messages, adjusted_max_tokens, temperature)
            
            if pred_raw is None:
                pred_raw = ""
            
            # 处理think模型的输出
            if strip_think and pred_raw:
                if has_think_tag(pred_raw):
                    # 有</think>标签，提取之后的内容
                    pred = extract_pred_after_think(pred_raw)
                    print(f"[INFO]   Extracted answer after </think> tag")
                else:
                    # 没有</think>标签，说明思考未完成，保留原输出
                    pred = pred_raw
                    print(f"[INFO]   No </think> tag found, using raw output")
            else:
                # 不是think模型，直接使用原输出
                pred = pred_raw
            
            # 计算该样本的F1 score
            answers = item.get("answers", [])
            sample_f1_score = 0.0
            
            if pred and answers:
                # 获取该数据集对应的评分函数
                score_func = dataset2metric.get(dataset_name, qa_f1_zh_score)
                
                # 对每个标准答案计算F1，取最高分
                for answer in answers:
                    try:
                        score = score_func(pred, answer, all_classes=None)
                        sample_f1_score = max(sample_f1_score, score)
                    except Exception as e:
                        print(f"[WARNING] F1 calculation failed for sample {idx + 1}: {e}")
                
                # 转换为百分制
                sample_f1_score = round(100 * sample_f1_score, 2)
            
            # 保存结果（添加完整字段）
            result = {
                "input": item.get("input", ""),           # 问题
                "pred": pred,                             # 预测答案（可能已提取）
                "pred_raw": pred_raw if strip_think else None,  # 原始输出（think模式下保留）
                "answers": answers,                       # 标准答案
                "prompt_tokens": prompt_token_count,      # 实际prompt token数
                "context_length": item.get("length", 0),  # 原始上下文字符数
                "f1_score": sample_f1_score,             # 该样本的F1分数
                "has_think": has_think_tag(pred_raw) if strip_think else None  # 是否包含think标签
            }
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            # 实时显示详细信息
            print(f"[INFO]   → Prompt tokens: {prompt_token_count}, F1 Score: {sample_f1_score}")
    
    print(f"[INFO] Inference completed. Results saved to: {answer_file}")
    
    # 调用run_eval.py进行评测
    eval_script = Path(__file__).parent.parent.parent / "common/sparse_method_accuracy_eval/run_eval.py"
    
    print(f"[INFO] Running evaluation script: {eval_script}")
    
    try:
        cmd = [
            "python",
            str(eval_script),
            "--model", model,
            "--answer", str(answer_file),
            "--dataset", dataset_name
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        print("[INFO] Evaluation output:")
        print(result.stdout)
        
        # 解析评测结果
        lines = result.stdout.strip().split('\n')
        scores = {}
        
        for line in lines:
            if "50 score:" in line:
                scores['score_50'] = float(line.split(':')[-1].strip())
            elif "All score:" in line:
                scores['score_all'] = float(line.split(':')[-1].strip())
            elif "有效条数:" in line:
                scores['valid_count'] = int(line.split(':')[-1].strip())
        
        return scores
        
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Evaluation failed: {e}")
        print(f"[ERROR] stdout: {e.stdout}")
        print(f"[ERROR] stderr: {e.stderr}")
        return None


@pytest.mark.parametrize("dataset_name", ["multifieldqa_zh"])
@pytest.mark.parametrize("max_samples", [None])  # None表示使用全部数据
@pytest.mark.parametrize("temperature", [0])
@pytest.mark.parametrize("max_tokens", [512])
@pytest.mark.feature("uc_accuracy_test")
@export_vars
def test_accuracy(
    dataset_name,
    max_samples,
    temperature,
    max_tokens,
    config
):
    """测试稀疏方法的精度（F1 score）"""
    # 从config中读取配置
    llm_config = config.get("llm_connection", {})
    accuracy_config = config.get("accuracy_evaluation", {})
    
    server_url = llm_config.get("server_url", "http://127.0.0.1:7800")
    model = llm_config.get("model", "/home/models/Qwen2.5-14B-Instruct")
    tokenizer_path = llm_config.get("tokenizer_path")
    strip_think = accuracy_config.get("strip_think", False)
    
    # 构建数据集路径
    dataset_dir = Path(__file__).parent.parent.parent / "common/sparse_method_accuracy_eval/data"
    dataset_path = dataset_dir / f"{dataset_name}.jsonl"
    
    # 如果配置中指定了参数，使用配置的值
    if accuracy_config.get("max_samples") is not None:
        max_samples = accuracy_config["max_samples"]
    if accuracy_config.get("temperature") is not None:
        temperature = accuracy_config["temperature"]
    if accuracy_config.get("max_tokens") is not None:
        max_tokens = accuracy_config["max_tokens"]
    
    print(f"\n[INFO] ========== Accuracy Evaluation Start ==========")
    print(f"[INFO] Dataset: {dataset_name}")
    print(f"[INFO] Server URL: {server_url}")
    print(f"[INFO] Model: {model}")
    print(f"[INFO] Max samples: {max_samples if max_samples else 'All'}")
    print(f"[INFO] Temperature: {temperature}")
    print(f"[INFO] Max tokens: {max_tokens}")
    print(f"[INFO] Strip think tags: {strip_think}")
    
    # 检查数据集文件是否存在
    assert dataset_path.exists(), f"Dataset file not found: {dataset_path}"
    
    # 运行评测
    scores = run_accuracy_evaluation(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        server_url=server_url,
        model=model,
        max_samples=max_samples,
        temperature=temperature,
        max_tokens=max_tokens,
        tokenizer_path=tokenizer_path,
        strip_think=strip_think
    )
    
    # 验证评测结果
    assert scores is not None, "Evaluation failed"
    assert "score_all" in scores, "Missing overall F1 score"
    
    print(f"\n[INFO] ========== Evaluation Results ==========")
    print(f"[INFO] Model: {model}")
    print(f"[INFO] Dataset: {dataset_name}")
    print(f"[INFO] Think mode: {'Enabled' if strip_think else 'Disabled'}")
    print(f"[INFO] Valid samples: {scores.get('valid_count', 'N/A')}")
    print(f"[INFO] F1 Score: {scores['score_all']}")
    
    # F1阈值设置（固定值）
    f1_threshold = 60.0
    
    # 检查F1 score是否达到阈值
    if f1_threshold is not None:
        score_all = scores['score_all']
        print(f"[INFO] F1 Score threshold: {f1_threshold}")
        try:
            assert score_all >= f1_threshold, f"F1 score: {score_all} < threshold: {f1_threshold}"
            print(f"[INFO] ✓ F1 Score passed threshold check")
        except AssertionError as e:
            print(f"[WARNING] ✗ F1 Score failed: {e}")
            print(f"[INFO] ========== Accuracy Evaluation End ==========\n")
            pytest.fail(f"F1 Score assertion failed: {e}")
    
    print(f"[INFO] ========== Accuracy Evaluation End ==========\n")
    
    # 返回结果供后续处理
    return {
        "_name": "accuracy_evaluation",
        "_data": {
            "dataset": dataset_name,
            "model": model,
            "f1_score": scores['score_all'],
            "f1_threshold": f1_threshold,
            "valid_count": scores.get('valid_count'),
            "temperature": temperature,
            "max_tokens": max_tokens
        }
    }

