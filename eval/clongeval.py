import contextlib
import os
import json
from dataclasses import asdict

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs

from ucm.logger import init_logger

from transformers import AutoTokenizer
from pathlib import Path
import argparse

MODEL_PATH = os.getenv("MODEL_PATH", "/home/models/Qwen2.5-14B-Instruct")
MODEL_NAME = MODEL_PATH.split('/')[-1].split('-')[0]
print(MODEL_NAME)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

logger = init_logger(__name__)

# tp_size = len()

def setup_environment_variables():
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["PYTHONHASHSEED"] = "123456"
    # os.environ["WORLD_SIZE"] = str(tp_size)


@contextlib.contextmanager
def print_output(module_path: str, name: str, model: str,
    sampling_params: SamplingParams,

):
    parser = argparse.ArgumentParser(description="ucm sparse config with dynamic hyperparameters")
    parser.add_argument('--sparse_ratio', type=float, default=0.3)
    parser.add_argument('--retrieval_stride', type=int, default=5)
    parser.add_argument('--init_window_sz', type=int, default=1)
    parser.add_argument('--local_window_sz', type=int, default=2)
    parser.add_argument('--min_blocks', type=int, default=4)
    parser.add_argument('--block_size', type=int, default=128)
    parser.add_argument('--dataset_file', type=str, default="./datasets/Lonbgench/multifieldqa_zh.jsonl")
    parser.add_argument('--output_path', type=str, default="./debug.txt")
    parser.add_argument('--storage_backends', type=str, default="./ucm_kv_cache")



    args = parser.parse_args()
    OUT_PUT_PATH = args.output_path

    print(OUT_PUT_PATH)
    txt_file = Path(OUT_PUT_PATH)
    txt_file.parent.mkdir(parents=True, exist_ok=True)

    ktc = KVTransferConfig(
        kv_connector=name,
        kv_connector_module_path=module_path,
        kv_role="kv_both",
        kv_connector_extra_config={
            "ucm_connector_name": "UcmNfsStore",
            "ucm_connector_config": {
                "storage_backends": args.storage_backends,
                "kv_block_size": 33554432,
            },

             "ucm_sparse_config": {
                "ESA": {
                    "init_window_sz": args.init_window_sz,
                    "local_window_sz": args.local_window_sz,
                    "min_blocks": args.min_blocks,
                    "sparse_ratio": args.sparse_ratio,
                    "retrieval_stride": args.retrieval_stride,
                }
            },
        },
    )

    llm_args = EngineArgs(
        model=model,
        kv_transfer_config=ktc,
        max_model_len=32768,
        gpu_memory_utilization=0.8,
        block_size = args.block_size,
        max_num_batched_tokens = 2048,
        trust_remote_code = True,
        enforce_eager = True,
        # tensor_parallel_size = 2,
    )

    llm = LLM(**asdict(llm_args))
    print("_" * 20)
    print(sampling_params)
    print("_" * 20)
    with open(args.dataset_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            assert line is not None
            data = json.loads(line)
            question = data["query"]
            answer = data["answer"]
            context = data["context"]
            outputs = llm.generate([get_prompt(f"{context}\n\n{question}")], sampling_params)

            for output in outputs:
                generated_text = output.outputs[0].text
                generated_text = generated_text.replace('\n', '').replace('"', '')
                with open(OUT_PUT_PATH, "a", encoding="utf-8") as file:
                    line = f'{{"pred": "{generated_text}", "answers": ["{answer}"], "length": {len(generated_text)}}}\n'
                    file.write(line)

def get_prompt(prompt):
    messages = [
            {
                "role": "system",
                "content": "先读问题，再根据下面的文章内容回答问题，不要进行分析，不要重复问题，用简短的语句给出答案。\n\n例如：“全国美国文学研究会的第十八届年会在哪所大学举办的？”\n回答应该为：“xx大学”。\n\n"
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    return tokenizer.apply_chat_template(
                        messages,
                        tokenize = False,
                        add_generation_prompt = True,
                        add_special_tokens = True)

def main():
    module_path = "ucm.integration.vllm.uc_connector"
    name = "UnifiedCacheConnectorV1"
    model = os.getenv("MODEL_PATH", "/home/models/Qwen2.5-14B-Instruct")
    setup_environment_variables()
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=64)
    print_output(module_path, name, model, sampling_params)

if __name__ == "__main__":
    main()


    