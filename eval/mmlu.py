import argparse
import os
import numpy as np
import pandas as pd
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.config import KVTransferConfig
from transformers import AutoTokenizer
from dataclasses import asdict
import json


subcategories = {
    "abstract_algebra": ["math"],
    "anatomy": ["health"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "clinical_knowledge": ["health"],
    "college_biology": ["biology"],
    "college_chemistry": ["chemistry"],
    "college_computer_science": ["computer science"],
    "college_mathematics": ["math"],
    "college_medicine": ["health"],
    "college_physics": ["physics"],
    "computer_security": ["computer science"],
    "conceptual_physics": ["physics"],
    "econometrics": ["economics"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "formal_logic": ["philosophy"],
    "global_facts": ["other"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_computer_science": ["computer science"],
    "high_school_european_history": ["history"],
    "high_school_geography": ["geography"],
    "high_school_government_and_politics": ["politics"],
    "high_school_macroeconomics": ["economics"],
    "high_school_mathematics": ["math"],
    "high_school_microeconomics": ["economics"],
    "high_school_physics": ["physics"],
    "high_school_psychology": ["psychology"],
    "high_school_statistics": ["math"],
    "high_school_us_history": ["history"],
    "high_school_world_history": ["history"],
    "human_aging": ["health"],
    "human_sexuality": ["culture"],
    "international_law": ["law"],
    "jurisprudence": ["law"],
    "logical_fallacies": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "medical_genetics": ["health"],
    "miscellaneous": ["other"],
    "moral_disputes": ["philosophy"],
    "moral_scenarios": ["philosophy"],
    "nutrition": ["health"],
    "philosophy": ["philosophy"],
    "prehistory": ["history"],
    "professional_accounting": ["other"],
    "professional_law": ["law"],
    "professional_medicine": ["health"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_studies": ["politics"],
    "sociology": ["culture"],
    "us_foreign_policy": ["politics"],
    "virology": ["health"],
    "world_religions": ["philosophy"],
}

categories = {
    "STEM": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
    "humanities": ["history", "philosophy", "law"],
    "social sciences": ["politics", "culture", "economics", "geography", "psychology"],
    "other (business, health, misc.)": ["other", "business", "health"],
}



choices = ["A", "B", "C", "D"]


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


def eval(args, subject, llm, tokenizer, dev_df, test_df):
    cors = []
    preds = []
    answers = choices[: test_df.shape[1] - 2]

    for i in range(test_df.shape[0]):
       
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        while input_ids.shape[-1] > 2048:
            k -= 1
            if k <= 0:
                break  
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids

        
        label = test_df.iloc[i, test_df.shape[1] - 1]

        sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=1, logprobs=4,  # return the top 4 probabilities
            skip_special_tokens=True)
        
        outputs = llm.generate(
            [prompt],
            sampling_params
            
        )

       
        pred = outputs[0].outputs[0].text.strip()
     
        cor = pred == label
        cors.append(cor)
        preds.append(pred)


    acc = np.mean(cors)
    cors = np.array(cors)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, preds


def main(args):
    module_path = "ucm.integration.vllm.uc_connector"
    name = "UnifiedCacheConnectorV1"
    
    ktc = KVTransferConfig(
        kv_connector=name,
        kv_connector_module_path=module_path,
        kv_role="kv_both",
        kv_connector_extra_config={
            "ucm_connector_name": "UcmNfsStore",
            "ucm_connector_config": {
                "storage_backends":args.storage_backends,
                "kv_block_size": 33554432,
            },
            "ucm_sparse_config": {
                "ESA": {
                    "init_window_sz": args.init_window_sz,
                    "local_window_sz": args.init_window_sz,
                    "min_blocks": args.min_blocks,
                    "sparse_ratio": args.sparse_ratio,
                    "retrieval_stride": args.retrieval_stride,
                }
            },
        },
    )

    model = os.getenv("MODEL_PATH", "/home/models/Qwen2.5-14B-Instruct")
    
    llm_args = EngineArgs(
         model=model,
        kv_transfer_config=ktc,
        max_model_len=16384,
        gpu_memory_utilization=0.6,
        max_num_batched_tokens=2048,
        block_size=128,
        enforce_eager=True,
        distributed_executor_backend="mp",
        tensor_parallel_size=1,
    )

    
    llm = LLM(**asdict(llm_args))
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(args.data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    model_name = os.path.basename(model)
    result_dir = os.path.join(args.save_dir, f"results_{model_name}")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    
    all_acc = {}
    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}

    
    for subject in subjects:
        
        dev_df = pd.read_csv(
            os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None
        )[: args.ntrain]
        test_df = pd.read_csv(
            os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None
        )

        
        cors, acc, preds = eval(args, subject, llm, tokenizer, dev_df, test_df)
        all_acc[subject] = acc
        # 更新分类统计
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

        # 保存结果
        test_df[f"{model_name}_correct"] = cors
        test_df[f"{model_name}_pred"] = preds
        test_df.to_csv( os.path.join(result_dir, f"{subject}.csv"), index=None,)
        
    
    # 保存各个类别的acc
    acc_df = pd.DataFrame(all_acc, index=['accuracy']) 
    acc_df.to_csv(os.path.join(result_dir, "all_categories.csv"), index=None,)

    # 打印各分类准确率
    output_file = os.path.join(result_dir, "accuracy_results.txt") 
    # 打开文件，使用 "w" 模式（覆盖写入），编码指定为 utf-8 以支持中文
    with open(output_file, "w", encoding="utf-8") as f:
        # 打印各子分类准确率并写入文件
        for subcat in subcat_cors:
            subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
            line = f"Average accuracy {subcat_acc:.3f} - {subcat}"
            print(line)  # 同时在控制台打印
            f.write(line + "\n")  # 写入文件，加换行符分隔
        
        # 空行分隔不同分类结果
        f.write("\n")
        print()  # 控制台也打印空行
        
        # 打印各分类准确率并写入文件
        for cat in cat_cors:
            cat_acc = np.mean(np.concatenate(cat_cors[cat]))
            line = f"Average accuracy {cat_acc:.3f} - {cat}"
            print(line)
            f.write(line + "\n")
        
        # 空行分隔
        f.write("\n")
        print()
        
        # 打印总体准确率并写入文件
        weighted_acc = np.mean(np.concatenate(all_cors))
        line = f"All categories Average accuracy: {weighted_acc:.3f}"
        print(line)
        f.write(line + "\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # esa
    parser.add_argument('--sparse_ratio', type=float, default=0.3)
    parser.add_argument('--retrieval_stride', type=int, default=5)
    parser.add_argument('--init_window_sz', type=int, default=1)
    parser.add_argument('--local_window_sz', type=int, default=2)
    parser.add_argument('--min_blocks', type=int, default=4)
    parser.add_argument('--block_size', type=int, default=128)

    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--data_dir", "-d", type=str, default="/home/externals/chenaozhu/datasets/MMLU/")
    parser.add_argument("--save_dir", "-s", type=str, default="/home/externals/chenaozhu/codes/vllm-workspace/mmlu_test_res")
    parser.add_argument('--storage_backends', type=str, default="./ucm_kv_cache")
    
    args = parser.parse_args()
    main(args)