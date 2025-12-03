#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析稀疏化准确度测试结果

提供详细的统计分析和错误案例分析
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

# 添加项目路径
TEST_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(TEST_ROOT))

from common.ucm_sparse_eval.eval_f1_score import qa_f1_zh_score


def load_predictions(predictions_file: str) -> List[Dict]:
    """加载预测结果"""
    predictions_path = TEST_ROOT / predictions_file
    
    if not predictions_path.exists():
        print(f"✗ Predictions file not found: {predictions_path}")
        return []
    
    predictions = []
    with open(predictions_path, 'r', encoding='utf-8') as f:
        for line in f:
            predictions.append(json.loads(line))
    
    return predictions


def analyze_predictions(predictions: List[Dict]) -> Dict:
    """分析预测结果"""
    
    analysis = {
        "total_samples": len(predictions),
        "scores": [],
        "score_distribution": defaultdict(int),
        "low_score_samples": [],
        "high_score_samples": [],
        "empty_predictions": 0,
        "context_length_stats": {
            "min": float('inf'),
            "max": 0,
            "avg": 0,
        }
    }
    
    total_context_length = 0
    
    for pred in predictions:
        # 计算每个样本的 F1
        answers = pred.get("answers", [])
        prediction = pred.get("pred", "")
        
        if not prediction:
            analysis["empty_predictions"] += 1
            score = 0.0
        else:
            # 计算 F1 Score
            score = max(
                qa_f1_zh_score(prediction, answer) 
                for answer in answers
            ) if answers else 0.0
        
        score_percent = score * 100
        analysis["scores"].append(score_percent)
        
        # 分数分布（0-10, 10-20, ..., 90-100）
        bucket = int(score_percent // 10) * 10
        analysis["score_distribution"][f"{bucket}-{bucket+10}"] += 1
        
        # 记录低分和高分样本
        if score_percent < 50:
            analysis["low_score_samples"].append({
                "id": pred.get("_id"),
                "question": pred.get("input", "")[:100],
                "prediction": prediction[:100],
                "answers": answers,
                "score": score_percent,
            })
        elif score_percent >= 90:
            analysis["high_score_samples"].append({
                "id": pred.get("_id"),
                "question": pred.get("input", "")[:100],
                "score": score_percent,
            })
        
        # 上下文长度统计
        ctx_len = pred.get("context_length", pred.get("length", 0))
        if ctx_len:
            total_context_length += ctx_len
            analysis["context_length_stats"]["min"] = min(
                analysis["context_length_stats"]["min"], ctx_len
            )
            analysis["context_length_stats"]["max"] = max(
                analysis["context_length_stats"]["max"], ctx_len
            )
    
    # 计算统计数据
    if analysis["total_samples"] > 0:
        analysis["avg_f1_score"] = sum(analysis["scores"]) / len(analysis["scores"])
        analysis["context_length_stats"]["avg"] = total_context_length / analysis["total_samples"]
    else:
        analysis["avg_f1_score"] = 0
        analysis["context_length_stats"]["avg"] = 0
    
    # 排序
    analysis["low_score_samples"].sort(key=lambda x: x["score"])
    analysis["high_score_samples"].sort(key=lambda x: x["score"], reverse=True)
    
    return analysis


def print_analysis(analysis: Dict):
    """打印分析结果"""
    
    print("\n" + "="*80)
    print("UCM 稀疏化准确度测试 - 结果分析")
    print("="*80)
    
    # 基本统计
    print("\n📊 基本统计")
    print("-" * 80)
    print(f"  总样本数: {analysis['total_samples']}")
    print(f"  平均 F1 Score: {analysis['avg_f1_score']:.2f}%")
    print(f"  空预测数: {analysis['empty_predictions']}")
    print(f"  空预测率: {analysis['empty_predictions']/analysis['total_samples']*100:.2f}%")
    
    # 上下文长度统计
    print("\n📏 上下文长度统计")
    print("-" * 80)
    ctx_stats = analysis["context_length_stats"]
    print(f"  最小长度: {ctx_stats['min']} 字符")
    print(f"  最大长度: {ctx_stats['max']} 字符")
    print(f"  平均长度: {ctx_stats['avg']:.0f} 字符")
    
    # 分数分布
    print("\n📈 F1 Score 分布")
    print("-" * 80)
    dist = analysis["score_distribution"]
    for bucket in ["0-10", "10-20", "20-30", "30-40", "40-50", 
                   "50-60", "60-70", "70-80", "80-90", "90-100"]:
        count = dist.get(bucket, 0)
        percentage = count / analysis['total_samples'] * 100 if analysis['total_samples'] > 0 else 0
        bar = "█" * int(percentage / 2)
        print(f"  {bucket}%: {count:3d} 样本 ({percentage:5.1f}%) {bar}")
    
    # 低分样本
    print(f"\n⚠️  低分样本（F1 < 50%, 共 {len(analysis['low_score_samples'])} 个）")
    print("-" * 80)
    for i, sample in enumerate(analysis["low_score_samples"][:5], 1):
        print(f"\n  {i}. F1 Score: {sample['score']:.2f}%")
        print(f"     问题: {sample['question']}")
        print(f"     预测: {sample['prediction']}")
        print(f"     答案: {sample['answers']}")
    
    if len(analysis["low_score_samples"]) > 5:
        print(f"\n  ... 还有 {len(analysis['low_score_samples']) - 5} 个低分样本")
    
    # 高分样本
    print(f"\n✨ 高分样本（F1 >= 90%, 共 {len(analysis['high_score_samples'])} 个）")
    print("-" * 80)
    for i, sample in enumerate(analysis["high_score_samples"][:3], 1):
        print(f"  {i}. F1 Score: {sample['score']:.2f}% - {sample['question']}")
    
    # 结论
    print("\n" + "="*80)
    print("📊 测试结论")
    print("="*80)
    
    avg_score = analysis["avg_f1_score"]
    
    if avg_score >= 80:
        level = "优秀 ✨"
        comment = "稀疏化方法对准确度影响很小，性能优异！"
    elif avg_score >= 60:
        level = "良好 ✓"
        comment = "稀疏化方法保持了较好的准确度，符合预期。"
    elif avg_score >= 40:
        level = "一般 ⚠️"
        comment = "稀疏化方法对准确度有一定影响，建议优化。"
    else:
        level = "较差 ✗"
        comment = "稀疏化方法严重影响准确度，需要重新设计。"
    
    print(f"  平均 F1 Score: {avg_score:.2f}%")
    print(f"  评级: {level}")
    print(f"  评语: {comment}")
    print("="*80)


def main():
    """主函数"""
    
    # 默认预测文件路径
    predictions_file = "results/sparse_accuracy_predictions.jsonl"
    
    # 从命令行参数获取路径（如果提供）
    if len(sys.argv) > 1:
        predictions_file = sys.argv[1]
    
    # 加载预测结果
    predictions = load_predictions(predictions_file)
    
    if not predictions:
        print("✗ No predictions found. Please run the test first:")
        print("  python test_sparse_method_accuracy.py")
        sys.exit(1)
    
    # 分析结果
    analysis = analyze_predictions(predictions)
    
    # 打印分析
    print_analysis(analysis)
    
    # 保存分析结果
    analysis_file = TEST_ROOT / "results/sparse_accuracy_analysis.json"
    analysis_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 转换为可序列化的格式
    save_data = {
        "total_samples": analysis["total_samples"],
        "avg_f1_score": analysis["avg_f1_score"],
        "empty_predictions": analysis["empty_predictions"],
        "score_distribution": dict(analysis["score_distribution"]),
        "context_length_stats": analysis["context_length_stats"],
        "num_low_score_samples": len(analysis["low_score_samples"]),
        "num_high_score_samples": len(analysis["high_score_samples"]),
    }
    
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n分析结果已保存到: {analysis_file}\n")


if __name__ == "__main__":
    main()

