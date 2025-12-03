# UCM Sparse Method Accuracy Evaluation

## 概述

该模块用于评估 UCM 稀疏化方法对模型准确度的影响，使用 F1 Score 作为评估指标。

## 目录结构

```
ucm_sparse_eval/
├── data/
│   └── multifieldqa_zh.jsonl      # 中文多领域 QA 数据集
├── eval_f1_score.py                # F1 Score 评估脚本
└── README.md                       # 本文档
```

## 数据集格式

`multifieldqa_zh.jsonl` 中每行是一个 JSON 对象：

```json
{
  "input": "问题文本",
  "context": "上下文文本",
  "answers": ["参考答案"],
  "length": 上下文长度,
  "dataset": "multifieldqa_zh",
  "language": "zh",
  "_id": "唯一标识"
}
```

## 评估指标

### F1 Score (qa_f1_zh_score)

用于中文 QA 任务的 F1 分数计算：

1. **分词**：使用 jieba 对预测答案和参考答案进行分词
2. **标准化**：去除标点符号、统一大小写、去除空格
3. **计算**：
   - Precision = 正确 tokens / 预测 tokens
   - Recall = 正确 tokens / 参考 tokens  
   - F1 = 2 × (Precision × Recall) / (Precision + Recall)
4. **汇总**：对所有样本的 F1 求平均，乘以 100

## 使用方法

### 方法 1：通过 Pytest 运行

```bash
# 运行稀疏化准确度测试
cd /home/externals/wangwenxin21/caz/unified-cache-management/test
pytest test_sparse_method_accuracy.py -v

# 只运行特定标记的测试
pytest test_sparse_method_accuracy.py --feature=sparse_accuracy

# 生成 HTML 报告
pytest test_sparse_method_accuracy.py --html=results/reports/accuracy_report.html
```

### 方法 2：直接运行脚本

```bash
cd /home/externals/wangwenxin21/caz/unified-cache-management/test
python test_sparse_method_accuracy.py
```

### 方法 3：单独使用评估脚本

```bash
cd /home/externals/wangwenxin21/caz/unified-cache-management/test/common/ucm_sparse_eval

# 评估预测结果
python eval_f1_score.py \
  --model qwen3 \
  --answer ../../results/sparse_accuracy_predictions.jsonl \
  --dataset multifieldqa_zh
```

## 配置说明

在 `test/config.yaml` 中配置测试参数：

```yaml
sparse_accuracy_test:
  # 数据集配置
  dataset:
    name: "multifieldqa_zh"
    path: "common/ucm_sparse_eval/data/multifieldqa_zh.jsonl"
    max_samples: null  # null=全部，可设置数字限制样本数
  
  # 模型推理配置
  inference:
    model: "qwen3"
    server_url: "http://141.111.32.70:9382"
    max_tokens: 512
    temperature: 0.7
    top_p: 0.9
  
  # 评估配置
  evaluation:
    expected_f1_score: 60.0   # F1 分数阈值
  
  # 输出配置
  output:
    predictions_file: "results/sparse_accuracy_predictions.jsonl"
    evaluation_file: "results/sparse_accuracy_evaluation.json"
```

## 测试流程

1. **加载数据集**：从 JSONL 文件加载测试数据
2. **模型推理**：调用 LLM API 对每个样本进行推理
3. **收集结果**：保存模型的预测答案
4. **计算 F1 Score**：使用 `eval_f1_score.py` 计算 F1 分数
5. **断言验证**：验证 F1 Score 是否达到预期阈值（默认 60）
6. **保存结果**：将预测和评估结果保存到文件

## 输出文件

### predictions_file (JSONL 格式)

每行包含一个预测结果：

```json
{
  "_id": "样本ID",
  "input": "问题",
  "context_length": 上下文长度,
  "answers": ["参考答案"],
  "pred": "模型预测答案",
  "dataset": "multifieldqa_zh"
}
```

### evaluation_file (JSON 格式)

评估结果摘要：

```json
{
  "dataset": "multifieldqa_zh",
  "total_samples": 200,
  "f1_score": 65.32,
  "metric": "qa_f1_zh_score"
}
```

## 依赖包

```bash
pip install pytest
pip install requests
pip install jieba
pip install fuzzywuzzy
pip install rouge
```

## 支持的数据集

`eval_f1_score.py` 支持多种数据集类型：

- **中文 QA**：`multifieldqa_zh`, `clongeval`, `dureader`, `vcsum`
- **英文 QA**：`narrativeqa`, `qasper`, `multifieldqa_en`, `hotpotqa`, etc.
- **分类任务**：`trec`, `lsht`
- **代码相似度**：`lcc`, `repobench-p`

## 注意事项

1. 确保 LLM 服务器可访问
2. 数据集文件路径正确
3. 有足够的磁盘空间保存结果
4. 网络连接稳定（推理可能需要较长时间）

## 示例结果

```
============================================================
Evaluation Results:
============================================================
  Dataset: multifieldqa_zh
  Total Samples: 200
  F1 Score: 65.32
  Metric: qa_f1_zh_score
============================================================

✓ Test passed! F1 Score 65.32 >= 60.00
```

## 疑难解答

**Q: 测试失败，F1 Score 过低怎么办？**
A: 检查模型推理质量、调整 temperature/top_p 参数、检查数据集质量。

**Q: 推理超时怎么办？**
A: 增加 `timeout` 参数值，或减少 `max_tokens`。

**Q: 如何只测试部分样本？**
A: 在配置中设置 `max_samples: 50`。

**Q: 如何更改 F1 阈值？**
A: 修改配置中的 `expected_f1_score` 值。

