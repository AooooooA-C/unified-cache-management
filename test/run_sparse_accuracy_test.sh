#!/bin/bash
# UCM Sparse Method Accuracy Test Runner

set -e

# 切换到测试目录
cd "$(dirname "$0")"

echo "=========================================="
echo "UCM Sparse Method Accuracy Test"
echo "=========================================="
echo ""

# 检查依赖
echo "[1/4] Checking dependencies..."
python3 -c "import pytest, requests, jieba, rouge" 2>/dev/null || {
    echo "Missing dependencies. Installing..."
    pip install pytest requests jieba fuzzywuzzy python-rouge -q
}
echo "✓ Dependencies OK"
echo ""

# 检查数据集
echo "[2/4] Checking dataset..."
DATASET_PATH="common/ucm_sparse_eval/data/multifieldqa_zh.jsonl"
if [ ! -f "$DATASET_PATH" ]; then
    echo "✗ Dataset not found: $DATASET_PATH"
    exit 1
fi
SAMPLE_COUNT=$(wc -l < "$DATASET_PATH")
echo "✓ Dataset found: $SAMPLE_COUNT samples"
echo ""

# 检查 LLM 服务器
echo "[3/4] Checking LLM server..."
SERVER_URL=$(python3 -c "import yaml; print(yaml.safe_load(open('config.yaml'))['sparse_accuracy_test']['inference']['server_url'])")
curl -s --connect-timeout 5 "$SERVER_URL/v1/models" > /dev/null 2>&1 && {
    echo "✓ LLM server is accessible: $SERVER_URL"
} || {
    echo "⚠ Warning: Cannot connect to LLM server: $SERVER_URL"
    echo "  Please check if the server is running"
}
echo ""

# 运行测试
echo "[4/4] Running accuracy test..."
echo "=========================================="
echo ""

# 可以选择运行方式
if [ "$1" == "--pytest" ]; then
    # 通过 pytest 运行
    pytest suites/E2E/test_sparse_method_accuracy.py -v --tb=short
else
    # 直接运行 Python 脚本
    python3 suites/E2E/test_sparse_method_accuracy.py
fi

echo ""
echo "=========================================="
echo "Test completed!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - Predictions: results/sparse_accuracy_predictions.jsonl"
echo "  - Evaluation: results/sparse_accuracy_evaluation.json"
echo ""

