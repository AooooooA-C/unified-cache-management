/**
 * @file asu_benchmark_test.cc
 * @brief ASU (Async Storage Unit) 基准测试
 *
 * 本文件是 ASU 系统的基准测试入口，测试以下功能：
 * - 数据完整性：Store/Load 数据一致性
 * - 查询准确性：Query 存在性检查
 * - 延迟测试：单条操作延迟
 * - 吞吐量测试：批量操作吞吐量
 * - 并发测试：多线程竞争
 * - 压力测试：大规模数据
 * - RDMA 统计：Mock 统计功能验证
 *
 * @details 架构调用关系图:
 *
 * ┌─────────────────────────────────────────────────────────────────────────────┐
 * │                          asu_benchmark_test.cc (本文件)                      │
 * │                              [测试入口层]                                     │
 * │                                                                              │
 * │  ┌───────────────────────────────────────────────────────────────────────┐  │
 * │  │                        SetUp() / TearDown()                            │  │
 * │  │                                                                        │  │
 * │  │  SetUp():                                                              │  │
 * │  │    client_ = CreateBenchClient()                                       │  │
 * │  │            │                                                           │  │
 * │  │            ▼                                                           │  │
 * │  │    CreateAsuClient(factory)                                            │  │
 * │  │            │                                                           │  │
 * │  │            ▼                                                           │  │
 * │  │    client_->Init(config)                                               │  │
 * │  │                                                                        │  │
 * │  │  TearDown():                                                           │  │
 * │  │    client_->Shutdown()                                                 │  │
 * │  │    验证关闭后调用失败                                                   │  │
 * │  └───────────────────────────────────────────────────────────────────────┘  │
 * │                                                                              │
 * │  ┌───────────────────────────────────────────────────────────────────────┐  │
 * │  │                     工厂函数调用链                                      │  │
 * │  │                                                                        │  │
 * │  │  CreateBenchClient()                                                   │  │
 * │  │  └─► factory = []() {                                                  │  │
 * │  │        return std::make_unique<AsuTransportImpl>(                      │  │
 * │  │            CreateMemoryIoBackend()    ◄─── 调用 memory_io_backend.cpp  │  │
 * │  │        );                                                              │  │
 * │  │      }                                                                 │  │
 * │  │  └─► return CreateAsuClient(factory)                                   │  │
 * │  │                                                                        │  │
 * │  │  CreateRdmaMockBenchClient(io_stats, options)                          │  │
 * │  │  └─► factory = [io_stats, options]() {                                 │  │
 * │  │        return std::make_unique<AsuTransportImpl>(                      │  │
 * │  │            CreateMemoryIoBackend(options, io_stats)                    │  │
 * │  │        );                              ◄─── 带 Mock 配置               │  │
 * │  │      }                                                                 │  │
 * │  │  └─► return CreateAsuClient(factory)                                   │  │
 * │  └───────────────────────────────────────────────────────────────────────┘  │
 * │                                                                              │
 * │  ┌───────────────────────────────────────────────────────────────────────┐  │
 * │  │                     测试调用流程                                        │  │
 * │  │                                                                        │  │
 * │  │  TEST_F(..., ...) {                                                    │  │
 * │  │    client_->StoreAsync(entries, task_id)                               │  │
 * │  │    │                                                                   │  │
 * │  │    ▼                                                                   │  │
 * │  │    AsuClient::StoreAsync()                                             │  │
 * │  │    │                                                                   │  │
 * │  │    ▼                                                                   │  │
 * │  │    AsuTransportImpl::SubmitTask()                                      │  │
 * │  │    │                                                                   │  │
 * │  │    ▼                                                                   │  │
 * │  │    MemoryIoBackend::Store()        ◄─── memory_io_backend.cpp 实现     │  │
 * │  │    │                                                                   │  │
 * │  │    ├─► AccountStoreTransfer()      (记录统计到 io_stats)               │  │
 * │  │    ├─► MaybeDelay()                (模拟 RDMA 延迟)                    │  │
 * │  │    └─► store_[key] = data          (内存存储)                          │  │
 * │  │                                                                        │  │
 * │  │    WaitWithTimeout(*client_, task_id, ...)                             │  │
 * │  │    │                                                                   │  │
 * │  │    ▼                                                                   │  │
 * │  │    AsuClient::Wait()                                                   │  │
 * │  │    │                                                                   │  │
 * │  │    ▼                                                                   │  │
 * │  │    获取 TaskResult                                                      │  │
 * │  │    │                                                                   │  │
 * │  │    ▼                                                                   │  │
 * │  │    验证 result.status / result.entry_status                            │  │
 * │  │  }                                                                     │  │
 * │  └───────────────────────────────────────────────────────────────────────┘  │
 * │                                                                              │
 * └─────────────────────────────────────────────────────────────────────────────┘
 *
 * @details 文件依赖关系:
 *
 * ┌─────────────────────┐
 * │ asu_benchmark_test  │ (本文件)
 * │      .cc            │
 * └─────────┬───────────┘
 *           │
 *           │ #include
 *           │
 *           ▼
 * ┌─────────────────────┐      ┌─────────────────────┐
 * │  memory_io_backend  │◄─────│    io_backend.h     │
 * │      .h             │ 继承 │     (抽象接口)      │
 * └─────────┬───────────┘      └─────────────────────┘
 *           │
 *           │ #include
 *           │
 *           ▼
 * ┌─────────────────────┐
 * │  memory_io_backend  │
 * │      .cpp           │ (实现)
 * └─────────────────────┘
 */

#include "asu_client/asu_client.h"
#include "io_backend.h"
#include "memory_io_backend.h"
#include "asu_transport_impl.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <future>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <sstream>
#include <thread>
#include <vector>

namespace UC::ASU {
namespace {

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double>;

/**
 * @struct BenchResult
 * @brief 基准测试结果
 *
 * 记录一次基准测试的性能指标：
 * - total_sec: 总耗时（秒）
 * - ops: 操作次数
 * - avg/min/max_lat_ms: 平均/最小/最大延迟（毫秒）
 * - throughput_ops_sec: 吞吐量（ops/秒）
 */
struct BenchResult {
    double total_sec{0};
    std::size_t ops{0};
    double avg_lat_ms{0};
    double min_lat_ms{0};
    double max_lat_ms{0};
    double throughput_ops_sec{0};
};

/**
 * @brief 计算基准测试结果
 *
 * @details 从延迟数组计算各项统计指标:
 * - avg = sum / count
 * - min = 最小值
 * - max = 最大值
 * - throughput = ops / total_sec
 *
 * @param latencies_ms 延迟数组（毫秒）
 * @return BenchResult 计算结果
 */
BenchResult ComputeBench(const std::vector<double>& latencies_ms)
{
    BenchResult r;
    r.ops = latencies_ms.size();
    r.total_sec = std::accumulate(latencies_ms.begin(), latencies_ms.end(), 0.0) / 1000.0;
    r.avg_lat_ms = r.ops > 0 ? std::accumulate(latencies_ms.begin(), latencies_ms.end(), 0.0) / r.ops : 0;
    r.min_lat_ms = r.ops > 0 ? *std::min_element(latencies_ms.begin(), latencies_ms.end()) : 0;
    r.max_lat_ms = r.ops > 0 ? *std::max_element(latencies_ms.begin(), latencies_ms.end()) : 0;
    r.throughput_ops_sec = r.total_sec > 0 ? r.ops / r.total_sec : 0;
    return r;
}

/**
 * @brief 打印基准测试结果
 *
 * @param label 测试标签
 * @param r 测试结果
 */
void PrintBench(const char* label, const BenchResult& r)
{
    std::printf("[BENCH] %s: ops=%zu  total=%.3fms  avg=%.3fms  min=%.3fms  max=%.3fms  "
                "throughput=%.1f ops/sec\n",
                label, r.ops, r.total_sec * 1000, r.avg_lat_ms, r.min_lat_ms, r.max_lat_ms,
                r.throughput_ops_sec);
}

/**
 * @brief 创建基准测试客户端（无 Mock 功能）
 *
 * @details 工厂函数模式:
 * ```
 * CreateBenchClient()
 *     │
 *     ├─► 定义 factory lambda:
 *     │   factory = []() -> std::unique_ptr<AsuTransport> {
 *     │       return std::make_unique<AsuTransportImpl>(
 *     │           CreateMemoryIoBackend()    ◄─── 默认配置，无 Mock
 *     │       );
 *     │   }
 *     │
 *     ├─► 调用 CreateAsuClient(factory)
 *     │   │
 *     │   └─► AsuClient 保存 factory
 *     │       等待 Init() 时调用 factory 创建 Transport
 *     │
 *     └─► 返回 std::unique_ptr<AsuClient>
 * ```
 *
 * @note Lambda 延迟执行模式:
 * factory lambda 被 CreateAsuClient 保存，但不立即执行。
 * 只有在 client_->Init() 时才调用 factory 创建 AsuTransportImpl。
 * 这是为了支持配置驱动的初始化。
 *
 * @return std::unique_ptr<AsuClient> 测试客户端
 */
std::unique_ptr<AsuClient> CreateBenchClient()
{
    auto factory = []() -> std::unique_ptr<AsuTransport> {
        return std::make_unique<AsuTransportImpl>(CreateMemoryIoBackend());
    };
    return CreateAsuClient(factory);
}

/**
 * @brief 创建带 RDMA Mock 功能的基准测试客户端
 *
 * @details 与 CreateBenchClient() 类似，但使用带 Mock 配置的 Backend:
 * ```
 * CreateRdmaMockBenchClient(io_stats, options)
 *     │
 *     ├─► 定义 factory lambda (捕获 io_stats 和 options):
 *     │   factory = [io_stats, options]() -> std::unique_ptr<AsuTransport> {
 *     │       return std::make_unique<AsuTransportImpl>(
 *     │           CreateMemoryIoBackend(options, io_stats)    ◄─── 带 Mock
 *     │       );
 *     │   }
 *     │
 *     ├─► Lambda 捕获说明:
 *     │   - io_stats: shared_ptr<MockIoStats>，按值捕获（拷贝 shared_ptr）
 *     │   - options: MemoryIoBackendOptions，按值捕获（拷贝配置）
 *     │
 *     └─► 返回 std::unique_ptr<AsuClient>
 * ```
 *
 * @note Lambda 捕获策略:
 * - io_stats 是 shared_ptr，按值捕获确保 lambda 持有引用
 * - options 是配置结构体，按值捕获确保配置不变
 *
 * @param io_stats 统计对象（用于验证传输次数和字节数）
 * @param options Mock 配置（延迟、带宽、统计等）
 * @return std::unique_ptr<AsuClient> 测试客户端
 *
 * @example RDMA 统计验证测试:
 * @code
 * auto io_stats = std::make_shared<MockIoStats>();
 * MemoryIoBackendOptions options;
 * options.enable_rdma_stats = true;
 * options.rdma_per_entry = true;
 *
 * auto client = CreateRdmaMockBenchClient(io_stats, options);
 * client->Init(config);
 *
 * // 执行 Store 操作
 * client->StoreAsync(entries, task_id);
 * WaitWithTimeout(*client, task_id, ...);
 *
 * // 验证统计
 * EXPECT_EQ(io_stats->store_transfers.load(), expected_count);
 * EXPECT_EQ(io_stats->store_bytes.load(), expected_bytes);
 * @endcode
 */
std::unique_ptr<AsuClient> CreateRdmaMockBenchClient(
    const std::shared_ptr<MockIoStats>& io_stats,
    MemoryIoBackendOptions options)
{
    auto factory = [io_stats, options]() -> std::unique_ptr<AsuTransport> {
        return std::make_unique<AsuTransportImpl>(
            CreateMemoryIoBackend(options, io_stats));
    };
    return CreateAsuClient(factory);
}

/**
 * @brief 创建基准测试配置
 *
 * @details 配置内容:
 * - client_id: "asu-bench-client"
 * - default_wait_timeout_ms: 5000ms
 * - Transport 配置:
 *   - asu_name: "asu-bench-{i}"
 *   - asu_id: 2001 + i
 *   - max_inflight_tasks: 4096
 *   - query_timeout_ms: 5000ms
 *
 * @param num_transports Transport 数量（默认 1）
 * @return AsuClientConfig 测试配置
 */
AsuClientConfig MakeBenchConfig(std::size_t num_transports = 1)
{
    AsuClientConfig config;
    config.client_id = "asu-bench-client";
    config.default_wait_timeout_ms = 5000;
    for (std::size_t i = 0; i < num_transports; ++i) {
        TransportConfig tc;
        tc.asu_name = "asu-bench-" + std::to_string(i);
        tc.asu_id = 2001 + static_cast<AsuId>(i);
        tc.max_inflight_tasks = 4096;
        tc.query_timeout_ms = 5000;
        config.transport_configs.push_back(tc);
    }
    return config;
}

/**
 * @brief 创建 KV Buffer 数据
 *
 * @details 创建指定数量的 KVBuffer，用于测试:
 * 1. 分配 payloads 数组（存储实际数据）
 * 2. 为每个 payload 分配指定大小的随机数据
 * 3. 创建 MemoryRegion (HOST 类型)
 * 4. 创建 Buffer + KVBuffer
 *
 * @note payloads 参数说明:
 * payloads 是 vector<vector<uint8_t>> 的引用，数据存储在外部。
 * 这是因为 KVBuffer 只持有指针，需要 payloads 持有实际数据。
 * payloads 必须在测试期间保持有效。
 *
 * @param payloads [out] 存储实际数据的数组
 * @param count KV 数量
 * @param payload_size 每个 payload 的大小
 * @param key_prefix Key 前缀
 * @return std::vector<KVBuffer> KVBuffer 列表
 */
std::vector<KVBuffer> MakeKVBuffers(std::vector<std::vector<std::uint8_t>>& payloads,
                                    std::size_t count, std::size_t payload_size,
                                    const std::string& key_prefix = "key-")
{
    payloads.resize(count);
    std::vector<KVBuffer> entries;
    entries.reserve(count);
    std::mt19937 rng(42);
    for (std::size_t i = 0; i < count; ++i) {
        payloads[i].resize(payload_size);
        std::generate(payloads[i].begin(), payloads[i].end(), rng);
        MemoryRegion region;
        region.memory_type = MemoryType::HOST;
        region.addr = reinterpret_cast<std::uint64_t>(payloads[i].data());
        region.size = payloads[i].size();
        Buffer buffer;
        buffer.region = region;
        entries.push_back(KVBuffer{key_prefix + std::to_string(i), buffer});
    }
    return entries;
}

/**
 * @brief 从 KVBuffer 提取 Key 列表
 *
 * @param entries KVBuffer 列表
 * @return std::vector<CacheKey> Key 列表
 */
std::vector<CacheKey> ExtractKeys(const std::vector<KVBuffer>& entries)
{
    std::vector<CacheKey> keys;
    keys.reserve(entries.size());
    for (const auto& e : entries) { keys.push_back(e.key); }
    return keys;
}

/**
 * @brief 等待任务完成（带超时）
 *
 * @details 调用流程:
 * ```
 * WaitWithTimeout(client, task_id, timeout_ms, result)
 *     │
 *     ├─► client.Wait(task_id, timeout_ms, result)
 *     │   - 返回 Status（Wait 操作本身是否成功）
 *     │
 *     ├─► if !status.ok(): return status    (Wait 失败)
 *     │
 *     ├─► if !result.status.ok(): return result.status  (任务整体失败)
 *     │
 *     └─► for each entry_status:
 *         if !es.ok(): return es            (某个条目失败)
 *
 * return Status::OK()
 * ```
 *
 * @param client AsuClient 实例
 * @param task_id 任务 ID
 * @param timeout_ms 超时时间（毫秒）
 * @param result [out] 任务结果
 * @return Status 最终状态（聚合所有错误）
 */
Status WaitWithTimeout(AsuClient& client, TaskId task_id, std::uint64_t timeout_ms,
                       TaskResult& result)
{
    auto status = client.Wait(task_id, timeout_ms, result);
    if (!status.ok()) { return status; }
    if (!result.status.ok()) { return result.status; }
    for (const auto& es : result.entry_status) {
        if (!es.ok()) { return es; }
    }
    return Status::OK();
}

/**
 * @struct StressTaskData
 * @brief 压力测试任务数据
 *
 * 用于多线程压力测试，每个线程持有独立的:
 * - payloads: 实际数据数组
 * - entries: KVBuffer 列表
 * - keys: Key 列表
 * - task_id: 任务 ID
 */
struct StressTaskData {
    std::vector<std::vector<std::uint8_t>> payloads;
    std::vector<KVBuffer> entries;
    std::vector<CacheKey> keys;
    TaskId task_id{kInvalidTaskId};
};

/**
 * @brief 创建 Store 压力测试任务数据
 *
 * @details 创建指定数量和大小 KV 数据，用于 Store 测试:
 * - payload 内容随机生成（基于 task_index + 12345 的种子）
 * - Key 格式: "{key_prefix}{task_index}-{i}"
 *
 * @param task_index 任务索引（用于区分不同线程）
 * @param entries_per_task 每个 Task 的 Entry 数量
 * @param payload_size Payload 大小
 * @param key_prefix Key 前缀
 * @return StressTaskData 任务数据
 */
StressTaskData MakeStressStoreTask(std::size_t task_index, std::size_t entries_per_task,
                                   std::size_t payload_size, const std::string& key_prefix)
{
    StressTaskData data;
    data.payloads.resize(entries_per_task);
    data.entries.reserve(entries_per_task);
    data.keys.reserve(entries_per_task);

    std::mt19937 rng(static_cast<std::uint32_t>(task_index + 12345));

    for (std::size_t i = 0; i < entries_per_task; ++i) {
        data.payloads[i].resize(payload_size);
        std::generate(data.payloads[i].begin(), data.payloads[i].end(), rng);

        auto key = key_prefix + std::to_string(task_index) + "-" + std::to_string(i);

        MemoryRegion region;
        region.memory_type = MemoryType::HOST;
        region.addr = reinterpret_cast<std::uint64_t>(data.payloads[i].data());
        region.size = data.payloads[i].size();

        Buffer buffer;
        buffer.region = region;

        data.entries.push_back(KVBuffer{key, buffer});
        data.keys.push_back(key);
    }

    return data;
}

/**
 * @brief 创建 Load 压力测试任务数据
 *
 * @details 创建预分配的 Buffer，用于 Load 测试:
 * - Buffer 内容初始化为 0
 * - Buffer 大小等于 payload_size
 *
 * @param keys Key 列表（要加载的 Key）
 * @param payload_size 预分配 Buffer 大小
 * @return StressTaskData 任务数据
 */
StressTaskData MakeStressLoadTask(const std::vector<CacheKey>& keys, std::size_t payload_size)
{
    StressTaskData data;
    data.keys = keys;
    data.payloads.resize(keys.size());
    data.entries.reserve(keys.size());

    for (std::size_t i = 0; i < keys.size(); ++i) {
        data.payloads[i].assign(payload_size, 0);

        MemoryRegion region;
        region.memory_type = MemoryType::HOST;
        region.addr = reinterpret_cast<std::uint64_t>(data.payloads[i].data());
        region.size = data.payloads[i].size();

        Buffer buffer;
        buffer.region = region;

        data.entries.push_back(KVBuffer{keys[i], buffer});
    }

    return data;
}

}  // namespace

/**
 * @class AsuBenchmarkTest
 * @brief ASU 基准测试基类
 *
 * 使用 Google Test 的 TEST_F 宏进行测试。
 * SetUp/TearDown 管理 Client 的生命周期。
 *
 * @details 生命周期:
 * ```
 * TEST_F 开始
 *     │
 *     ├─► SetUp()
 *     │   ├─► CreateBenchClient()
 *     │   ├─► client_->Init(config)
 *     │   └─► ASSERT 成功
 *     │
 *     ├─► 测试代码执行
 *     │   ├─► client_->StoreAsync/LoadAsync/Query/Delete
 *     │   ├─► WaitWithTimeout
 *     │   └─► EXPECT 验证结果
 *     │
 *     └─► TearDown()
 *     │   ├─► client_->Shutdown()
 *     │   └─► 验证关闭后调用失败
 *     │
 * TEST_F 结束
 * ```
 */
class AsuBenchmarkTest : public ::testing::Test {
protected:
    /**
     * @brief 测试前初始化
     *
     * @details 初始化步骤:
     * 1. CreateBenchClient() 创建 Client
     * 2. MakeBenchConfig() 创建配置
     * 3. client_->Init(config) 初始化 Client
     *
     * @note Init 时才调用 factory 创建 Transport
     */
    void SetUp() override
    {
        client_ = CreateBenchClient();
        ASSERT_NE(client_, nullptr);
        auto status = client_->Init(MakeBenchConfig());
        ASSERT_TRUE(status.ok()) << status.message;
    }

    /**
     * @brief 测试后清理
     *
     * @details 清理步骤:
     * 1. client_->Shutdown() 关闭 Client
     * 2. 尝试 LoadAsync 验证关闭后调用失败
     *
     * @note 验证 NOT_INITIALIZED 状态:
     * Shutdown 后调用异步操作应返回 NOT_INITIALIZED。
     */
    void TearDown() override
    {
        if (client_) {
            auto status = client_->Shutdown();
            ASSERT_TRUE(status.ok()) << status.message;
        }
        TaskId post_shutdown_task{kInvalidTaskId};
        std::vector<std::uint8_t> dummy_payload(1, 0);
        MemoryRegion region;
        region.memory_type = MemoryType::HOST;
        region.addr = reinterpret_cast<std::uint64_t>(dummy_payload.data());
        region.size = 1;
        Buffer buffer;
        buffer.region = region;
        KVBuffer entry{"shutdown-check", buffer};
        auto s = client_->LoadAsync({entry}, post_shutdown_task);
        ASSERT_EQ(s.code, StatusCode::NOT_INITIALIZED);
    }

    std::unique_ptr<AsuClient> client_;
};

/**
 * @brief 数据完整性测试 - Store/Load 验证
 *
 * @details 测试流程:
 * ```
 * DataIntegrity_StoreLoadVerify
 *     │
 *     ├─► [阶段1: Store]
 *     │   ├─► MakeKVBuffers(64 entries, 4096 bytes each)
 *     │   ├─► StoreAsync(entries, store_id)
 *     │   ├─► WaitWithTimeout(store_id)
 *     │   └─► ASSERT 成功
 *     │
 *     ├─► [阶段2: Load]
 *     │   ├─► 创建 load_payloads (预分配 4096 bytes)
 *     │   ├─► LoadAsync(load_entries, load_id)
 *     │   ├─► WaitWithTimeout(load_id)
 *     │   └─► ASSERT 成功
 *     │
 *     └─► [阶段3: 验证]
 *     │   for each entry:
 *     │     EXPECT size 匹配
 *     │     EXPECT memcmp == 0 (数据一致)
 * ```
 *
 * @note 测试目的:
 * - 验证 Store 后数据完整存储
 * - 验证 Load 后数据完整恢复
 * - 验证数据一致性（bit-by-bit 比较）
 */
TEST_F(AsuBenchmarkTest, DataIntegrity_StoreLoadVerify)
{
    constexpr std::size_t kCount = 64;
    constexpr std::size_t kPayloadSize = 4096;

    std::vector<std::vector<std::uint8_t>> original_payloads;
    auto entries = MakeKVBuffers(original_payloads, kCount, kPayloadSize, "integrity-");
    auto keys = ExtractKeys(entries);

    TaskId store_id{kInvalidTaskId};
    auto status = client_->StoreAsync(entries, store_id);
    ASSERT_TRUE(status.ok()) << status.message;
    TaskResult store_result;
    status = WaitWithTimeout(*client_, store_id, 5000, store_result);
    ASSERT_TRUE(status.ok()) << status.message;

    std::vector<std::vector<std::uint8_t>> load_payloads(kCount);
    std::vector<KVBuffer> load_entries;
    load_entries.reserve(kCount);
    for (std::size_t i = 0; i < kCount; ++i) {
        load_payloads[i].resize(kPayloadSize, 0);
        MemoryRegion region;
        region.memory_type = MemoryType::HOST;
        region.addr = reinterpret_cast<std::uint64_t>(load_payloads[i].data());
        region.size = kPayloadSize;
        Buffer buffer;
        buffer.region = region;
        load_entries.push_back(KVBuffer{keys[i], buffer});
    }

    TaskId load_id{kInvalidTaskId};
    status = client_->LoadAsync(load_entries, load_id);
    ASSERT_TRUE(status.ok()) << status.message;
    TaskResult load_result;
    status = WaitWithTimeout(*client_, load_id, 5000, load_result);
    ASSERT_TRUE(status.ok()) << status.message;

    for (std::size_t i = 0; i < kCount; ++i) {
        EXPECT_EQ(original_payloads[i].size(), load_payloads[i].size())
            << "size mismatch for key " << keys[i];
        EXPECT_EQ(std::memcmp(original_payloads[i].data(), load_payloads[i].data(),
                              original_payloads[i].size()),
                  0)
            << "data mismatch for key " << keys[i];
    }
}

/**
 * @brief 查询准确性测试 - Store 后 Query
 *
 * @details 测试流程:
 * ```
 * QueryAccuracy_StoreThenQuery
 *     │
 *     ├─► [阶段1: Store]
 *     │   ├─► MakeKVBuffers(32 entries)
 *     │   ├─► StoreAsync + Wait
 *     │
 *     ├─► [阶段2: Query]
 *     │   ├─► stored_keys: 已存储的 Key
 *     │   ├─► missing_keys: 未存储的 Key (query-32..query-39)
 *     │   ├─► mixed_keys: stored_keys + missing_keys
 *     │   ├─► Query(mixed_keys, result)
 *     │
 *     └─► [阶段3: 验证]
 *     │   for stored_keys: EXPECT exists[i] == 1
 *     │   for missing_keys: EXPECT exists[i] == 0
 * ```
 *
 * @note 测试目的:
 * - 验证 Query 正确识别已存在的 Key
 * - 验证 Query 正确识别不存在的 Key
 */
TEST_F(AsuBenchmarkTest, QueryAccuracy_StoreThenQuery)
{
    constexpr std::size_t kCount = 32;
    constexpr std::size_t kPayloadSize = 1024;

    std::vector<std::vector<std::uint8_t>> payloads;
    auto entries = MakeKVBuffers(payloads, kCount, kPayloadSize, "query-");

    TaskId store_id{kInvalidTaskId};
    auto status = client_->StoreAsync(entries, store_id);
    ASSERT_TRUE(status.ok()) << status.message;
    TaskResult store_result;
    status = WaitWithTimeout(*client_, store_id, 5000, store_result);
    ASSERT_TRUE(status.ok()) << status.message;

    std::vector<CacheKey> stored_keys = ExtractKeys(entries);
    std::vector<CacheKey> missing_keys;
    for (std::size_t i = kCount; i < kCount + 8; ++i) {
        missing_keys.push_back("query-" + std::to_string(i));
    }
    std::vector<CacheKey> mixed_keys = stored_keys;
    mixed_keys.insert(mixed_keys.end(), missing_keys.begin(), missing_keys.end());

    QueryOptions opts;
    opts.timeout_ms = 5000;
    QueryResult result;
    status = client_->Query(mixed_keys, opts, result);
    ASSERT_TRUE(status.ok()) << status.message;
    ASSERT_EQ(result.exists.size(), mixed_keys.size());

    for (std::size_t i = 0; i < stored_keys.size(); ++i) {
        EXPECT_EQ(result.exists[i], 1) << "stored key " << stored_keys[i] << " should exist";
    }
    for (std::size_t i = stored_keys.size(); i < mixed_keys.size(); ++i) {
        EXPECT_EQ(result.exists[i], 0) << "missing key " << mixed_keys[i] << " should not exist";
    }
}

/**
 * @brief 延迟测试 - 单条 Store/Load/Query
 *
 * @details 测试流程:
 * ```
 * Latency_SingleStoreLoadQuery
 *     │
 *     ├─► [阶段1: Store 延迟]
 *     │   for each entry (10 entries):
 *     │     ├─► t0 = Clock::now()
 *     │     ├─► StoreAsync({entry}, id)
 *     │     ├─► WaitWithTimeout(id)
 *     │     ├─► t1 = Clock::now()
 *     │     └─► store_lats.push_back(t1 - t0)
 *     │
 *     ├─► [阶段2: Load 延迟]
 *     │   (类似 Store)
 *     │
 *     ├─► [阶段3: Query 延迟]
 *     │   (类似 Store)
 *     │
 *     └─► [阶段4: 打印结果]
 *     │   PrintBench("StoreAsync", ComputeBench(store_lats))
 *     │   PrintBench("LoadAsync", ComputeBench(load_lats))
 *     │   PrintBench("Query", ComputeBench(query_lats))
 * ```
 *
 * @note 测试目的:
 * - 测量单条操作的延迟分布
 * - 提供 avg/min/max 延迟指标
 * - 用于性能基准对比
 */
TEST_F(AsuBenchmarkTest, Latency_SingleStoreLoadQuery)
{
    constexpr std::size_t kCount = 10;
    constexpr std::size_t kPayloadSize = 4096;

    std::vector<std::vector<std::uint8_t>> payloads;
    auto entries = MakeKVBuffers(payloads, kCount, kPayloadSize, "lat-");
    auto keys = ExtractKeys(entries);

    std::vector<double> store_lats;
    std::vector<double> load_lats;
    std::vector<double> query_lats;

    for (std::size_t i = 0; i < kCount; ++i) {
        TaskId id{kInvalidTaskId};
        auto t0 = Clock::now();
        auto status = client_->StoreAsync({entries[i]}, id);
        ASSERT_TRUE(status.ok()) << status.message;
        TaskResult result;
        status = WaitWithTimeout(*client_, id, 5000, result);
        ASSERT_TRUE(status.ok()) << status.message;
        auto t1 = Clock::now();
        store_lats.push_back(Duration(t1 - t0).count() * 1000);
    }

    std::vector<std::vector<std::uint8_t>> load_payloads(kCount);
    std::vector<KVBuffer> load_entries(kCount);
    for (std::size_t i = 0; i < kCount; ++i) {
        load_payloads[i].resize(kPayloadSize, 0);
        MemoryRegion region;
        region.memory_type = MemoryType::HOST;
        region.addr = reinterpret_cast<std::uint64_t>(load_payloads[i].data());
        region.size = kPayloadSize;
        Buffer buffer;
        buffer.region = region;
        load_entries[i] = KVBuffer{keys[i], buffer};
    }

    for (std::size_t i = 0; i < kCount; ++i) {
        TaskId id{kInvalidTaskId};
        auto t0 = Clock::now();
        auto status = client_->LoadAsync({load_entries[i]}, id);
        ASSERT_TRUE(status.ok()) << status.message;
        TaskResult result;
        status = WaitWithTimeout(*client_, id, 5000, result);
        ASSERT_TRUE(status.ok()) << status.message;
        auto t1 = Clock::now();
        load_lats.push_back(Duration(t1 - t0).count() * 1000);
    }

    for (std::size_t i = 0; i < kCount; ++i) {
        QueryOptions opts;
        opts.timeout_ms = 5000;
        QueryResult qresult;
        auto t0 = Clock::now();
        auto status = client_->Query({keys[i]}, opts, qresult);
        ASSERT_TRUE(status.ok()) << status.message;
        auto t1 = Clock::now();
        query_lats.push_back(Duration(t1 - t0).count() * 1000);
    }

    PrintBench("StoreAsync (single entry)", ComputeBench(store_lats));
    PrintBench("LoadAsync  (single entry)", ComputeBench(load_lats));
    PrintBench("Query      (single key)  ", ComputeBench(query_lats));
}

/**
 * @brief 吞吐量测试 - 批量 Store/Load/Query
 *
 * @details 测试流程:
 * ```
 * Throughput_BulkStoreLoad
 *     │
 *     ├─► [阶段1: 批量 Store]
 *     │   ├─► MakeKVBuffers(512 entries, 4096 bytes)
 *     │   ├─► t0 = now
 *     │   ├─► StoreAsync(all entries)
 *     │   ├─► WaitWithTimeout
 *     │   ├─► t1 = now
 *     │   └─► 计算吞吐量: 512 / (t1 - t0)
 *     │
 *     ├─► [阶段2: 批量 Load]
 *     │   (类似 Store)
 *     │
 *     ├─► [阶段3: 批量 Query]
 *     │   (类似 Store)
 *     │
 *     ├─► [阶段4: 打印结果]
 *     │
 *     └─► [阶段5: 验证数据一致性]
 * ```
 *
 * @note 测试目的:
 * - 测量批量操作的吞吐量
 * - 验证批量操作的数据一致性
 */
TEST_F(AsuBenchmarkTest, Throughput_BulkStoreLoad)
{
    constexpr std::size_t kCount = 512;
    constexpr std::size_t kPayloadSize = 4096;

    std::vector<std::vector<std::uint8_t>> payloads;
    auto entries = MakeKVBuffers(payloads, kCount, kPayloadSize, "bulk-");
    auto keys = ExtractKeys(entries);

    auto t0 = Clock::now();
    TaskId store_id{kInvalidTaskId};
    auto status = client_->StoreAsync(entries, store_id);
    ASSERT_TRUE(status.ok()) << status.message;
    TaskResult store_result;
    status = WaitWithTimeout(*client_, store_id, 10000, store_result);
    ASSERT_TRUE(status.ok()) << status.message;
    auto t1 = Clock::now();
    double store_sec = Duration(t1 - t0).count();

    std::vector<std::vector<std::uint8_t>> load_payloads(kCount);
    std::vector<KVBuffer> load_entries(kCount);
    for (std::size_t i = 0; i < kCount; ++i) {
        load_payloads[i].resize(kPayloadSize, 0);
        MemoryRegion region;
        region.memory_type = MemoryType::HOST;
        region.addr = reinterpret_cast<std::uint64_t>(load_payloads[i].data());
        region.size = kPayloadSize;
        Buffer buffer;
        buffer.region = region;
        load_entries[i] = KVBuffer{keys[i], buffer};
    }

    auto t2 = Clock::now();
    TaskId load_id{kInvalidTaskId};
    status = client_->LoadAsync(load_entries, load_id);
    ASSERT_TRUE(status.ok()) << status.message;
    TaskResult load_result;
    status = WaitWithTimeout(*client_, load_id, 10000, load_result);
    ASSERT_TRUE(status.ok()) << status.message;
    auto t3 = Clock::now();
    double load_sec = Duration(t3 - t2).count();

    QueryOptions opts;
    opts.timeout_ms = 10000;
    QueryResult qresult;
    auto t4 = Clock::now();
    status = client_->Query(keys, opts, qresult);
    ASSERT_TRUE(status.ok()) << status.message;
    auto t5 = Clock::now();
    double query_sec = Duration(t5 - t4).count();

    std::printf("[BENCH] BulkStore:   %zu entries in %.3fms  throughput=%.1f entries/sec\n",
                kCount, store_sec * 1000, kCount / store_sec);
    std::printf("[BENCH] BulkLoad:    %zu entries in %.3fms  throughput=%.1f entries/sec\n",
                kCount, load_sec * 1000, kCount / load_sec);
    std::printf("[BENCH] BulkQuery:   %zu keys    in %.3fms  throughput=%.1f keys/sec\n",
                kCount, query_sec * 1000, kCount / query_sec);

    for (std::size_t i = 0; i < kCount; ++i) {
        EXPECT_EQ(std::memcmp(payloads[i].data(), load_payloads[i].data(), kPayloadSize), 0)
            << "data integrity failed for key " << keys[i];
    }
}

/**
 * @brief 并发竞争测试 - 多线程 Store/Load
 *
 * @details 测试流程:
 * ```
 * ConcurrentCompetition_MultiThreadStoreLoad
 *     │
 *     ├─► 启动 4 个线程
 *     │   each thread:
 *     │     ├─► MakeKVBuffers(64 entries, thread-specific prefix)
 *     │     ├─► for each entry: StoreAsync + Wait
 *     │     ├─► 记录延迟
 *     │     ├─► for each entry: LoadAsync + Wait
 *     │     ├─► 记录延迟
 *     │     ├─► 统计 success/fail
 *     │     └─► return latencies
 *     │
 *     ├─► 等待所有线程完成
 *     │
 *     ├─► 聚合延迟
 *     │
 *     ├─► 打印结果
 *     │   PrintBench("ConcurrentStoreLoad")
 *     │   printf success/fail count
 *     │
 *     └─► EXPECT fail == 0
 * ```
 *
 * @note 测试目的:
 * - 验证多线程并发操作的正确性
 * - 测量并发场景下的延迟和吞吐量
 * - 验证线程安全性
 */
TEST_F(AsuBenchmarkTest, ConcurrentCompetition_MultiThreadStoreLoad)
{
    constexpr std::size_t kThreads = 4;
    constexpr std::size_t kOpsPerThread = 64;
    constexpr std::size_t kPayloadSize = 4096;

    std::atomic<std::size_t> total_success{0};
    std::atomic<std::size_t> total_fail{0};
    std::vector<std::future<std::vector<double>>> futures;
    futures.reserve(kThreads);

    for (std::size_t t = 0; t < kThreads; ++t) {
        futures.push_back(std::async(std::launch::async, [this, t, &total_success, &total_fail]() {
            std::vector<double> lats;
            std::vector<std::vector<std::uint8_t>> payloads;
            auto entries = MakeKVBuffers(payloads, kOpsPerThread, kPayloadSize,
                                        "conc-" + std::to_string(t) + "-");

            for (std::size_t i = 0; i < kOpsPerThread; ++i) {
                TaskId id{kInvalidTaskId};
                auto t0 = Clock::now();
                auto status = client_->StoreAsync({entries[i]}, id);
                if (!status.ok()) {
                    total_fail.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                TaskResult result;
                status = WaitWithTimeout(*client_, id, 5000, result);
                if (!status.ok()) {
                    total_fail.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                auto t1 = Clock::now();
                lats.push_back(Duration(t1 - t0).count() * 1000);
                total_success.fetch_add(1, std::memory_order_relaxed);
            }

            std::vector<std::vector<std::uint8_t>> load_payloads(kOpsPerThread);
            std::vector<KVBuffer> load_entries(kOpsPerThread);
            auto keys = ExtractKeys(entries);
            for (std::size_t i = 0; i < kOpsPerThread; ++i) {
                load_payloads[i].resize(kPayloadSize, 0);
                MemoryRegion region;
                region.memory_type = MemoryType::HOST;
                region.addr = reinterpret_cast<std::uint64_t>(load_payloads[i].data());
                region.size = kPayloadSize;
                Buffer buffer;
                buffer.region = region;
                load_entries[i] = KVBuffer{keys[i], buffer};
            }

            for (std::size_t i = 0; i < kOpsPerThread; ++i) {
                TaskId id{kInvalidTaskId};
                auto t0 = Clock::now();
                auto status = client_->LoadAsync({load_entries[i]}, id);
                if (!status.ok()) {
                    total_fail.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                TaskResult result;
                status = WaitWithTimeout(*client_, id, 5000, result);
                if (!status.ok()) {
                    total_fail.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                auto t1 = Clock::now();
                lats.push_back(Duration(t1 - t0).count() * 1000);
                total_success.fetch_add(1, std::memory_order_relaxed);
            }

            return lats;
        }));
    }

    std::vector<double> all_lats;
    for (auto& f : futures) {
        auto lats = f.get();
        all_lats.insert(all_lats.end(), lats.begin(), lats.end());
    }

    auto bench = ComputeBench(all_lats);
    PrintBench("ConcurrentStoreLoad", bench);
    std::printf("[BENCH] ConcurrentStoreLoad: success=%zu  fail=%zu  threads=%zu  "
                "ops_per_thread=%zu\n",
                total_success.load(), total_fail.load(), kThreads, kOpsPerThread);

    EXPECT_EQ(total_fail.load(), 0u);
}

/**
 * @brief 大规模压力测试
 *
 * @details 测试流程:
 * ```
 * LargeScaleStress_ManyKeysLargePayload
 *     │
 *     ├─► MakeKVBuffers(2048 entries, 8192 bytes)
 *     │
 *     ├─► 批量 Store (timeout 30s)
 *     │
 *     ├─► 批量 Load
 *     │
 *     ├─► 打印吞吐量
 *     │
 *     └─► 验证数据一致性
 * ```
 *
 * @note 测试目的:
 * - 测试大规模数据处理能力
 * - 测试大 Payload (8KB) 的处理
 * - 验证内存管理正确性
 */
TEST_F(AsuBenchmarkTest, LargeScaleStress_ManyKeysLargePayload)
{
    constexpr std::size_t kCount = 2048;
    constexpr std::size_t kPayloadSize = 8192;

    std::vector<std::vector<std::uint8_t>> payloads;
    auto entries = MakeKVBuffers(payloads, kCount, kPayloadSize, "stress-");
    auto keys = ExtractKeys(entries);

    auto t0 = Clock::now();
    TaskId store_id{kInvalidTaskId};
    auto status = client_->StoreAsync(entries, store_id);
    ASSERT_TRUE(status.ok()) << status.message;
    TaskResult store_result;
    status = WaitWithTimeout(*client_, store_id, 30000, store_result);
    ASSERT_TRUE(status.ok()) << status.message;
    auto t1 = Clock::now();
    double store_sec = Duration(t1 - t0).count();

    std::vector<std::vector<std::uint8_t>> load_payloads(kCount);
    std::vector<KVBuffer> load_entries(kCount);
    for (std::size_t i = 0; i < kCount; ++i) {
        load_payloads[i].resize(kPayloadSize, 0);
        MemoryRegion region;
        region.memory_type = MemoryType::HOST;
        region.addr = reinterpret_cast<std::uint64_t>(load_payloads[i].data());
        region.size = kPayloadSize;
        Buffer buffer;
        buffer.region = region;
        load_entries[i] = KVBuffer{keys[i], buffer};
    }

    auto t2 = Clock::now();
    TaskId load_id{kInvalidTaskId};
    status = client_->LoadAsync(load_entries, load_id);
    ASSERT_TRUE(status.ok()) << status.message;
    TaskResult load_result;
    status = WaitWithTimeout(*client_, load_id, 30000, load_result);
    ASSERT_TRUE(status.ok()) << status.message;
    auto t3 = Clock::now();
    double load_sec = Duration(t3 - t2).count();

    std::printf("[BENCH] StressStore: %zu entries (%zu bytes each) in %.3fms  "
                "throughput=%.1f entries/sec  %.1f MB/sec\n",
                kCount, kPayloadSize, store_sec * 1000,
                kCount / store_sec,
                (kCount * kPayloadSize) / store_sec / (1024 * 1024));
    std::printf("[BENCH] StressLoad:  %zu entries (%zu bytes each) in %.3fms  "
                "throughput=%.1f entries/sec  %.1f MB/sec\n",
                kCount, kPayloadSize, load_sec * 1000,
                kCount / load_sec,
                (kCount * kPayloadSize) / load_sec / (1024 * 1024));

    for (std::size_t i = 0; i < kCount; ++i) {
        EXPECT_EQ(std::memcmp(payloads[i].data(), load_payloads[i].data(), kPayloadSize), 0)
            << "data integrity failed for key " << keys[i];
    }
}

/**
 * @brief RDMA 统计验证测试 - Store 统计
 *
 * @details 测试流程:
 * ```
 * RdmaStatsVerification_StoreTransfers
 *     │
 *     ├─► 创建 MockIoStats
 *     ├─► 配置 enable_rdma_stats = true
 *     ├─► 配置 rdma_per_entry = true
 *     │
 *     ├─► CreateRdmaMockBenchClient(io_stats, options)
 *     ├─► Init(config)
 *     │
 *     ├─► MakeKVBuffers(8 entries, 4096 bytes)
 *     ├─► StoreAsync + Wait
 *     │
 *     ├─► 验证统计:
 *     │   EXPECT store_transfers == 8
 *     │   EXPECT store_bytes == 8 * 4096 = 32768
 *     │
 *     └─► Shutdown
 * ```
 *
 * @note 测试目的:
 * - 验证 MockIoStats 正确记录 Store 统计
 * - 验证 rdma_per_entry 模式正确工作
 */
TEST_F(AsuBenchmarkTest, RdmaStatsVerification_StoreTransfers)
{
    auto io_stats = std::make_shared<MockIoStats>();
    MemoryIoBackendOptions options;
    options.enable_rdma_stats = true;
    options.rdma_per_entry = true;

    auto rdma_client = CreateRdmaMockBenchClient(io_stats, options);
    ASSERT_NE(rdma_client, nullptr);
    auto status = rdma_client->Init(MakeBenchConfig());
    ASSERT_TRUE(status.ok()) << status.message;

    constexpr std::size_t kCount = 8;
    constexpr std::size_t kPayloadSize = 4096;

    std::vector<std::vector<std::uint8_t>> payloads;
    auto entries = MakeKVBuffers(payloads, kCount, kPayloadSize, "rdma-store-");

    TaskId store_id{kInvalidTaskId};
    status = rdma_client->StoreAsync(entries, store_id);
    ASSERT_TRUE(status.ok()) << status.message;
    TaskResult store_result;
    status = WaitWithTimeout(*rdma_client, store_id, 5000, store_result);
    ASSERT_TRUE(status.ok()) << status.message;

    EXPECT_EQ(io_stats->store_transfers.load(), kCount);
    EXPECT_EQ(io_stats->store_bytes.load(), kCount * kPayloadSize);

    rdma_client->Shutdown();
}

/**
 * @brief RDMA 统计验证测试 - Load 统计
 *
 * @details 类似 Store 测试，验证 Load 统计正确记录。
 */
TEST_F(AsuBenchmarkTest, RdmaStatsVerification_LoadTransfers)
{
    auto io_stats = std::make_shared<MockIoStats>();
    MemoryIoBackendOptions options;
    options.enable_rdma_stats = true;
    options.rdma_per_entry = true;

    auto rdma_client = CreateRdmaMockBenchClient(io_stats, options);
    ASSERT_NE(rdma_client, nullptr);
    auto status = rdma_client->Init(MakeBenchConfig());
    ASSERT_TRUE(status.ok()) << status.message;

    constexpr std::size_t kCount = 8;
    constexpr std::size_t kPayloadSize = 4096;

    std::vector<std::vector<std::uint8_t>> store_payloads;
    auto entries = MakeKVBuffers(store_payloads, kCount, kPayloadSize, "rdma-load-");
    auto keys = ExtractKeys(entries);

    TaskId store_id{kInvalidTaskId};
    status = rdma_client->StoreAsync(entries, store_id);
    ASSERT_TRUE(status.ok()) << status.message;
    TaskResult store_result;
    status = WaitWithTimeout(*rdma_client, store_id, 5000, store_result);
    ASSERT_TRUE(status.ok()) << status.message;

    io_stats->Reset();

    std::vector<std::vector<std::uint8_t>> load_payloads(kCount);
    std::vector<KVBuffer> load_entries;
    load_entries.reserve(kCount);
    for (std::size_t i = 0; i < kCount; ++i) {
        load_payloads[i].resize(kPayloadSize, 0);
        MemoryRegion region;
        region.memory_type = MemoryType::HOST;
        region.addr = reinterpret_cast<std::uint64_t>(load_payloads[i].data());
        region.size = kPayloadSize;
        Buffer buffer;
        buffer.region = region;
        load_entries.push_back(KVBuffer{keys[i], buffer});
    }

    TaskId load_id{kInvalidTaskId};
    status = rdma_client->LoadAsync(load_entries, load_id);
    ASSERT_TRUE(status.ok()) << status.message;
    TaskResult load_result;
    status = WaitWithTimeout(*rdma_client, load_id, 5000, load_result);
    ASSERT_TRUE(status.ok()) << status.message;

    EXPECT_EQ(io_stats->load_transfers.load(), kCount);
    EXPECT_EQ(io_stats->load_bytes.load(), kCount * kPayloadSize);

    rdma_client->Shutdown();
}

/**
 * @brief RDMA 统计验证测试 - Query 统计
 *
 * @details Query 字节数 = Key 大小 + 1 (响应字节)。
 */
TEST_F(AsuBenchmarkTest, RdmaStatsVerification_QueryTransfers)
{
    auto io_stats = std::make_shared<MockIoStats>();
    MemoryIoBackendOptions options;
    options.enable_rdma_stats = true;
    options.rdma_per_entry = true;

    auto rdma_client = CreateRdmaMockBenchClient(io_stats, options);
    ASSERT_NE(rdma_client, nullptr);
    auto status = rdma_client->Init(MakeBenchConfig());
    ASSERT_TRUE(status.ok()) << status.message;

    constexpr std::size_t kCount = 8;
    constexpr std::size_t kPayloadSize = 4096;

    std::vector<std::vector<std::uint8_t>> payloads;
    auto entries = MakeKVBuffers(payloads, kCount, kPayloadSize, "rdma-query-");
    auto keys = ExtractKeys(entries);

    TaskId store_id{kInvalidTaskId};
    status = rdma_client->StoreAsync(entries, store_id);
    ASSERT_TRUE(status.ok()) << status.message;
    TaskResult store_result;
    status = WaitWithTimeout(*rdma_client, store_id, 5000, store_result);
    ASSERT_TRUE(status.ok()) << status.message;

    io_stats->Reset();

    QueryOptions opts;
    opts.timeout_ms = 5000;
    QueryResult result;
    status = rdma_client->Query(keys, opts, result);
    ASSERT_TRUE(status.ok()) << status.message;

    EXPECT_EQ(io_stats->query_transfers.load(), kCount);

    std::uint64_t expected_query_bytes = 0;
    for (const auto& key : keys) {
        expected_query_bytes += key.size() + 1;
    }
    EXPECT_EQ(io_stats->query_bytes.load(), expected_query_bytes);

    rdma_client->Shutdown();
}

/**
 * @brief RDMA 统计验证测试 - Delete 统计
 *
 * @details Delete 字节数 = Key 大小（无数据传输）。
 */
TEST_F(AsuBenchmarkTest, RdmaStatsVerification_DeleteTransfers)
{
    auto io_stats = std::make_shared<MockIoStats>();
    MemoryIoBackendOptions options;
    options.enable_rdma_stats = true;
    options.rdma_per_entry = true;

    auto rdma_client = CreateRdmaMockBenchClient(io_stats, options);
    ASSERT_NE(rdma_client, nullptr);
    auto status = rdma_client->Init(MakeBenchConfig());
    ASSERT_TRUE(status.ok()) << status.message;

    constexpr std::size_t kCount = 8;
    constexpr std::size_t kPayloadSize = 4096;

    std::vector<std::vector<std::uint8_t>> payloads;
    auto entries = MakeKVBuffers(payloads, kCount, kPayloadSize, "rdma-delete-");
    auto keys = ExtractKeys(entries);

    TaskId store_id{kInvalidTaskId};
    status = rdma_client->StoreAsync(entries, store_id);
    ASSERT_TRUE(status.ok()) << status.message;
    TaskResult store_result;
    status = WaitWithTimeout(*rdma_client, store_id, 5000, store_result);
    ASSERT_TRUE(status.ok()) << status.message;

    io_stats->Reset();

    TaskId delete_id{kInvalidTaskId};
    status = rdma_client->DeleteAsync(keys, delete_id);
    ASSERT_TRUE(status.ok()) << status.message;
    TaskResult delete_result;
    status = WaitWithTimeout(*rdma_client, delete_id, 5000, delete_result);
    ASSERT_TRUE(status.ok()) << status.message;

    EXPECT_EQ(io_stats->delete_transfers.load(), kCount);

    std::uint64_t expected_delete_bytes = 0;
    for (const auto& key : keys) {
        expected_delete_bytes += key.size();
    }
    EXPECT_EQ(io_stats->delete_bytes.load(), expected_delete_bytes);

    rdma_client->Shutdown();
}

/**
 * @brief RDMA 统计验证测试 - 批量模式
 *
 * @details rdma_per_entry = false 时，批量统计。
 * - transfers = 1 (一次批量操作)
 * - bytes = 所有 entry 的总字节数
 */
TEST_F(AsuBenchmarkTest, RdmaStatsVerification_BulkMode)
{
    auto io_stats = std::make_shared<MockIoStats>();
    MemoryIoBackendOptions options;
    options.enable_rdma_stats = true;
    options.rdma_per_entry = false;

    auto rdma_client = CreateRdmaMockBenchClient(io_stats, options);
    ASSERT_NE(rdma_client, nullptr);
    auto status = rdma_client->Init(MakeBenchConfig());
    ASSERT_TRUE(status.ok()) << status.message;

    constexpr std::size_t kCount = 8;
    constexpr std::size_t kPayloadSize = 4096;

    std::vector<std::vector<std::uint8_t>> payloads;
    auto entries = MakeKVBuffers(payloads, kCount, kPayloadSize, "rdma-bulk-");

    TaskId store_id{kInvalidTaskId};
    status = rdma_client->StoreAsync(entries, store_id);
    ASSERT_TRUE(status.ok()) << status.message;
    TaskResult store_result;
    status = WaitWithTimeout(*rdma_client, store_id, 5000, store_result);
    ASSERT_TRUE(status.ok()) << status.message;

    EXPECT_EQ(io_stats->store_transfers.load(), 1u);
    EXPECT_EQ(io_stats->store_bytes.load(), kCount * kPayloadSize);

    rdma_client->Shutdown();
}

}  // namespace UC::ASU