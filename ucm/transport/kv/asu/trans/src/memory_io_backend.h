/**
 * @file memory_io_backend.h
 * @brief 内存 IO Backend 实现（含 Mock 功能）
 *
 * 本文件实现了 IoBackend 接口，提供内存存储功能，同时内置 Mock 功能：
 * - 纯内存存储：快速测试，无实际 IO
 * - 延迟模拟：模拟 RDMA 传输延迟
 * - 统计功能：记录传输次数和字节数
 * - 错误注入：模拟 IO 错误
 *
 * @details 与其他文件的调用关系:
 *
 * ┌────────────────────────────────────────────────────────────────────────┐
 * │                     asu_benchmark_test.cc (测试入口)                    │
 * │                                                                         │
 * │  1. CreateBenchClient()                                                 │
 * │     └─► CreateAsuClient(factory)                                        │
 * │         └─► factory() = []() {                                          │
 * │               return AsuTransportImpl(                                  │
 * │                   CreateMemoryIoBackend()  ────────────────────────┐   │
 * │               );                                                    │   │
 * │             }                                                        │   │
 * │                                                                       │   │
 * │  2. CreateRdmaMockBenchClient(io_stats, options)                     │   │
 * │     └─► CreateAsuClient(factory)                                      │   │
 * │         └─► factory() = []() {                                        │   │
 * │               return AsuTransportImpl(                                │   │
 * │                   CreateMemoryIoBackend(options, io_stats) ◄─────────┘ │
 * │               );                                                       │
 * │             }                                                          │
 * │                                                                        │
 * │  3. 测试调用流程:                                                       │
 * │     client_->StoreAsync(entries, task_id)                              │
 * │     └─► AsuTransportImpl::SubmitTask()                                 │
 * │         └─► MemoryIoBackend::Store()  (本文件实现)                      │
 * │             ├─► MaybeDelay()        (模拟延迟)                         │
 * │             ├─► AccountStoreTransfer() (记录统计)                      │
 * │             └─► store_[key] = data  (内存存储)                         │
 * │                                                                        │
 * └────────────────────────────────────────────────────────────────────────┘
 *
 * @details 类结构:
 *
 * ┌──────────────────────────────────────────────────────────────────┐
 * │                     MemoryIoBackend                               │
 * │                    (继承 IoBackend)                                │
 * │                                                                   │
 * │  公有接口:                                                         │
 * │  ├─► Store(entries, entry_status)  存储 KV 数据                   │
 * │  ├─► Load(entries, entry_status)   加载 KV 数据                   │
 * │  ├─► Query(keys, result)           查询 Key 存在性                 │
 * │  └─► Delete(keys, entry_status)    删除 KV 数据                   │
 * │                                                                   │
 * │  私有辅助:                                                         │
 * │  ├─► MaybeDelay(bytes)             模拟传输延迟                    │
 * │  ├─► ShouldFail()                  模拟错误注入                    │
 * │  ├─► Account*Transfer(bytes)       记录统计数据                    │
 * │  ├─► CalcBytes(entries)            计算总字节数                    │
 * │  └─► CalcKeyBytes(key)             计算 Key 字节数                 │
 * │                                                                   │
 * │  成员变量:                                                         │
 * │  ├─► options_        Mock 配置（延迟、错误率等）                   │
 * │  ├─► stats_          统计对象（MockIoStats）                       │
 * │  ├─► store_          内存存储 map<key, data>                      │
 * │  ├─► store_mu_       保护 store_ 的 mutex                         │
 * │  ├─► rng_            随机数生成器（错误/抖动）                     │
 * │  └─► error_dist_     错误概率分布                                  │
 * │                                                                   │
 * └──────────────────────────────────────────────────────────────────┘
 */

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <random>
#include <unordered_map>
#include <vector>

#include "io_backend.h"

namespace UC::ASU {

/**
 * @struct MockIoStats
 * @brief IO 统计数据结构
 *
 * 用于记录 Store/Load/Query/Delete 操作的传输次数和字节数。
 * 所有计数器都是 atomic，支持多线程并发更新。
 *
 * @details 使用场景:
 * - 基准测试：验证传输次数是否符合预期
 * - 性能分析：计算吞吐量、带宽利用率
 * - RDMA 模拟：估算总传输延迟
 *
 * @note 线程安全:
 * 使用 std::atomic + memory_order_relaxed，适合统计场景（允许少量精度损失）。
 *
 * @example 测试中使用:
 * @code
 * auto stats = std::make_shared<MockIoStats>();
 * MemoryIoBackendOptions options;
 * options.enable_rdma_stats = true;
 *
 * auto backend = CreateMemoryIoBackend(options, stats);
 * backend->Store(entries, ...);
 *
 * // 验证传输次数
 * EXPECT_EQ(stats->store_transfers.load(), 8);
 * EXPECT_EQ(stats->store_bytes.load(), 8 * 4096);
 * @endcode
 */
struct MockIoStats {
    std::atomic<std::uint64_t> store_transfers{0};
    std::atomic<std::uint64_t> load_transfers{0};
    std::atomic<std::uint64_t> query_transfers{0};
    std::atomic<std::uint64_t> delete_transfers{0};

    std::atomic<std::uint64_t> store_bytes{0};
    std::atomic<std::uint64_t> load_bytes{0};
    std::atomic<std::uint64_t> query_bytes{0};
    std::atomic<std::uint64_t> delete_bytes{0};

    /**
     * @brief 重置所有计数器
     *
     * 用于测试前清空统计数据，确保每次测试从 0 开始。
     */
    void Reset()
    {
        store_transfers.store(0, std::memory_order_relaxed);
        load_transfers.store(0, std::memory_order_relaxed);
        query_transfers.store(0, std::memory_order_relaxed);
        delete_transfers.store(0, std::memory_order_relaxed);
        store_bytes.store(0, std::memory_order_relaxed);
        load_bytes.store(0, std::memory_order_relaxed);
        query_bytes.store(0, std::memory_order_relaxed);
        delete_bytes.store(0, std::memory_order_relaxed);
    }
};

/**
 * @struct MemoryIoBackendOptions
 * @brief MemoryIoBackend 配置选项
 *
 * 控制 Mock 功能的各个方面：延迟、带宽、抖动、错误率、统计等。
 *
 * @details 配置说明:
 *
 * | 配置项               | 默认值 | 说明                           |
 * |---------------------|--------|-------------------------------|
 * | fixed_latency_us    | 0      | 固定延迟（微秒）                |
 * | bandwidth_bytes_per_sec | 0  | 带宽限制（字节/秒）             |
 * | jitter_us           | 0      | 延迟抖动范围（微秒）            |
 * | error_rate          | 0.0    | 错误概率（0.0-1.0）            |
 * | delay_query         | true   | Query 是否模拟延迟             |
 * | delay_delete        | true   | Delete 是否模拟延迟            |
 * | enable_rdma_stats   | false  | 是否启用统计                   |
 * | rdma_per_entry      | false  | 是否按 entry 统计延迟          |
 * | min_query_bytes     | 64     | Query 最小统计字节数           |
 * | min_delete_bytes    | 64     | Delete 最小统计字节数          |
 *
 * @example 基准测试配置:
 * @code
 * MemoryIoBackendOptions options;
 * options.fixed_latency_us = 10;     // 10μs 固定延迟
 * options.bandwidth_bytes_per_sec = 100ULL * 1024 * 1024 * 1024;  // 100GB/s
 * options.jitter_us = 2;             // ±2μs 抖动
 * options.enable_rdma_stats = true;  // 启用统计
 * options.rdma_per_entry = true;     // 每个 entry 单独统计
 * @endcode
 */
struct MemoryIoBackendOptions {
    std::uint64_t fixed_latency_us{0};
    std::uint64_t bandwidth_bytes_per_sec{0};
    std::uint64_t jitter_us{0};
    double error_rate{0.0};

    bool delay_query{true};
    bool delay_delete{true};

    bool enable_rdma_stats{false};
    bool rdma_per_entry{false};

    std::uint64_t min_query_bytes{64};
    std::uint64_t min_delete_bytes{64};
};

/**
 * @class MemoryIoBackend
 * @brief 内存 IO Backend 实现
 *
 * MemoryIoBackend 是 IoBackend 的具体实现，提供：
 * 1. 内存存储：使用 unordered_map 存储 KV 数据
 * 2. Mock 功能：延迟模拟、错误注入、统计
 *
 * @details Store 操作流程:
 * ```
 * Store(entries, entry_status)
 *     │
 *     ├─► [Mock 功能]
 *     │   ├─► if rdma_per_entry: 每个 entry 单独延迟/统计
 *     │   │   else: 批量延迟/统计
 *     │   ├─► AccountStoreTransfer(bytes)  记录统计
 *     │   ├─► MaybeDelay(bytes)            模拟延迟
 *     │   └─► if ShouldFail(): 返回错误    模拟错误
 *     │
 *     ├─► [实际存储]
 *     │   ├─► lock(store_mu_)              获取锁
 *     │   ├─► for each entry:
 *     │   │   ├─► 校验 buffer 类型
 *     │   │   ├─► 拷贝数据: store_[key] = data
 *     │   │   └─► 设置 entry_status[i]
 *     │   └─► unlock                       释放锁
 *     │
 *     └─► return Status::OK()
 * ```
 *
 * @note 线程安全:
 * - store_mu_ 保护 store_ 的并发访问
 * - rng_mu_ 保护随机数生成器
 * - stats_ 使用 atomic，无需额外锁
 *
 * @note 内存管理:
 * - Store 时拷贝数据到内部 vector
 * - Load 时拷贝数据到调用者的 buffer
 * - 调用者负责管理自己的 buffer 内存
 */
class MemoryIoBackend final : public IoBackend {
public:
    /**
     * @brief 构造函数
     *
     * @param options Mock 配置选项
     * @param stats 统计对象（可选）
     *
     * @example 创建默认 Backend:
     * @code
     * auto backend = CreateMemoryIoBackend();  // 无 Mock 功能
     * @endcode
     *
     * @example 创建带 Mock 的 Backend:
     * @code
     * auto stats = std::make_shared<MockIoStats>();
     * MemoryIoBackendOptions options;
     * options.enable_rdma_stats = true;
     * options.fixed_latency_us = 10;
     *
     * auto backend = CreateMemoryIoBackend(options, stats);
     * @endcode
     */
    explicit MemoryIoBackend(MemoryIoBackendOptions options = MemoryIoBackendOptions{},
                             std::shared_ptr<MockIoStats> stats = nullptr);

    /**
     * @brief 批量存储 KV 数据
     *
     * @details 实现流程:
     * 1. Mock 处理：延迟模拟、统计记录、错误注入
     * 2. 内存存储：拷贝数据到 store_ map
     *
     * @param entries KV 数据列表
     * @param entry_status [out] 每个条目的状态
     * @return Status 整体操作状态
     *
     * @note 仅支持 HOST 内存类型，其他类型返回 BUFFER_NOT_SUPPORTED
     */
    Status Store(const std::vector<KVBuffer>& entries,
                 std::vector<Status>& entry_status) override;

    /**
     * @brief 批量加载 KV 数据
     *
     * @details 实现流程:
     * 1. Mock 处理：延迟模拟、统计记录
     * 2. 内存加载：从 store_ map 拷贝到 buffer
     *
     * @param entries KV 数据列表（key + 目标 buffer）
     * @param entry_status [out] 每个条目的状态
     * @return Status 整体操作状态
     *
     * @note Key 不存在返回 NOT_FOUND
     * @note Buffer 太小返回 BUFFER_NOT_SUPPORTED
     */
    Status Load(const std::vector<KVBuffer>& entries,
                std::vector<Status>& entry_status) override;

    /**
     * @brief 批量查询 Key 是否存在
     *
     * @details 实现流程:
     * 1. Mock 处理：延迟模拟（可选）、统计记录
     * 2. 内存查询：检查 store_ map 中是否存在
     *
     * @param keys Key 列表
     * @param result [out] 查询结果
     * @return Status 整体操作状态
     *
     * @note Query 延迟可通过 delay_query 配置控制
     */
    Status Query(const std::vector<CacheKey>& keys,
                 QueryResult& result) override;

    /**
     * @brief 批量删除 KV 数据
     *
     * @details 实现流程:
     * 1. Mock 处理：延迟模拟（可选）、统计记录
     * 2. 内存删除：从 store_ map 中移除
     *
     * @param keys Key 列表
     * @param entry_status [out] 每个条目的状态
     * @return Status 整体操作状态
     *
     * @note Key 不存在返回 NOT_FOUND
     * @note Delete 延迟可通过 delay_delete 配置控制
     */
    Status Delete(const std::vector<CacheKey>& keys,
                  std::vector<Status>& entry_status) override;

private:
    /**
     * @brief 计算批量数据的总字节数
     */
    static std::uint64_t CalcBytes(const std::vector<KVBuffer>& entries);

    /**
     * @brief 计算 Key 的字节数
     */
    static std::uint64_t CalcKeyBytes(const CacheKey& key);

    /**
     * @brief 模拟传输延迟
     *
     * @details 延迟计算公式:
     * ```
     * delay_us = fixed_latency_us
     *          + (bytes / bandwidth_bytes_per_sec) * 1e6  (带宽延迟)
     *          + random(0, jitter_us)                    (抖动)
     * ```
     *
     * @param bytes 传输的字节数
     */
    void MaybeDelay(std::uint64_t bytes);

    /**
     * @brief 判断是否应该模拟错误
     *
     * @details 基于 error_rate 配置，使用随机数判断是否注入错误。
     * @return true 应该返回错误
     * @return false 正常执行
     */
    bool ShouldFail();

    /**
     * @brief 记录 Store 统计
     */
    void AccountStoreTransfer(std::uint64_t bytes);

    /**
     * @brief 记录 Load 统计
     */
    void AccountLoadTransfer(std::uint64_t bytes);

    /**
     * @brief 记录 Query 统计
     */
    void AccountQueryTransfer(std::uint64_t bytes);

    /**
     * @brief 记录 Delete 统计
     */
    void AccountDeleteTransfer(std::uint64_t bytes);

private:
    MemoryIoBackendOptions options_;
    std::shared_ptr<MockIoStats> stats_;

    std::mutex store_mu_;
    std::unordered_map<CacheKey, std::vector<std::uint8_t>> store_;

    std::mutex rng_mu_;
    std::mt19937_64 rng_;
    std::uniform_real_distribution<double> error_dist_;
};

/**
 * @brief 创建默认配置的 MemoryIoBackend
 *
 * @return std::unique_ptr<IoBackend> IoBackend 实例
 *
 * @note 返回基类指针，隐藏具体实现类型
 *
 * @example 使用示例:
 * @code
 * auto backend = CreateMemoryIoBackend();
 * // backend 类型是 unique_ptr<IoBackend>，但实际是 MemoryIoBackend
 * @endcode
 */
std::unique_ptr<IoBackend> CreateMemoryIoBackend();

/**
 * @brief 创建带配置的 MemoryIoBackend
 *
 * @param options Mock 配置
 * @param stats 统计对象（可选）
 * @return std::unique_ptr<IoBackend> IoBackend 实例
 *
 * @example 基准测试使用:
 * @code
 * auto stats = std::make_shared<MockIoStats>();
 * MemoryIoBackendOptions options;
 * options.enable_rdma_stats = true;
 * options.fixed_latency_us = 10;
 *
 * auto backend = CreateMemoryIoBackend(options, stats);
 *
 * // 执行操作后检查统计
 * backend->Store(entries, ...);
 * std::cout << "transfers: " << stats->store_transfers.load() << std::endl;
 * @endcode
 */
std::unique_ptr<IoBackend> CreateMemoryIoBackend(
    MemoryIoBackendOptions options,
    std::shared_ptr<MockIoStats> stats = nullptr);

}  // namespace UC::ASU