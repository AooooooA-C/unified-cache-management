/**
 * @file memory_io_backend.h
 * @brief 内存 IO Backend 实现
 *
 * 提供 KV 存储功能 + Mock 功能（延迟模拟、错误注入、统计）
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
 * @brief IO 统计数据
 *
 * 记录 Store/Load/Query/Delete 的传输次数和字节数。
 * 所有计数器都是 atomic，支持多线程并发更新。
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
 * @brief Mock 配置选项
 *
 * - fixed_latency_us: 固定延迟（微秒）
 * - bandwidth_bytes_per_sec: 带宽限制（字节/秒）
 * - jitter_us: 延迟抖动范围（微秒）
 * - error_rate: 错误概率（0.0-1.0）
 * - delay_query/delay_delete: Query/Delete 是否模拟延迟
 * - enable_rdma_stats: 是否启用统计
 * - rdma_per_entry: 是否按 entry 统计
 * - min_query_bytes/min_delete_bytes: 最小统计字节数
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
 * IoBackend 的具体实现：
 * - 内存存储：使用 unordered_map 存储 KV 数据
 * - Mock 功能：延迟模拟、错误注入、统计
 */
class MemoryIoBackend final : public IoBackend {
public:
    explicit MemoryIoBackend(MemoryIoBackendOptions options = MemoryIoBackendOptions{},
                             std::shared_ptr<MockIoStats> stats = nullptr);

    Status Store(const std::vector<KVBuffer>& entries,
                 std::vector<Status>& entry_status) override;

    Status Load(const std::vector<KVBuffer>& entries,
                std::vector<Status>& entry_status) override;

    Status Query(const std::vector<CacheKey>& keys,
                 QueryResult& result) override;

    Status Delete(const std::vector<CacheKey>& keys,
                  std::vector<Status>& entry_status) override;

private:
    static std::uint64_t CalcBytes(const std::vector<KVBuffer>& entries);
    static std::uint64_t CalcKeyBytes(const CacheKey& key);

    void MaybeDelay(std::uint64_t bytes);
    bool ShouldFail();

    void AccountStoreTransfer(std::uint64_t bytes);
    void AccountLoadTransfer(std::uint64_t bytes);
    void AccountQueryTransfer(std::uint64_t bytes);
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
 */
std::unique_ptr<IoBackend> CreateMemoryIoBackend();

/**
 * @brief 创建带配置的 MemoryIoBackend
 */
std::unique_ptr<IoBackend> CreateMemoryIoBackend(
    MemoryIoBackendOptions options,
    std::shared_ptr<MockIoStats> stats = nullptr);

}  // namespace UC::ASU