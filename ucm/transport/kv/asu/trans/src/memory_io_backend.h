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

std::unique_ptr<IoBackend> CreateMemoryIoBackend();

std::unique_ptr<IoBackend> CreateMemoryIoBackend(
    MemoryIoBackendOptions options,
    std::shared_ptr<MockIoStats> stats = nullptr);

using MockIoBackendOptions = MemoryIoBackendOptions;
using MockIoBackend = MemoryIoBackend;

inline std::unique_ptr<IoBackend> CreateMockIoBackend(
    MockIoBackendOptions options,
    std::unique_ptr<IoBackend> inner = nullptr,
    std::shared_ptr<MockIoStats> stats = nullptr)
{
    (void)inner;
    return CreateMemoryIoBackend(options, stats);
}

}  // namespace UC::ASU