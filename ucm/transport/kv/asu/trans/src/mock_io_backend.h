#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <random>
#include <vector>

#include "io_backend.h"

namespace UC::ASU {

struct MockIoBackendOptions {
    // Fixed latency for each simulated IO/RDMA transfer.
    std::uint64_t fixed_latency_us{0};

    // Simulated bandwidth. 0 means no bandwidth-based delay.
    std::uint64_t bandwidth_bytes_per_sec{0};

    // Additional random latency in [0, jitter_us].
    std::uint64_t jitter_us{0};

    // Error injection probability. 0.0 means never fail, 1.0 means always fail.
    double error_rate{0.0};

    // Whether Query/Delete should also simulate delay.
    bool delay_query{true};
    bool delay_delete{true};

    // Whether to record simulated RDMA transfer statistics.
    bool enable_rdma_stats{false};

    // If true, each entry/key is counted as one RDMA transfer:
    //   Store(entries): entries.size() transfers
    //   Load(entries):  entries.size() transfers
    //   Query(keys):    keys.size() transfers
    //   Delete(keys):   keys.size() transfers
    // If false, one API call is counted as one transfer.
    bool rdma_per_entry{false};

    // Small request minimum bytes for RDMA metadata operations.
    std::uint64_t min_query_bytes{64};
    std::uint64_t min_delete_bytes{64};
};

struct MockRdmaStats {
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

class MockIoBackend final : public IoBackend {
public:
    explicit MockIoBackend(MockIoBackendOptions options,
                           std::unique_ptr<IoBackend> inner = nullptr,
                           std::shared_ptr<MockRdmaStats> rdma_stats = nullptr);

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
    MockIoBackendOptions options_;
    std::unique_ptr<IoBackend> inner_;
    std::shared_ptr<MockRdmaStats> rdma_stats_;

    std::mutex rng_mu_;
    std::mt19937_64 rng_;
    std::uniform_real_distribution<double> error_dist_;
};

std::unique_ptr<IoBackend> CreateMockIoBackend(
    MockIoBackendOptions options,
    std::unique_ptr<IoBackend> inner = nullptr,
    std::shared_ptr<MockRdmaStats> rdma_stats = nullptr);

}  // namespace UC::ASU
