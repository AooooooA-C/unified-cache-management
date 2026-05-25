#include "mock_io_backend.h"

#include <algorithm>
#include <chrono>
#include <thread>
#include <utility>

#include "memory_io_backend.h"

namespace UC::ASU {

MockIoBackend::MockIoBackend(MockIoBackendOptions options,
                             std::unique_ptr<IoBackend> inner,
                             std::shared_ptr<MockRdmaStats> rdma_stats)
    : options_(options),
      inner_(std::move(inner)),
      rdma_stats_(std::move(rdma_stats)),
      rng_(std::random_device{}()),
      error_dist_(0.0, 1.0)
{
    if (!inner_) {
        inner_ = CreateMemoryIoBackend();
    }
}

Status MockIoBackend::Store(const std::vector<KVBuffer>& entries,
                            std::vector<Status>& entry_status)
{
    if (options_.rdma_per_entry) {
        for (const auto& entry : entries) {
            const auto bytes = entry.buffer.region.size;
            AccountStoreTransfer(bytes);
            MaybeDelay(bytes);
        }
    } else {
        const auto bytes = CalcBytes(entries);
        AccountStoreTransfer(bytes);
        MaybeDelay(bytes);
    }

    if (ShouldFail()) {
        entry_status.assign(entries.size(),
                            Status::Error(StatusCode::IO_ERROR, "mock store io error"));
        return Status::Error(StatusCode::IO_ERROR, "mock store io error");
    }

    return inner_->Store(entries, entry_status);
}

Status MockIoBackend::Load(const std::vector<KVBuffer>& entries,
                           std::vector<Status>& entry_status)
{
    if (options_.rdma_per_entry) {
        for (const auto& entry : entries) {
            const auto bytes = entry.buffer.region.size;
            AccountLoadTransfer(bytes);
            MaybeDelay(bytes);
        }
    } else {
        const auto bytes = CalcBytes(entries);
        AccountLoadTransfer(bytes);
        MaybeDelay(bytes);
    }

    if (ShouldFail()) {
        entry_status.assign(entries.size(),
                            Status::Error(StatusCode::IO_ERROR, "mock load io error"));
        return Status::Error(StatusCode::IO_ERROR, "mock load io error");
    }

    return inner_->Load(entries, entry_status);
}

Status MockIoBackend::Query(const std::vector<CacheKey>& keys,
                            QueryResult& result)
{
    if (options_.rdma_per_entry) {
        for (const auto& key : keys) {
            const auto bytes = std::max<std::uint64_t>(CalcKeyBytes(key) + 1,
                                                       options_.min_query_bytes);
            AccountQueryTransfer(bytes);
            if (options_.delay_query) {
                MaybeDelay(bytes);
            }
        }
    } else {
        std::uint64_t bytes = 0;
        for (const auto& key : keys) {
            bytes += CalcKeyBytes(key) + 1;
        }
        bytes = std::max<std::uint64_t>(bytes, options_.min_query_bytes);
        AccountQueryTransfer(bytes);
        if (options_.delay_query) {
            MaybeDelay(bytes);
        }
    }

    if (ShouldFail()) {
        result.exists.assign(keys.size(), 0);
        result.prefix_hit_keys = 0;
        return Status::Error(StatusCode::IO_ERROR, "mock query io error");
    }

    return inner_->Query(keys, result);
}

Status MockIoBackend::Delete(const std::vector<CacheKey>& keys,
                             std::vector<Status>& entry_status)
{
    if (options_.rdma_per_entry) {
        for (const auto& key : keys) {
            const auto bytes = std::max<std::uint64_t>(CalcKeyBytes(key),
                                                       options_.min_delete_bytes);
            AccountDeleteTransfer(bytes);
            if (options_.delay_delete) {
                MaybeDelay(bytes);
            }
        }
    } else {
        std::uint64_t bytes = 0;
        for (const auto& key : keys) {
            bytes += CalcKeyBytes(key);
        }
        bytes = std::max<std::uint64_t>(bytes, options_.min_delete_bytes);
        AccountDeleteTransfer(bytes);
        if (options_.delay_delete) {
            MaybeDelay(bytes);
        }
    }

    if (ShouldFail()) {
        entry_status.assign(keys.size(),
                            Status::Error(StatusCode::IO_ERROR, "mock delete io error"));
        return Status::Error(StatusCode::IO_ERROR, "mock delete io error");
    }

    return inner_->Delete(keys, entry_status);
}

std::uint64_t MockIoBackend::CalcBytes(const std::vector<KVBuffer>& entries)
{
    std::uint64_t bytes = 0;
    for (const auto& entry : entries) {
        bytes += entry.buffer.region.size;
    }
    return bytes;
}

std::uint64_t MockIoBackend::CalcKeyBytes(const CacheKey& key)
{
    return static_cast<std::uint64_t>(key.size());
}

void MockIoBackend::MaybeDelay(std::uint64_t bytes)
{
    std::uint64_t delay_us = options_.fixed_latency_us;

    if (options_.bandwidth_bytes_per_sec > 0 && bytes > 0) {
        const long double seconds = static_cast<long double>(bytes) /
            static_cast<long double>(options_.bandwidth_bytes_per_sec);
        delay_us += static_cast<std::uint64_t>(seconds * 1000.0L * 1000.0L);
    }

    if (options_.jitter_us > 0) {
        std::lock_guard<std::mutex> lock(rng_mu_);
        std::uniform_int_distribution<std::uint64_t> jitter_dist(0, options_.jitter_us);
        delay_us += jitter_dist(rng_);
    }

    if (delay_us > 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(delay_us));
    }
}

bool MockIoBackend::ShouldFail()
{
    if (options_.error_rate <= 0.0) {
        return false;
    }
    if (options_.error_rate >= 1.0) {
        return true;
    }

    std::lock_guard<std::mutex> lock(rng_mu_);
    return error_dist_(rng_) < options_.error_rate;
}

void MockIoBackend::AccountStoreTransfer(std::uint64_t bytes)
{
    if (!options_.enable_rdma_stats || !rdma_stats_) {
        return;
    }
    rdma_stats_->store_transfers.fetch_add(1, std::memory_order_relaxed);
    rdma_stats_->store_bytes.fetch_add(bytes, std::memory_order_relaxed);
}

void MockIoBackend::AccountLoadTransfer(std::uint64_t bytes)
{
    if (!options_.enable_rdma_stats || !rdma_stats_) {
        return;
    }
    rdma_stats_->load_transfers.fetch_add(1, std::memory_order_relaxed);
    rdma_stats_->load_bytes.fetch_add(bytes, std::memory_order_relaxed);
}

void MockIoBackend::AccountQueryTransfer(std::uint64_t bytes)
{
    if (!options_.enable_rdma_stats || !rdma_stats_) {
        return;
    }
    rdma_stats_->query_transfers.fetch_add(1, std::memory_order_relaxed);
    rdma_stats_->query_bytes.fetch_add(bytes, std::memory_order_relaxed);
}

void MockIoBackend::AccountDeleteTransfer(std::uint64_t bytes)
{
    if (!options_.enable_rdma_stats || !rdma_stats_) {
        return;
    }
    rdma_stats_->delete_transfers.fetch_add(1, std::memory_order_relaxed);
    rdma_stats_->delete_bytes.fetch_add(bytes, std::memory_order_relaxed);
}

std::unique_ptr<IoBackend> CreateMockIoBackend(
    MockIoBackendOptions options,
    std::unique_ptr<IoBackend> inner,
    std::shared_ptr<MockRdmaStats> rdma_stats)
{
    return std::make_unique<MockIoBackend>(options, std::move(inner), std::move(rdma_stats));
}

}  // namespace UC::ASU
