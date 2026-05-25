/**
 * @file memory_io_backend.cpp
 * @brief MemoryIoBackend 实现
 */

#include "memory_io_backend.h"

#include <algorithm>
#include <cstring>
#include <thread>

namespace UC::ASU {

MemoryIoBackend::MemoryIoBackend(MemoryIoBackendOptions options,
                                 std::shared_ptr<MockIoStats> stats)
    : options_(options)
    , stats_(std::move(stats))
    , rng_(std::random_device{}())
    , error_dist_(0.0, 1.0)
{
}

Status MemoryIoBackend::Store(const std::vector<KVBuffer>& entries,
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

    std::lock_guard<std::mutex> lock(store_mu_);
    entry_status.resize(entries.size());

    for (std::size_t i = 0; i < entries.size(); ++i) {
        const auto& region = entries[i].buffer.region;
        if (region.memory_type != MemoryType::HOST) {
            entry_status[i] = Status::Error(StatusCode::BUFFER_NOT_SUPPORTED,
                                            "memory_io_backend only supports HOST memory");
            continue;
        }
        const auto* src = reinterpret_cast<const std::uint8_t*>(region.addr);
        if (!src || region.size == 0) {
            entry_status[i] = Status::Error(StatusCode::INVALID_ARGUMENT,
                                            "invalid source buffer for store");
            continue;
        }
        store_[entries[i].key] = std::vector<std::uint8_t>(src, src + region.size);
        entry_status[i] = Status::OK();
    }
    return Status::OK();
}

Status MemoryIoBackend::Load(const std::vector<KVBuffer>& entries,
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

    std::lock_guard<std::mutex> lock(store_mu_);
    entry_status.resize(entries.size());

    for (std::size_t i = 0; i < entries.size(); ++i) {
        auto it = store_.find(entries[i].key);
        if (it == store_.end()) {
            entry_status[i] = Status::Error(StatusCode::NOT_FOUND, "key not found in store");
            continue;
        }
        const auto& region = entries[i].buffer.region;
        if (region.memory_type != MemoryType::HOST) {
            entry_status[i] = Status::Error(StatusCode::BUFFER_NOT_SUPPORTED,
                                            "memory_io_backend only supports HOST memory");
            continue;
        }
        auto* dst = reinterpret_cast<std::uint8_t*>(region.addr);
        if (!dst) {
            entry_status[i] = Status::Error(StatusCode::INVALID_ARGUMENT,
                                            "invalid destination buffer for load");
            continue;
        }
        if (region.size < it->second.size()) {
            entry_status[i] = Status::Error(StatusCode::BUFFER_NOT_SUPPORTED,
                                            "destination buffer too small");
            continue;
        }
        std::memcpy(dst, it->second.data(), it->second.size());
        entry_status[i] = Status::OK();
    }
    return Status::OK();
}

Status MemoryIoBackend::Query(const std::vector<CacheKey>& keys,
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

    std::lock_guard<std::mutex> lock(store_mu_);
    result.exists.resize(keys.size());
    result.prefix_hit_keys = 0;

    for (std::size_t i = 0; i < keys.size(); ++i) {
        result.exists[i] = (store_.find(keys[i]) != store_.end()) ? 1 : 0;
    }
    return Status::OK();
}

Status MemoryIoBackend::Delete(const std::vector<CacheKey>& keys,
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

    std::lock_guard<std::mutex> lock(store_mu_);
    entry_status.resize(keys.size());

    for (std::size_t i = 0; i < keys.size(); ++i) {
        auto it = store_.find(keys[i]);
        if (it == store_.end()) {
            entry_status[i] = Status::Error(StatusCode::NOT_FOUND, "key not found in store");
        } else {
            store_.erase(it);
            entry_status[i] = Status::OK();
        }
    }
    return Status::OK();
}

std::uint64_t MemoryIoBackend::CalcBytes(const std::vector<KVBuffer>& entries)
{
    std::uint64_t bytes = 0;
    for (const auto& entry : entries) {
        bytes += entry.buffer.region.size;
    }
    return bytes;
}

std::uint64_t MemoryIoBackend::CalcKeyBytes(const CacheKey& key)
{
    return static_cast<std::uint64_t>(key.size());
}

void MemoryIoBackend::MaybeDelay(std::uint64_t bytes)
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

bool MemoryIoBackend::ShouldFail()
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

void MemoryIoBackend::AccountStoreTransfer(std::uint64_t bytes)
{
    if (!options_.enable_rdma_stats || !stats_) {
        return;
    }
    stats_->store_transfers.fetch_add(1, std::memory_order_relaxed);
    stats_->store_bytes.fetch_add(bytes, std::memory_order_relaxed);
}

void MemoryIoBackend::AccountLoadTransfer(std::uint64_t bytes)
{
    if (!options_.enable_rdma_stats || !stats_) {
        return;
    }
    stats_->load_transfers.fetch_add(1, std::memory_order_relaxed);
    stats_->load_bytes.fetch_add(bytes, std::memory_order_relaxed);
}

void MemoryIoBackend::AccountQueryTransfer(std::uint64_t bytes)
{
    if (!options_.enable_rdma_stats || !stats_) {
        return;
    }
    stats_->query_transfers.fetch_add(1, std::memory_order_relaxed);
    stats_->query_bytes.fetch_add(bytes, std::memory_order_relaxed);
}

void MemoryIoBackend::AccountDeleteTransfer(std::uint64_t bytes)
{
    if (!options_.enable_rdma_stats || !stats_) {
        return;
    }
    stats_->delete_transfers.fetch_add(1, std::memory_order_relaxed);
    stats_->delete_bytes.fetch_add(bytes, std::memory_order_relaxed);
}

std::unique_ptr<IoBackend> CreateMemoryIoBackend()
{
    return std::make_unique<MemoryIoBackend>();
}

std::unique_ptr<IoBackend> CreateMemoryIoBackend(MemoryIoBackendOptions options,
                                                  std::shared_ptr<MockIoStats> stats)
{
    return std::make_unique<MemoryIoBackend>(options, std::move(stats));
}

}  // namespace UC::ASU