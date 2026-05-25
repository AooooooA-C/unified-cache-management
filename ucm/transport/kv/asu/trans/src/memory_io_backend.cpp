#include "memory_io_backend.h"
#include <cstring>

namespace UC::ASU {

Status MemoryIoBackend::Store(const std::vector<KVBuffer>& entries,
                              std::vector<Status>& entry_status)
{
    std::lock_guard<std::mutex> lock(mu_);
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
    std::lock_guard<std::mutex> lock(mu_);
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
    std::lock_guard<std::mutex> lock(mu_);
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
    std::lock_guard<std::mutex> lock(mu_);
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

std::unique_ptr<IoBackend> CreateMemoryIoBackend()
{
    return std::make_unique<MemoryIoBackend>();
}

}  // namespace UC::ASU