#pragma once

#include <mutex>
#include <unordered_map>
#include <vector>
#include "io_backend.h"

namespace UC::ASU {

class MemoryIoBackend final : public IoBackend {
public:
    Status Store(const std::vector<KVBuffer>& entries,
                 std::vector<Status>& entry_status) override;

    Status Load(const std::vector<KVBuffer>& entries,
                std::vector<Status>& entry_status) override;

    Status Query(const std::vector<CacheKey>& keys,
                 QueryResult& result) override;

    Status Delete(const std::vector<CacheKey>& keys,
                  std::vector<Status>& entry_status) override;

private:
    std::mutex mu_;
    std::unordered_map<CacheKey, std::vector<std::uint8_t>> store_;
};

std::unique_ptr<IoBackend> CreateMemoryIoBackend();

}  // namespace UC::ASU