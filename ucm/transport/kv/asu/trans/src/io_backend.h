#pragma once

#include <functional>
#include <memory>
#include <vector>
#include "asu_transport/types.h"

namespace UC::ASU {

class IoBackend {
public:
    virtual ~IoBackend() = default;

    virtual Status Store(const std::vector<KVBuffer>& entries,
                         std::vector<Status>& entry_status) = 0;

    virtual Status Load(const std::vector<KVBuffer>& entries,
                        std::vector<Status>& entry_status) = 0;

    virtual Status Query(const std::vector<CacheKey>& keys,
                         QueryResult& result) = 0;

    virtual Status Delete(const std::vector<CacheKey>& keys,
                          std::vector<Status>& entry_status) = 0;
};

using IoBackendFactory = std::function<std::unique_ptr<IoBackend>()>;

}  // namespace UC::ASU