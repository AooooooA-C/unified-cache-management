/**
 * @file io_backend.h
 * @brief IO Backend 抽象接口
 *
 * 定义 KV 存储的四种核心操作：Store/Load/Query/Delete
 */

#pragma once

#include <functional>
#include <memory>
#include <vector>
#include "asu_transport/types.h"

namespace UC::ASU {

/**
 * @class IoBackend
 * @brief IO 后端抽象接口
 *
 * 所有方法都是纯虚函数，由具体实现类提供实现。
 * 支持批量操作以提高吞吐量。
 */
class IoBackend {
public:
    virtual ~IoBackend() = default;

    /**
     * @brief 批量存储 KV 数据
     * @param entries KV 数据列表
     * @param entry_status [out] 每个条目的存储状态
     * @return Status 整体操作状态
     */
    virtual Status Store(const std::vector<KVBuffer>& entries,
                         std::vector<Status>& entry_status) = 0;

    /**
     * @brief 批量加载 KV 数据
     * @param entries KV 数据列表（key + 目标 buffer）
     * @param entry_status [out] 每个条目的加载状态
     * @return Status 整体操作状态
     */
    virtual Status Load(const std::vector<KVBuffer>& entries,
                        std::vector<Status>& entry_status) = 0;

    /**
     * @brief 批量查询 Key 是否存在
     * @param keys Key 列表
     * @param result [out] 查询结果
     * @return Status 整体操作状态
     */
    virtual Status Query(const std::vector<CacheKey>& keys,
                         QueryResult& result) = 0;

    /**
     * @brief 批量删除 KV 数据
     * @param keys Key 列表
     * @param entry_status [out] 每个条目的删除状态
     * @return Status 整体操作状态
     */
    virtual Status Delete(const std::vector<CacheKey>& keys,
                          std::vector<Status>& entry_status) = 0;
};

/**
 * @typedef IoBackendFactory
 * @brief IoBackend 工厂函数类型，用于延迟创建实例
 */
using IoBackendFactory = std::function<std::unique_ptr<IoBackend>()>;

}  // namespace UC::ASU