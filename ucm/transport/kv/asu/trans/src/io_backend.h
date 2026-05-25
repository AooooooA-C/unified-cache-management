/**
 * @file io_backend.h
 * @brief IO Backend 抽象接口定义
 *
 * 本文件定义了 IoBackend 抽象基类，是 ASU 传输层的核心接口。
 * IoBackend 负责 KV (Key-Value) 存储的实际 IO 操作，支持多种实现：
 * - MemoryIoBackend: 内存存储（用于测试和 Mock）
 * - RdmaIoBackend: RDMA 远程存储（生产环境）
 *
 * @details 架构调用关系:
 *
 * ┌─────────────────────────────────────────────────────────────────────┐
 * │                          测试层 (asu_benchmark_test.cc)              │
 * │                                                                      │
 * │  CreateBenchClient() ──► CreateAsuClient(factory)                   │
 * │                              │                                       │
 * │                              ▼                                       │
 * │                      factory() ──► AsuTransportImpl                  │
 * │                              │        │                              │
 * │                              │        ▼                              │
 * │                              │   IoBackend (本文件定义的接口)         │
 * │                              │        │                              │
 * │                              │        ▼                              │
 * │                              │   MemoryIoBackend (具体实现)          │
 * │                              │        │                              │
 * │                              │        ▼                              │
 * │                              │   Store/Load/Query/Delete            │
 * └─────────────────────────────────────────────────────────────────────┘
 *
 * 数据流:
 * ┌──────────┐     ┌──────────────┐     ┌───────────────┐     ┌─────────────┐
 * │ AsuClient│ ──► │AsuTransportImpl│ ──► │ IoBackend     │ ──► │ 存储介质    │
 * │          │     │              │     │ (Store/Load) │     │ (内存/RDMA) │
 * └──────────┘     └──────────────┘     └───────────────┘     └─────────────┘
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
 * IoBackend 定义了 KV 存储的四种核心操作接口：
 * - Store: 存储 KV 数据
 * - Load: 加载 KV 数据
 * - Query: 查询 Key 是否存在
 * - Delete: 删除 KV 数据
 *
 * 所有方法都是纯虚函数，由具体实现类（如 MemoryIoBackend）提供实现。
 *
 * @note 线程安全要求:
 * 实现类必须保证在多线程环境下安全调用。MemoryIoBackend 使用 mutex 保护内部 store_。
 *
 * @note 批量操作设计:
 * 所有接口都支持批量操作（vector），减少 IO 调用次数，提高吞吐量。
 */
class IoBackend {
public:
    /**
     * @brief 虚析构函数
     *
     * 确保 derived class 通过基类指针删除时能正确析构。
     */
    virtual ~IoBackend() = default;

    /**
     * @brief 批量存储 KV 数据
     *
     * @param entries KV 数据列表vector，每个包含 key 和 buffer（数据指针+大小）
     * @param entry_status [out] 每个条目的存储状态（成功/失败）
     * @return Status 整体操作状态
     *
     * @note 数据所有权:
     * IoBackend 负责拷贝数据，调用者可在调用后立即释放 buffer。
     *
     * @note 执行流程:
     * 1. 校验 buffer 类型（仅支持 HOST 内存）
     * 2. 拷贝数据到内部存储
     * 3. 设置每个 entry 的状态
     *
     * @example 使用示例:
     * @code
     * std::vector<KVBuffer> entries;
     * std::vector<Status> entry_status;
     * auto status = backend->Store(entries, entry_status);
     * if (status.ok()) {
     *     for (const auto& es : entry_status) {
     *         if (es.ok()) { std::cout << "stored successfully"; }
     *     }
     * }
     * @endcode
     */
    virtual Status Store(const std::vector<KVBuffer>& entries,
                         std::vector<Status>& entry_status) = 0;

    /**
     * @brief 批量加载 KV 数据
     *
     * @param entries KV 数据列表，key 指定要加载的数据，buffer 指定目标内存
     * @param entry_status [out] 每个条目的加载状态
     * @return Status 整体操作状态
     *
     * @note buffer 要求:
     * 调用者必须预分配足够大的 buffer，否则返回 BUFFER_NOT_SUPPORTED。
     *
     * @note 执行流程:
     * 1. 按 key 查找内部存储
     * 2. 将数据拷贝到 buffer
     * 3. 设置每个 entry 的状态
     */
    virtual Status Load(const std::vector<KVBuffer>& entries,
                        std::vector<Status>& entry_status) = 0;

    /**
     * @brief 批量查询 Key 是否存在
     *
     * @param keys 要查询的 Key 列表
     * @param result [out] 查询结果，包含每个 key 的存在状态
     * @return Status 整体操作状态
     *
     * @note 执行流程:
     * 1. 按 key 查找内部存储
     * 2. 设置 exists[i] = 1 (存在) 或 0 (不存在)
     *
     * @note 应用场景:
     * 用于检查数据是否存在，避免无效的 Load 操作。
     */
    virtual Status Query(const std::vector<CacheKey>& keys,
                         QueryResult& result) = 0;

    /**
     * @brief 批量删除 KV 数据
     *
     * @param keys 要删除的 Key 列表
     * @param entry_status [out] 每个条目的删除状态
     * @return Status 整体操作状态
     *
     * @note 执行流程:
     * 1. 按 key 查找内部存储
     * 2. 删除找到的数据
     * 3. 设置状态（成功或 NOT_FOUND）
     */
    virtual Status Delete(const std::vector<CacheKey>& keys,
                          std::vector<Status>& entry_status) = 0;
};

/**
 * @typedef IoBackendFactory
 * @brief IoBackend 工厂函数类型
 *
 * 用于延迟创建 IoBackend 实例，支持依赖注入。
 *
 * @example 使用示例:
 * @code
 * // 定义工厂函数
 * IoBackendFactory factory = []() {
 *     return CreateMemoryIoBackend();
 * };
 *
 * // 在 AsuClient Init 时调用工厂
 * auto client = CreateAsuClient([factory]() {
 *     return std::make_unique<AsuTransportImpl>(factory());
 * });
 * @endcode
 */
using IoBackendFactory = std::function<std::unique_ptr<IoBackend>()>;

}  // namespace UC::ASU