/**
 * @file memory_io_backend.cpp
 * @brief MemoryIoBackend 实现
 *
 * 本文件实现了 memory_io_backend.h 中定义的 MemoryIoBackend 类。
 * 提供内存 KV 存储功能，同时支持 Mock 功能（延迟、错误、统计）。
 *
 * @details 与其他文件的调用关系:
 *
 * ┌───────────────────────────────────────────────────────────────────────────┐
 * │                        asu_benchmark_test.cc                               │
 * │                                                                            │
 * │  CreateBenchClient()                                                       │
 * │  └─► factory() = [io_stats, options]() {                                   │
 * │        return std::make_unique<AsuTransportImpl>(                          │
 * │            CreateMemoryIoBackend(options, io_stats)  ◄─── 调用本文件       │
 * │        );                                                                  │
 * │      }                                                                     │
 * │                                                                            │
 * │  client_->StoreAsync(entries, task_id)                                     │
 * │  └─► AsuTransportImpl::SubmitTask()                                        │
 * │      └─► MemoryIoBackend::Store()  ◄─── 本文件实现                         │
 * │          ├─► AccountStoreTransfer()  (记录统计到 stats_)                   │
 * │          ├─► MaybeDelay()            (模拟延迟)                            │
 * │          ├─► ShouldFail()            (模拟错误)                            │
 * │          └─► store_[key] = data      (内存存储)                            │
 * │                                                                            │
 * └───────────────────────────────────────────────────────────────────────────┘
 */

#include "memory_io_backend.h"

#include <algorithm>
#include <cstring>
#include <thread>

namespace UC::ASU {

/**
 * @brief 构造函数实现
 *
 * @details 初始化步骤:
 * 1. 存储 Mock 配置 (options_)
 * 2. 存储统计对象 (stats_)，使用 std::move 避免拷贝
 * 3. 初始化随机数生成器 (rng_)，用于错误注入和抖动
 * 4. 初始化错误概率分布 (error_dist_)，范围 [0.0, 1.0]
 *
 * @param options Mock 配置
 * @param stats 统计对象（shared_ptr，可被多个 Backend 共享）
 */
MemoryIoBackend::MemoryIoBackend(MemoryIoBackendOptions options,
                                 std::shared_ptr<MockIoStats> stats)
    : options_(options)
    , stats_(std::move(stats))
    , rng_(std::random_device{}())
    , error_dist_(0.0, 1.0)
{
}
// TODO 后期entries 会被分成sub batch ，每个若干个sub batch 传输一次
/**
 * @brief Store 实现 - 批量存储 KV 数据
 *
 * @details 执行流程图:
 * ```
 * Store(entries, entry_status)
 *     │
 *     ├─► [阶段1: Mock 功能]
 *     │   │
 *     │   ├─► if rdma_per_entry:                        (按 entry 统计)
 *     │   │   for each entry:
 *     │   │     bytes = entry.buffer.region.size
 *     │   │     AccountStoreTransfer(bytes)             (记录统计)
 *     │   │     MaybeDelay(bytes)                       (模拟延迟)
 *     │   │
 *     │   └─► else:                                      (批量统计)
 *     │       bytes = CalcBytes(entries)                (计算总字节数)
 *     │       AccountStoreTransfer(bytes)               (记录统计)
 *     │       MaybeDelay(bytes)                         (模拟延迟)
 *     │
 *     ├─► [阶段2: 错误注入]
 *     │   if ShouldFail():                              (基于 error_rate)
 *     │     entry_status.assign(IO_ERROR)               (所有条目失败)
 *     │     return Status::Error(IO_ERROR)              (整体失败)
 *     │
 *     ├─► [阶段3: 内存存储]
 *     │   │
 *     │   ├─► lock_guard(store_mu_)                     (获取锁，保护 store_)
 *     │   ├─► entry_status.resize(entries.size())       (预分配状态数组)
 *     │   │
 *     │   └─► for each entry (i = 0..n):
 *     │       │
 *     │       ├─► 校验 buffer.memory_type == HOST       (仅支持 HOST 内存)
 *     │       │   else: entry_status[i] = BUFFER_NOT_SUPPORTED
 *     │       │
 *     │       ├─► 校验 buffer.addr != null && size > 0  (有效性检查)
 *     │       │   else: entry_status[i] = INVALID_ARGUMENT
 *     │       │
 *     │       └─► store_[key] = vector(src, src+size)   (拷贝数据到内存)
 *     │           entry_status[i] = OK                  (成功)
 *     │
 *     └─► return Status::OK()
 * ```
 *
 * @note rdma_per_entry 模式:
 * 当 rdma_per_entry = true 时，每个 entry 单独统计延迟，
 * 更真实模拟 RDMA 逐条传输的场景。
 *
 * @param entries KV 数据列表
 * @param entry_status [out] 每个条目的存储状态
 * @return Status 整体操作状态
 */
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

/**
 * @brief Load 实现 - 批量加载 KV 数据
 *
 * @details 执行流程图:
 * ```
 * Load(entries, entry_status)
 *     │
 *     ├─► [阶段1: Mock 功能] (与 Store 类似)
 *     │   ├─► 统计/延迟 (rdma_per_entry 或批量)
 *     │
 *     ├─► [阶段2: 错误注入]
 *     │   if ShouldFail(): return IO_ERROR
 *     │
 *     ├─► [阶段3: 内存加载]
 *     │   │
 *     │   ├─► lock_guard(store_mu_)
 *     │   │
 *     │   └─► for each entry:
 *     │       │
 *     │       ├─► it = store_.find(key)                (查找 Key)
 *     │       │   if not found: entry_status[i] = NOT_FOUND
 *     │       │
 *     │       ├─► 校验 buffer.memory_type == HOST
 *     │       │
 *     │       ├─► 校验 buffer.addr != null
 *     │       │
 *     │       ├─► 校验 buffer.size >= data.size        (目标 buffer 足够大)
 *     │       │   else: entry_status[i] = BUFFER_NOT_SUPPORTED
 *     │       │
 *     │       └─► memcpy(dst, data, size)              (拷贝数据)
 *     │           entry_status[i] = OK
 *     │
 *     └─► return Status::OK()
 * ```
 *
 * @note Buffer 大小检查:
 * Load 要求调用者预分配足够大的 buffer。
 * 如果 buffer.size < 存储的数据大小，返回 BUFFER_NOT_SUPPORTED。
 *
 * @param entries KV 数据列表（key + 目标 buffer）
 * @param entry_status [out] 每个条目的加载状态
 * @return Status 整体操作状态
 */
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

/**
 * @brief Query 实现 - 批量查询 Key 是否存在
 *
 * @details 执行流程图:
 * ```
 * Query(keys, result)
 *     │
 *     ├─► [阶段1: Mock 功能]
 *     │   │
 *     │   ├─► if rdma_per_entry:
 *     │   │   for each key:
 *     │   │     bytes = max(CalcKeyBytes(key) + 1, min_query_bytes)
 *     │   │     AccountQueryTransfer(bytes)
 *     │   │     if delay_query: MaybeDelay(bytes)     (可选延迟)
 *     │   │
 *     │   └─► else:
 *     │       bytes = sum(CalcKeyBytes(key) + 1)      (+1 是响应字节)
 *     │       bytes = max(bytes, min_query_bytes)     (最小字节数限制)
 *     │       AccountQueryTransfer(bytes)
 *     │       if delay_query: MaybeDelay(bytes)
 *     │
 *     ├─► [阶段2: 错误注入]
 *     │   if ShouldFail():
 *     │     result.exists.assign(0)                   (所有 Key 不存在)
 *     │     return IO_ERROR
 *     │
 *     ├─► [阶段3: 内存查询]
 *     │   │
 *     │   ├─► lock_guard(store_mu_)
 *     │   ├─► result.exists.resize(keys.size())
 *     │   ├─► result.prefix_hit_keys = 0              (清零计数)
 *     │   │
 *     │   └─► for each key:
 *     │       exists[i] = (store_.find(key) != end) ? 1 : 0
 *     │
 *     └─► return Status::OK()
 * ```
 *
 * @note 字节数计算:
 * Query 字节数 = Key 大小 + 1 (响应字节: 存在/不存在)
 * 最小值由 min_query_bytes 限制，模拟最小传输开销。
 *
 * @note delay_query 配置:
 * Query 操作可能不需要延迟（仅查询元数据），可通过 delay_query = false 禁用。
 *
 * @param keys Key 列表
 * @param result [out] 查询结果
 * @return Status 整体操作状态
 */
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

/**
 * @brief Delete 实现 - 批量删除 KV 数据
 *
 * @details 执行流程图:
 * ```
 * Delete(keys, entry_status)
 *     │
 *     ├─► [阶段1: Mock 功能] (与 Query 类似)
 *     │   ├─► 字节数 = CalcKeyBytes(key)             (仅 Key，无数据)
 *     │   ├─► 最小值由 min_delete_bytes 限制
 *     │   ├─► 统计/延迟 (delay_delete 可选)
 *     │
 *     ├─► [阶段2: 错误注入]
 *     │   if ShouldFail(): return IO_ERROR
 *     │
 *     ├─► [阶段3: 内存删除]
 *     │   │
 *     │   ├─► lock_guard(store_mu_)
 *     │   │
 *     │   └─► for each key:
 *     │       │
 *     │       ├─► it = store_.find(key)
 *     │       │
 *     │       ├─► if not found:
 *     │       │   entry_status[i] = NOT_FOUND       (Key 不存在)
 *     │       │
 *     │       └─► else:
 *     │           store_.erase(it)                  (删除数据)
 *     │           entry_status[i] = OK
 *     │
 *     └─► return Status::OK()
 * ```
 *
 * @note 字节数计算:
 * Delete 字节数仅包含 Key 大小（删除操作不传输数据）。
 * 最小值由 min_delete_bytes 限制。
 *
 * @param keys Key 列表
 * @param entry_status [out] 每个条目的删除状态
 * @return Status 整体操作状态
 */
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

/**
 * @brief 计算批量数据的总字节数
 *
 * @details 遍历所有 entry，累加 buffer.region.size。
 *
 * @param entries KV 数据列表
 * @return std::uint64_t 总字节数
 */
std::uint64_t MemoryIoBackend::CalcBytes(const std::vector<KVBuffer>& entries)
{
    std::uint64_t bytes = 0;
    for (const auto& entry : entries) {
        bytes += entry.buffer.region.size;
    }
    return bytes;
}

/**
 * @brief 计算 Key 的字节数
 *
 * @details Key 类型是 CacheKey（通常是 std::string 或类似类型），
 * 返回其 size() 作为字节数。
 *
 * @param key CacheKey
 * @return std::uint64_t Key 的字节数
 */
std::uint64_t MemoryIoBackend::CalcKeyBytes(const CacheKey& key)
{
    return static_cast<std::uint64_t>(key.size());
}

/**
 * @brief 模拟传输延迟
 *
 * @details 延迟计算公式:
 * ```
 * 总延迟 = 固定延迟 + 带宽延迟 + 抖动
 *
 * 固定延迟 = options_.fixed_latency_us
 * 带宽延迟 = (bytes / bandwidth_bytes_per_sec) * 1,000,000 μs
 * 抖动     = random(0, jitter_us)
 * ```
 *
 * @example 100GB/s 带宽，4KB 数据的延迟:
 * ```
 * fixed = 10 μs
 * bandwidth = (4096 / (100 * 1024^3)) * 1e6 ≈ 0.04 μs
 * jitter = random(0, 2) μs
 * total ≈ 10 + 0.04 + 1 = 11 μs
 * ```
 *
 * @param bytes 传输的字节数
 */
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

/**
 * @brief 判断是否应该模拟错误
 *
 * @details 基于 error_rate 配置，使用均匀分布随机数判断:
 * ```
 * if error_rate <= 0.0: 永不失败
 * if error_rate >= 1.0: 永远失败
 * else: return (random(0,1) < error_rate)
 * ```
 *
 * @example error_rate = 0.1 (10% 错误率):
 * - 约 10% 的操作会返回 IO_ERROR
 * - 用于测试错误处理逻辑
 *
 * @return true 应该返回错误
 * @return false 正常执行
 */
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

/**
 * @brief 记录 Store 统计
 *
 * @details 统计内容:
 * - store_transfers: Store 操作次数
 * - store_bytes: Store 传输的总字节数
 *
 * @note 统计条件:
 * 仅当 enable_rdma_stats = true 且 stats_ != null 时才记录。
 *
 * @param bytes 本次传输的字节数
 */
void MemoryIoBackend::AccountStoreTransfer(std::uint64_t bytes)
{
    if (!options_.enable_rdma_stats || !stats_) {
        return;
    }
    stats_->store_transfers.fetch_add(1, std::memory_order_relaxed);
    stats_->store_bytes.fetch_add(bytes, std::memory_order_relaxed);
}

/**
 * @brief 记录 Load 统计
 *
 * @param bytes 本次传输的字节数
 */
void MemoryIoBackend::AccountLoadTransfer(std::uint64_t bytes)
{
    if (!options_.enable_rdma_stats || !stats_) {
        return;
    }
    stats_->load_transfers.fetch_add(1, std::memory_order_relaxed);
    stats_->load_bytes.fetch_add(bytes, std::memory_order_relaxed);
}

/**
 * @brief 记录 Query 统计
 *
 * @param bytes 本次传输的字节数
 */
void MemoryIoBackend::AccountQueryTransfer(std::uint64_t bytes)
{
    if (!options_.enable_rdma_stats || !stats_) {
        return;
    }
    stats_->query_transfers.fetch_add(1, std::memory_order_relaxed);
    stats_->query_bytes.fetch_add(bytes, std::memory_order_relaxed);
}

/**
 * @brief 记录 Delete 统计
 *
 * @param bytes 本次传输的字节数
 */
void MemoryIoBackend::AccountDeleteTransfer(std::uint64_t bytes)
{
    if (!options_.enable_rdma_stats || !stats_) {
        return;
    }
    stats_->delete_transfers.fetch_add(1, std::memory_order_relaxed);
    stats_->delete_bytes.fetch_add(bytes, std::memory_order_relaxed);
}

/**
 * @brief 工厂函数 - 创建默认配置的 MemoryIoBackend
 *
 * @details 使用默认构造函数创建，无 Mock 功能。
 *
 * @return std::unique_ptr<IoBackend> IoBackend 实例
 *
 * @note 返回类型是基类指针 IoBackend，隐藏具体实现类型。
 *       这是工厂模式的常见做法，支持未来扩展其他 Backend。
 *
 * @example 测试中使用:
 * @code
 * // CreateBenchClient() 内部调用
 * auto factory = []() -> std::unique_ptr<AsuTransport> {
 *     return std::make_unique<AsuTransportImpl>(
 *         CreateMemoryIoBackend());  // 调用此函数
 * };
 * return CreateAsuClient(factory);
 * @endcode
 */
std::unique_ptr<IoBackend> CreateMemoryIoBackend()
{
    return std::make_unique<MemoryIoBackend>();
}

/**
 * @brief 工厂函数 - 创建带配置的 MemoryIoBackend
 *
 * @details 创建带 Mock 功能的 Backend，用于基准测试和性能分析。
 *
 * @param options Mock 配置（延迟、错误率、统计等）
 * @param stats 统计对象（shared_ptr，可被多个测试共享）
 * @return std::unique_ptr<IoBackend> IoBackend 实例
 *
 * @note std::move(stats):
 * 使用 std::move 将 shared_ptr 移动到 Backend 内部，
 * 避免不必要的引用计数增加（虽然 shared_ptr 拷贝成本低，
 * 但移动是更优的做法）。
 *
 * @example 基准测试使用:
 * @code
 * // CreateRdmaMockBenchClient() 内部调用
 * auto io_stats = std::make_shared<MockIoStats>();
 * MemoryIoBackendOptions options;
 * options.enable_rdma_stats = true;
 * options.fixed_latency_us = 10;
 * options.rdma_per_entry = true;
 *
 * auto factory = [io_stats, options]() -> std::unique_ptr<AsuTransport> {
 *     return std::make_unique<AsuTransportImpl>(
 *         CreateMemoryIoBackend(options, io_stats));  // 调用此函数
 * };
 * return CreateAsuClient(factory);
 * @endcode
 */
std::unique_ptr<IoBackend> CreateMemoryIoBackend(MemoryIoBackendOptions options,
                                                  std::shared_ptr<MockIoStats> stats)
{
    return std::make_unique<MemoryIoBackend>(options, std::move(stats));
}

}  // namespace UC::ASU