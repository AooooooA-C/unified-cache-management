#include "asu_client/asu_client.h"
#include "io_backend.h"
#include "memory_io_backend.h"
#include "asu_transport_impl.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <future>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <sstream>
#include <thread>
#include <vector>

namespace UC::ASU {
namespace {

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double>;

struct BenchResult {
    double total_sec{0};
    std::size_t ops{0};
    double avg_lat_ms{0};
    double min_lat_ms{0};
    double max_lat_ms{0};
    double throughput_ops_sec{0};
};

BenchResult ComputeBench(const std::vector<double>& latencies_ms)
{
    BenchResult r;
    r.ops = latencies_ms.size();
    r.total_sec = std::accumulate(latencies_ms.begin(), latencies_ms.end(), 0.0) / 1000.0;
    r.avg_lat_ms = r.ops > 0 ? std::accumulate(latencies_ms.begin(), latencies_ms.end(), 0.0) / r.ops : 0;
    r.min_lat_ms = r.ops > 0 ? *std::min_element(latencies_ms.begin(), latencies_ms.end()) : 0;
    r.max_lat_ms = r.ops > 0 ? *std::max_element(latencies_ms.begin(), latencies_ms.end()) : 0;
    r.throughput_ops_sec = r.total_sec > 0 ? r.ops / r.total_sec : 0;
    return r;
}

void PrintBench(const char* label, const BenchResult& r)
{
    std::printf("[BENCH] %s: ops=%zu  total=%.3fms  avg=%.3fms  min=%.3fms  max=%.3fms  "
                "throughput=%.1f ops/sec\n",
                label, r.ops, r.total_sec * 1000, r.avg_lat_ms, r.min_lat_ms, r.max_lat_ms,
                r.throughput_ops_sec);
}

std::unique_ptr<AsuClient> CreateBenchClient()
{
    auto factory = []() -> std::unique_ptr<AsuTransport> {
        return std::make_unique<AsuTransportImpl>(CreateMemoryIoBackend());
    };
    return CreateAsuClient(factory);
}

[[maybe_unused]] static std::unique_ptr<AsuClient> CreateRdmaMockBenchClient(
    const std::shared_ptr<MockIoStats>& rdma_stats,
    MemoryIoBackendOptions options)
{
    auto factory = [rdma_stats, options]() -> std::unique_ptr<AsuTransport> {
        return std::make_unique<AsuTransportImpl>(
            CreateMemoryIoBackend(options, rdma_stats));
    };
    return CreateAsuClient(factory);
}

AsuClientConfig MakeBenchConfig(std::size_t num_transports = 1)
{
    AsuClientConfig config;
    config.client_id = "asu-bench-client";
    config.default_wait_timeout_ms = 5000;
    for (std::size_t i = 0; i < num_transports; ++i) {
        TransportConfig tc;
        tc.asu_name = "asu-bench-" + std::to_string(i);
        tc.asu_id = 2001 + static_cast<AsuId>(i);
        tc.max_inflight_tasks = 4096;
        tc.query_timeout_ms = 5000;
        config.transport_configs.push_back(tc);
    }
    return config;
}

std::vector<KVBuffer> MakeKVBuffers(std::vector<std::vector<std::uint8_t>>& payloads,
                                    std::size_t count, std::size_t payload_size,
                                    const std::string& key_prefix = "key-")
{
    payloads.resize(count);
    std::vector<KVBuffer> entries;
    entries.reserve(count);
    std::mt19937 rng(42);
    for (std::size_t i = 0; i < count; ++i) {
        payloads[i].resize(payload_size);
        std::generate(payloads[i].begin(), payloads[i].end(), rng);
        MemoryRegion region;
        region.memory_type = MemoryType::HOST;
        region.addr = reinterpret_cast<std::uint64_t>(payloads[i].data());
        region.size = payloads[i].size();
        Buffer buffer;
        buffer.region = region;
        entries.push_back(KVBuffer{key_prefix + std::to_string(i), buffer});
    }
    return entries;
}

std::vector<CacheKey> ExtractKeys(const std::vector<KVBuffer>& entries)
{
    std::vector<CacheKey> keys;
    keys.reserve(entries.size());
    for (const auto& e : entries) { keys.push_back(e.key); }
    return keys;
}

Status WaitWithTimeout(AsuClient& client, TaskId task_id, std::uint64_t timeout_ms,
                       TaskResult& result)
{
    auto status = client.Wait(task_id, timeout_ms, result);
    if (!status.ok()) { return status; }
    if (!result.status.ok()) { return result.status; }
    for (const auto& es : result.entry_status) {
        if (!es.ok()) { return es; }
    }
    return Status::OK();
}



struct StressTaskData {
    std::vector<std::vector<std::uint8_t>> payloads;
    std::vector<KVBuffer> entries;
    std::vector<CacheKey> keys;
    TaskId task_id{kInvalidTaskId};
};

StressTaskData MakeStressStoreTask(std::size_t task_index, std::size_t entries_per_task,
                                   std::size_t payload_size, const std::string& key_prefix)
{
    StressTaskData data;
    data.payloads.resize(entries_per_task);
    data.entries.reserve(entries_per_task);
    data.keys.reserve(entries_per_task);

    std::mt19937 rng(static_cast<std::uint32_t>(task_index + 12345));

    for (std::size_t i = 0; i < entries_per_task; ++i) {
        data.payloads[i].resize(payload_size);
        std::generate(data.payloads[i].begin(), data.payloads[i].end(), rng);

        auto key = key_prefix + std::to_string(task_index) + "-" + std::to_string(i);

        MemoryRegion region;
        region.memory_type = MemoryType::HOST;
        region.addr = reinterpret_cast<std::uint64_t>(data.payloads[i].data());
        region.size = data.payloads[i].size();

        Buffer buffer;
        buffer.region = region;

        data.entries.push_back(KVBuffer{key, buffer});
        data.keys.push_back(key);
    }

    return data;
}

StressTaskData MakeStressLoadTask(const std::vector<CacheKey>& keys, std::size_t payload_size)
{
    StressTaskData data;
    data.keys = keys;
    data.payloads.resize(keys.size());
    data.entries.reserve(keys.size());

    for (std::size_t i = 0; i < keys.size(); ++i) {
        data.payloads[i].assign(payload_size, 0);

        MemoryRegion region;
        region.memory_type = MemoryType::HOST;
        region.addr = reinterpret_cast<std::uint64_t>(data.payloads[i].data());
        region.size = data.payloads[i].size();

        Buffer buffer;
        buffer.region = region;

        data.entries.push_back(KVBuffer{keys[i], buffer});
    }

    return data;
}

}  // namespace

class AsuBenchmarkTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        client_ = CreateBenchClient();
        ASSERT_NE(client_, nullptr);
        auto status = client_->Init(MakeBenchConfig());
        ASSERT_TRUE(status.ok()) << status.message;
    }

    void TearDown() override
    {
        if (client_) {
            auto status = client_->Shutdown();
            ASSERT_TRUE(status.ok()) << status.message;
        }
        TaskId post_shutdown_task{kInvalidTaskId};
        std::vector<std::uint8_t> dummy_payload(1, 0);
        MemoryRegion region;
        region.memory_type = MemoryType::HOST;
        region.addr = reinterpret_cast<std::uint64_t>(dummy_payload.data());
        region.size = 1;
        Buffer buffer;
        buffer.region = region;
        KVBuffer entry{"shutdown-check", buffer};
        auto s = client_->LoadAsync({entry}, post_shutdown_task);
        ASSERT_EQ(s.code, StatusCode::NOT_INITIALIZED);
    }

    std::unique_ptr<AsuClient> client_;
};

TEST_F(AsuBenchmarkTest, DataIntegrity_StoreLoadVerify)
{
    constexpr std::size_t kCount = 64;
    constexpr std::size_t kPayloadSize = 4096;

    std::vector<std::vector<std::uint8_t>> original_payloads;
    auto entries = MakeKVBuffers(original_payloads, kCount, kPayloadSize, "integrity-");
    auto keys = ExtractKeys(entries);

    TaskId store_id{kInvalidTaskId};
    auto status = client_->StoreAsync(entries, store_id);
    ASSERT_TRUE(status.ok()) << status.message;
    TaskResult store_result;
    status = WaitWithTimeout(*client_, store_id, 5000, store_result);
    ASSERT_TRUE(status.ok()) << status.message;

    std::vector<std::vector<std::uint8_t>> load_payloads(kCount);
    std::vector<KVBuffer> load_entries;
    load_entries.reserve(kCount);
    for (std::size_t i = 0; i < kCount; ++i) {
        load_payloads[i].resize(kPayloadSize, 0);
        MemoryRegion region;
        region.memory_type = MemoryType::HOST;
        region.addr = reinterpret_cast<std::uint64_t>(load_payloads[i].data());
        region.size = kPayloadSize;
        Buffer buffer;
        buffer.region = region;
        load_entries.push_back(KVBuffer{keys[i], buffer});
    }

    TaskId load_id{kInvalidTaskId};
    status = client_->LoadAsync(load_entries, load_id);
    ASSERT_TRUE(status.ok()) << status.message;
    TaskResult load_result;
    status = WaitWithTimeout(*client_, load_id, 5000, load_result);
    ASSERT_TRUE(status.ok()) << status.message;

    for (std::size_t i = 0; i < kCount; ++i) {
        EXPECT_EQ(original_payloads[i].size(), load_payloads[i].size())
            << "size mismatch for key " << keys[i];
        EXPECT_EQ(std::memcmp(original_payloads[i].data(), load_payloads[i].data(),
                              original_payloads[i].size()),
                  0)
            << "data mismatch for key " << keys[i];
    }
}

TEST_F(AsuBenchmarkTest, QueryAccuracy_StoreThenQuery)
{
    constexpr std::size_t kCount = 32;
    constexpr std::size_t kPayloadSize = 1024;

    std::vector<std::vector<std::uint8_t>> payloads;
    auto entries = MakeKVBuffers(payloads, kCount, kPayloadSize, "query-");

    TaskId store_id{kInvalidTaskId};
    auto status = client_->StoreAsync(entries, store_id);
    ASSERT_TRUE(status.ok()) << status.message;
    TaskResult store_result;
    status = WaitWithTimeout(*client_, store_id, 5000, store_result);
    ASSERT_TRUE(status.ok()) << status.message;

    std::vector<CacheKey> stored_keys = ExtractKeys(entries);
    std::vector<CacheKey> missing_keys;
    for (std::size_t i = kCount; i < kCount + 8; ++i) {
        missing_keys.push_back("query-" + std::to_string(i));
    }
    std::vector<CacheKey> mixed_keys = stored_keys;
    mixed_keys.insert(mixed_keys.end(), missing_keys.begin(), missing_keys.end());

    QueryOptions opts;
    opts.timeout_ms = 5000;
    QueryResult result;
    status = client_->Query(mixed_keys, opts, result);
    ASSERT_TRUE(status.ok()) << status.message;
    ASSERT_EQ(result.exists.size(), mixed_keys.size());

    for (std::size_t i = 0; i < stored_keys.size(); ++i) {
        EXPECT_EQ(result.exists[i], 1) << "stored key " << stored_keys[i] << " should exist";
    }
    for (std::size_t i = stored_keys.size(); i < mixed_keys.size(); ++i) {
        EXPECT_EQ(result.exists[i], 0) << "missing key " << mixed_keys[i] << " should not exist";
    }
}

TEST_F(AsuBenchmarkTest, Latency_SingleStoreLoadQuery)
{
    constexpr std::size_t kCount = 10;
    constexpr std::size_t kPayloadSize = 4096;

    std::vector<std::vector<std::uint8_t>> payloads;
    auto entries = MakeKVBuffers(payloads, kCount, kPayloadSize, "lat-");
    auto keys = ExtractKeys(entries);

    std::vector<double> store_lats;
    std::vector<double> load_lats;
    std::vector<double> query_lats;

    for (std::size_t i = 0; i < kCount; ++i) {
        TaskId id{kInvalidTaskId};
        auto t0 = Clock::now();
        auto status = client_->StoreAsync({entries[i]}, id);
        ASSERT_TRUE(status.ok()) << status.message;
        TaskResult result;
        status = WaitWithTimeout(*client_, id, 5000, result);
        ASSERT_TRUE(status.ok()) << status.message;
        auto t1 = Clock::now();
        store_lats.push_back(Duration(t1 - t0).count() * 1000);
    }

    std::vector<std::vector<std::uint8_t>> load_payloads(kCount);
    std::vector<KVBuffer> load_entries(kCount);
    for (std::size_t i = 0; i < kCount; ++i) {
        load_payloads[i].resize(kPayloadSize, 0);
        MemoryRegion region;
        region.memory_type = MemoryType::HOST;
        region.addr = reinterpret_cast<std::uint64_t>(load_payloads[i].data());
        region.size = kPayloadSize;
        Buffer buffer;
        buffer.region = region;
        load_entries[i] = KVBuffer{keys[i], buffer};
    }

    for (std::size_t i = 0; i < kCount; ++i) {
        TaskId id{kInvalidTaskId};
        auto t0 = Clock::now();
        auto status = client_->LoadAsync({load_entries[i]}, id);
        ASSERT_TRUE(status.ok()) << status.message;
        TaskResult result;
        status = WaitWithTimeout(*client_, id, 5000, result);
        ASSERT_TRUE(status.ok()) << status.message;
        auto t1 = Clock::now();
        load_lats.push_back(Duration(t1 - t0).count() * 1000);
    }

    for (std::size_t i = 0; i < kCount; ++i) {
        QueryOptions opts;
        opts.timeout_ms = 5000;
        QueryResult qresult;
        auto t0 = Clock::now();
        auto status = client_->Query({keys[i]}, opts, qresult);
        ASSERT_TRUE(status.ok()) << status.message;
        auto t1 = Clock::now();
        query_lats.push_back(Duration(t1 - t0).count() * 1000);
    }

    PrintBench("StoreAsync (single entry)", ComputeBench(store_lats));
    PrintBench("LoadAsync  (single entry)", ComputeBench(load_lats));
    PrintBench("Query      (single key)  ", ComputeBench(query_lats));
}

TEST_F(AsuBenchmarkTest, Throughput_BulkStoreLoad)
{
    constexpr std::size_t kCount = 512;
    constexpr std::size_t kPayloadSize = 4096;

    std::vector<std::vector<std::uint8_t>> payloads;
    auto entries = MakeKVBuffers(payloads, kCount, kPayloadSize, "bulk-");
    auto keys = ExtractKeys(entries);

    auto t0 = Clock::now();
    TaskId store_id{kInvalidTaskId};
    auto status = client_->StoreAsync(entries, store_id);
    ASSERT_TRUE(status.ok()) << status.message;
    TaskResult store_result;
    status = WaitWithTimeout(*client_, store_id, 10000, store_result);
    ASSERT_TRUE(status.ok()) << status.message;
    auto t1 = Clock::now();
    double store_sec = Duration(t1 - t0).count();

    std::vector<std::vector<std::uint8_t>> load_payloads(kCount);
    std::vector<KVBuffer> load_entries(kCount);
    for (std::size_t i = 0; i < kCount; ++i) {
        load_payloads[i].resize(kPayloadSize, 0);
        MemoryRegion region;
        region.memory_type = MemoryType::HOST;
        region.addr = reinterpret_cast<std::uint64_t>(load_payloads[i].data());
        region.size = kPayloadSize;
        Buffer buffer;
        buffer.region = region;
        load_entries[i] = KVBuffer{keys[i], buffer};
    }

    auto t2 = Clock::now();
    TaskId load_id{kInvalidTaskId};
    status = client_->LoadAsync(load_entries, load_id);
    ASSERT_TRUE(status.ok()) << status.message;
    TaskResult load_result;
    status = WaitWithTimeout(*client_, load_id, 10000, load_result);
    ASSERT_TRUE(status.ok()) << status.message;
    auto t3 = Clock::now();
    double load_sec = Duration(t3 - t2).count();

    QueryOptions opts;
    opts.timeout_ms = 10000;
    QueryResult qresult;
    auto t4 = Clock::now();
    status = client_->Query(keys, opts, qresult);
    ASSERT_TRUE(status.ok()) << status.message;
    auto t5 = Clock::now();
    double query_sec = Duration(t5 - t4).count();

    std::printf("[BENCH] BulkStore:   %zu entries in %.3fms  throughput=%.1f entries/sec\n",
                kCount, store_sec * 1000, kCount / store_sec);
    std::printf("[BENCH] BulkLoad:    %zu entries in %.3fms  throughput=%.1f entries/sec\n",
                kCount, load_sec * 1000, kCount / load_sec);
    std::printf("[BENCH] BulkQuery:   %zu keys    in %.3fms  throughput=%.1f keys/sec\n",
                kCount, query_sec * 1000, kCount / query_sec);

    for (std::size_t i = 0; i < kCount; ++i) {
        EXPECT_EQ(std::memcmp(payloads[i].data(), load_payloads[i].data(), kPayloadSize), 0)
            << "data integrity failed for key " << keys[i];
    }
}

TEST_F(AsuBenchmarkTest, ConcurrentCompetition_MultiThreadStoreLoad)
{
    constexpr std::size_t kThreads = 4;
    constexpr std::size_t kOpsPerThread = 64;
    constexpr std::size_t kPayloadSize = 4096;

    std::atomic<std::size_t> total_success{0};
    std::atomic<std::size_t> total_fail{0};
    std::vector<std::future<std::vector<double>>> futures;
    futures.reserve(kThreads);

    for (std::size_t t = 0; t < kThreads; ++t) {
        futures.push_back(std::async(std::launch::async, [this, t, &total_success, &total_fail]() {
            std::vector<double> lats;
            std::vector<std::vector<std::uint8_t>> payloads;
            auto entries = MakeKVBuffers(payloads, kOpsPerThread, kPayloadSize,
                                        "conc-" + std::to_string(t) + "-");

            for (std::size_t i = 0; i < kOpsPerThread; ++i) {
                TaskId id{kInvalidTaskId};
                auto t0 = Clock::now();
                auto status = client_->StoreAsync({entries[i]}, id);
                if (!status.ok()) {
                    total_fail.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                TaskResult result;
                status = WaitWithTimeout(*client_, id, 5000, result);
                if (!status.ok()) {
                    total_fail.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                auto t1 = Clock::now();
                lats.push_back(Duration(t1 - t0).count() * 1000);
                total_success.fetch_add(1, std::memory_order_relaxed);
            }

            std::vector<std::vector<std::uint8_t>> load_payloads(kOpsPerThread);
            std::vector<KVBuffer> load_entries(kOpsPerThread);
            auto keys = ExtractKeys(entries);
            for (std::size_t i = 0; i < kOpsPerThread; ++i) {
                load_payloads[i].resize(kPayloadSize, 0);
                MemoryRegion region;
                region.memory_type = MemoryType::HOST;
                region.addr = reinterpret_cast<std::uint64_t>(load_payloads[i].data());
                region.size = kPayloadSize;
                Buffer buffer;
                buffer.region = region;
                load_entries[i] = KVBuffer{keys[i], buffer};
            }

            for (std::size_t i = 0; i < kOpsPerThread; ++i) {
                TaskId id{kInvalidTaskId};
                auto t0 = Clock::now();
                auto status = client_->LoadAsync({load_entries[i]}, id);
                if (!status.ok()) {
                    total_fail.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                TaskResult result;
                status = WaitWithTimeout(*client_, id, 5000, result);
                if (!status.ok()) {
                    total_fail.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                auto t1 = Clock::now();
                lats.push_back(Duration(t1 - t0).count() * 1000);
                total_success.fetch_add(1, std::memory_order_relaxed);
            }

            return lats;
        }));
    }

    std::vector<double> all_lats;
    for (auto& f : futures) {
        auto lats = f.get();
        all_lats.insert(all_lats.end(), lats.begin(), lats.end());
    }

    auto bench = ComputeBench(all_lats);
    PrintBench("ConcurrentStoreLoad", bench);
    std::printf("[BENCH] ConcurrentStoreLoad: success=%zu  fail=%zu  threads=%zu  "
                "ops_per_thread=%zu\n",
                total_success.load(), total_fail.load(), kThreads, kOpsPerThread);

    EXPECT_EQ(total_fail.load(), 0u);
}

TEST_F(AsuBenchmarkTest, LargeScaleStress_ManyKeysLargePayload)
{
    constexpr std::size_t kCount = 2048;
    constexpr std::size_t kPayloadSize = 8192;

    std::vector<std::vector<std::uint8_t>> payloads;
    auto entries = MakeKVBuffers(payloads, kCount, kPayloadSize, "stress-");
    auto keys = ExtractKeys(entries);

    auto t0 = Clock::now();
    TaskId store_id{kInvalidTaskId};
    auto status = client_->StoreAsync(entries, store_id);
    ASSERT_TRUE(status.ok()) << status.message;
    TaskResult store_result;
    status = WaitWithTimeout(*client_, store_id, 30000, store_result);
    ASSERT_TRUE(status.ok()) << status.message;
    auto t1 = Clock::now();
    double store_sec = Duration(t1 - t0).count();

    std::vector<std::vector<std::uint8_t>> load_payloads(kCount);
    std::vector<KVBuffer> load_entries(kCount);
    for (std::size_t i = 0; i < kCount; ++i) {
        load_payloads[i].resize(kPayloadSize, 0);
        MemoryRegion region;
        region.memory_type = MemoryType::HOST;
        region.addr = reinterpret_cast<std::uint64_t>(load_payloads[i].data());
        region.size = kPayloadSize;
        Buffer buffer;
        buffer.region = region;
        load_entries[i] = KVBuffer{keys[i], buffer};
    }

    auto t2 = Clock::now();
    TaskId load_id{kInvalidTaskId};
    status = client_->LoadAsync(load_entries, load_id);
    ASSERT_TRUE(status.ok()) << status.message;
    TaskResult load_result;
    status = WaitWithTimeout(*client_, load_id, 30000, load_result);
    ASSERT_TRUE(status.ok()) << status.message;
    auto t3 = Clock::now();
    double load_sec = Duration(t3 - t2).count();

    QueryOptions opts;
    opts.timeout_ms = 30000;
    QueryResult qresult;
    auto t4 = Clock::now();
    status = client_->Query(keys, opts, qresult);
    ASSERT_TRUE(status.ok()) << status.message;
    auto t5 = Clock::now();
    double query_sec = Duration(t5 - t4).count();

    std::printf("[BENCH] StressStore: %zu entries x %zu bytes  total=%.3fms  "
                "throughput=%.1f entries/sec  data=%.1f MB/sec\n",
                kCount, kPayloadSize, store_sec * 1000, kCount / store_sec,
                (kCount * kPayloadSize) / (store_sec * 1024 * 1024));
    std::printf("[BENCH] StressLoad:  %zu entries x %zu bytes  total=%.3fms  "
                "throughput=%.1f entries/sec  data=%.1f MB/sec\n",
                kCount, kPayloadSize, load_sec * 1000, kCount / load_sec,
                (kCount * kPayloadSize) / (load_sec * 1024 * 1024));
    std::printf("[BENCH] StressQuery: %zu keys               total=%.3fms  "
                "throughput=%.1f keys/sec\n",
                kCount, query_sec * 1000, kCount / query_sec);

    std::size_t integrity_fail = 0;
    for (std::size_t i = 0; i < kCount; ++i) {
        if (std::memcmp(payloads[i].data(), load_payloads[i].data(), kPayloadSize) != 0) {
            ++integrity_fail;
        }
    }
    EXPECT_EQ(integrity_fail, 0u) << "data integrity failures in stress test";

    for (std::size_t i = 0; i < kCount; ++i) {
        EXPECT_EQ(qresult.exists[i], 1) << "key " << keys[i] << " should exist after store";
    }
}

TEST_F(AsuBenchmarkTest, DeleteUnsupported_VerifyError)
{
    std::vector<CacheKey> keys{"del-key-0", "del-key-1"};
    TaskId delete_id{kInvalidTaskId};
    auto status = client_->DeleteAsync(keys, delete_id);
    ASSERT_EQ(status.code, StatusCode::UNSUPPORTED);
    ASSERT_EQ(delete_id, kInvalidTaskId);
}



class AsuRdmaMockBenchmarkTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        rdma_stats_ = std::make_shared<MockIoStats>();

        MemoryIoBackendOptions options;
        options.enable_rdma_stats = true;
        options.rdma_per_entry = true;

        // Set to 0 for stable and fast unit tests. Increase this value if you want
        // to simulate RDMA latency, for example 2/5/10 us per entry/key.
        options.fixed_latency_us = 0;

        // 0 means no bandwidth-based sleep. To simulate bandwidth, set for example:
        // 100ULL * 1024 * 1024 * 1024 for 100GB/s.
        options.bandwidth_bytes_per_sec = 0;
        options.jitter_us = 0;
        options.error_rate = 0.0;
        options.min_query_bytes = 64;
        options.min_delete_bytes = 64;

        client_ = CreateRdmaMockBenchClient(rdma_stats_, options);
        ASSERT_NE(client_, nullptr);

        auto status = client_->Init(MakeBenchConfig());
        ASSERT_TRUE(status.ok()) << status.message;
    }

    void TearDown() override
    {
        if (client_) {
            auto status = client_->Shutdown();
            ASSERT_TRUE(status.ok()) << status.message;
        }
    }

    std::shared_ptr<MockIoStats> rdma_stats_;
    std::unique_ptr<AsuClient> client_;
};

TEST_F(AsuRdmaMockBenchmarkTest, RdmaMock_EachEntryOneTransfer)
{
    constexpr std::size_t kCount = 8;
    constexpr std::size_t kPayloadSize = 4096;

    std::vector<std::vector<std::uint8_t>> payloads;
    auto entries = MakeKVBuffers(payloads, kCount, kPayloadSize, "rdma-entry-");
    auto keys = ExtractKeys(entries);

    rdma_stats_->Reset();

    TaskId store_id{kInvalidTaskId};
    auto status = client_->StoreAsync(entries, store_id);
    ASSERT_TRUE(status.ok()) << status.message;

    TaskResult store_result;
    status = WaitWithTimeout(*client_, store_id, 5000, store_result);
    ASSERT_TRUE(status.ok()) << status.message;

    EXPECT_EQ(rdma_stats_->store_transfers.load(std::memory_order_relaxed), kCount);
    EXPECT_EQ(rdma_stats_->store_bytes.load(std::memory_order_relaxed), kCount * kPayloadSize);

    std::vector<std::vector<std::uint8_t>> load_payloads(kCount);
    std::vector<KVBuffer> load_entries;
    load_entries.reserve(kCount);

    for (std::size_t i = 0; i < kCount; ++i) {
        load_payloads[i].resize(kPayloadSize, 0);

        MemoryRegion region;
        region.memory_type = MemoryType::HOST;
        region.addr = reinterpret_cast<std::uint64_t>(load_payloads[i].data());
        region.size = kPayloadSize;

        Buffer buffer;
        buffer.region = region;
        load_entries.push_back(KVBuffer{keys[i], buffer});
    }

    TaskId load_id{kInvalidTaskId};
    status = client_->LoadAsync(load_entries, load_id);
    ASSERT_TRUE(status.ok()) << status.message;

    TaskResult load_result;
    status = WaitWithTimeout(*client_, load_id, 5000, load_result);
    ASSERT_TRUE(status.ok()) << status.message;

    EXPECT_EQ(rdma_stats_->load_transfers.load(std::memory_order_relaxed), kCount);
    EXPECT_EQ(rdma_stats_->load_bytes.load(std::memory_order_relaxed), kCount * kPayloadSize);

    for (std::size_t i = 0; i < kCount; ++i) {
        EXPECT_EQ(std::memcmp(payloads[i].data(), load_payloads[i].data(), kPayloadSize), 0)
            << "data mismatch for key " << keys[i];
    }

    QueryOptions opts;
    opts.timeout_ms = 5000;

    QueryResult qresult;
    status = client_->Query(keys, opts, qresult);
    ASSERT_TRUE(status.ok()) << status.message;
    ASSERT_EQ(qresult.exists.size(), kCount);

    EXPECT_EQ(rdma_stats_->query_transfers.load(std::memory_order_relaxed), kCount);
    EXPECT_GE(rdma_stats_->query_bytes.load(std::memory_order_relaxed), kCount * 64);

    for (std::size_t i = 0; i < kCount; ++i) {
        EXPECT_EQ(qresult.exists[i], 1) << "key should exist: " << keys[i];
    }

    std::printf("[BENCH][RDMA-MOCK] transfers: store=%llu load=%llu query=%llu\n",
                static_cast<unsigned long long>(
                    rdma_stats_->store_transfers.load(std::memory_order_relaxed)),
                static_cast<unsigned long long>(
                    rdma_stats_->load_transfers.load(std::memory_order_relaxed)),
                static_cast<unsigned long long>(
                    rdma_stats_->query_transfers.load(std::memory_order_relaxed)));
}

TEST(AsuRdmaMockBenchmarkStandaloneTest, RdmaMock_LatencyWithPerEntryTransfer)
{
    auto rdma_stats = std::make_shared<MockIoStats>();

    MemoryIoBackendOptions options;
    options.enable_rdma_stats = true;
    options.rdma_per_entry = true;

    // Simulate one RDMA operation per entry/key with fixed 10us latency.
    options.fixed_latency_us = 10;
    options.bandwidth_bytes_per_sec = 0;
    options.jitter_us = 0;
    options.error_rate = 0.0;
    options.min_query_bytes = 64;
    options.min_delete_bytes = 64;

    auto client = CreateRdmaMockBenchClient(rdma_stats, options);
    ASSERT_NE(client, nullptr);

    auto status = client->Init(MakeBenchConfig());
    ASSERT_TRUE(status.ok()) << status.message;

    constexpr std::size_t kCount = 128;
    constexpr std::size_t kPayloadSize = 4096;

    std::vector<std::vector<std::uint8_t>> payloads;
    auto entries = MakeKVBuffers(payloads, kCount, kPayloadSize, "rdma-lat-");
    auto keys = ExtractKeys(entries);

    auto t0 = Clock::now();

    TaskId store_id{kInvalidTaskId};
    status = client->StoreAsync(entries, store_id);
    ASSERT_TRUE(status.ok()) << status.message;

    TaskResult store_result;
    status = WaitWithTimeout(*client, store_id, 10000, store_result);
    ASSERT_TRUE(status.ok()) << status.message;

    auto t1 = Clock::now();

    QueryOptions opts;
    opts.timeout_ms = 10000;

    QueryResult qresult;
    status = client->Query(keys, opts, qresult);
    ASSERT_TRUE(status.ok()) << status.message;

    auto t2 = Clock::now();

    const double store_ms = Duration(t1 - t0).count() * 1000;
    const double query_ms = Duration(t2 - t1).count() * 1000;

    std::printf("[BENCH][RDMA-MOCK] Store: entries=%zu payload=%zuB "
                "rdma_transfers=%llu total=%.3fms\n",
                kCount, kPayloadSize,
                static_cast<unsigned long long>(
                    rdma_stats->store_transfers.load(std::memory_order_relaxed)),
                store_ms);

    std::printf("[BENCH][RDMA-MOCK] Query: keys=%zu "
                "rdma_transfers=%llu total=%.3fms\n",
                kCount,
                static_cast<unsigned long long>(
                    rdma_stats->query_transfers.load(std::memory_order_relaxed)),
                query_ms);

    EXPECT_EQ(rdma_stats->store_transfers.load(std::memory_order_relaxed), kCount);
    EXPECT_EQ(rdma_stats->store_bytes.load(std::memory_order_relaxed), kCount * kPayloadSize);
    EXPECT_EQ(rdma_stats->query_transfers.load(std::memory_order_relaxed), kCount);

    status = client->Shutdown();
    ASSERT_TRUE(status.ok()) << status.message;
}



TEST(AsuBenchmarkStressTest, InFlightStoreLoadQueryPressure)
{ 
    constexpr std::size_t kTransportNum = 4;
    constexpr std::size_t kTaskCount = 2048;
    constexpr std::size_t kEntriesPerTask = 64;
    constexpr std::size_t kPayloadSize = 4096;

    auto rdma_stats = std::make_shared<MockIoStats>();

    MemoryIoBackendOptions options;
    options.enable_rdma_stats = true;
    options.rdma_per_entry = true;
    options.fixed_latency_us = 0;//测试目的：并发吞吐量和数据完整性，不关注延迟
    options.bandwidth_bytes_per_sec = 0;
    options.jitter_us = 0;
    options.error_rate = 0.0;
    options.min_query_bytes = 64;
    options.min_delete_bytes = 64;

    auto client = CreateRdmaMockBenchClient(rdma_stats, options);
    ASSERT_NE(client, nullptr);

    auto status = client->Init(MakeBenchConfig(kTransportNum));
    ASSERT_TRUE(status.ok()) << status.message;

    std::vector<StressTaskData> store_tasks;
    store_tasks.reserve(kTaskCount);

    auto t0 = Clock::now();

    for (std::size_t i = 0; i < kTaskCount; ++i) {
        store_tasks.push_back(MakeStressStoreTask(i, kEntriesPerTask, kPayloadSize, "rdma-stress-"));

        status = client->StoreAsync(store_tasks.back().entries, store_tasks.back().task_id);
        ASSERT_TRUE(status.ok()) << status.message;
        ASSERT_NE(store_tasks.back().task_id, kInvalidTaskId);
    }

    for (auto& task : store_tasks) {
        TaskResult result;
        status = WaitWithTimeout(*client, task.task_id, 30000, result);
        ASSERT_TRUE(status.ok()) << status.message;
    }

    auto t1 = Clock::now();

    const auto expected_entry_count = kTaskCount * kEntriesPerTask;
    EXPECT_EQ(rdma_stats->store_transfers.load(std::memory_order_relaxed), expected_entry_count);
    EXPECT_EQ(rdma_stats->store_bytes.load(std::memory_order_relaxed), expected_entry_count * kPayloadSize);

    std::vector<CacheKey> all_keys;
    all_keys.reserve(expected_entry_count);
    for (const auto& task : store_tasks) {
        all_keys.insert(all_keys.end(), task.keys.begin(), task.keys.end());
    }

    QueryOptions query_options;
    query_options.timeout_ms = 30000;

    QueryResult query_result;
    status = client->Query(all_keys, query_options, query_result);
    ASSERT_TRUE(status.ok()) << status.message;
    ASSERT_EQ(query_result.exists.size(), all_keys.size());

    for (std::size_t i = 0; i < query_result.exists.size(); ++i) {
        ASSERT_EQ(query_result.exists[i], 1) << "missing key: " << all_keys[i];
    }

    auto t2 = Clock::now();

    EXPECT_EQ(rdma_stats->query_transfers.load(std::memory_order_relaxed), expected_entry_count);

    std::vector<StressTaskData> load_tasks;
    load_tasks.reserve(kTaskCount);

    for (std::size_t i = 0; i < kTaskCount; ++i) {
        load_tasks.push_back(MakeStressLoadTask(store_tasks[i].keys, kPayloadSize));

        status = client->LoadAsync(load_tasks.back().entries, load_tasks.back().task_id);
        ASSERT_TRUE(status.ok()) << status.message;
        ASSERT_NE(load_tasks.back().task_id, kInvalidTaskId);
    }

    for (auto& task : load_tasks) {
        TaskResult result;
        status = WaitWithTimeout(*client, task.task_id, 30000, result);
        ASSERT_TRUE(status.ok()) << status.message;
    }

    auto t3 = Clock::now();

    EXPECT_EQ(rdma_stats->load_transfers.load(std::memory_order_relaxed), expected_entry_count);
    EXPECT_EQ(rdma_stats->load_bytes.load(std::memory_order_relaxed), expected_entry_count * kPayloadSize);

    for (std::size_t task_idx : {std::size_t{0}, kTaskCount / 2, kTaskCount - 1}) {
        for (std::size_t entry_idx = 0; entry_idx < kEntriesPerTask; ++entry_idx) {
            EXPECT_EQ(std::memcmp(store_tasks[task_idx].payloads[entry_idx].data(),
                                  load_tasks[task_idx].payloads[entry_idx].data(), kPayloadSize),
                      0)
                << "data mismatch, task=" << task_idx << ", entry=" << entry_idx;
        }
    }

    const double store_ms = Duration(t1 - t0).count() * 1000.0;
    const double query_ms = Duration(t2 - t1).count() * 1000.0;
    const double load_ms = Duration(t3 - t2).count() * 1000.0;
    const double total_ms = Duration(t3 - t0).count() * 1000.0;

    std::printf("[STRESS][RDMA-MOCK] transports=%zu tasks=%zu entries_per_task=%zu payload=%zuB\n",
                kTransportNum, kTaskCount, kEntriesPerTask, kPayloadSize);
    std::printf("[STRESS][RDMA-MOCK] store_transfers=%llu load_transfers=%llu query_transfers=%llu\n",
                static_cast<unsigned long long>(rdma_stats->store_transfers.load(std::memory_order_relaxed)),
                static_cast<unsigned long long>(rdma_stats->load_transfers.load(std::memory_order_relaxed)),
                static_cast<unsigned long long>(rdma_stats->query_transfers.load(std::memory_order_relaxed)));
    std::printf("[STRESS][RDMA-MOCK] store=%.3fms query=%.3fms load=%.3fms total=%.3fms\n",
                store_ms, query_ms, load_ms, total_ms);

    status = client->Shutdown();
    ASSERT_TRUE(status.ok()) << status.message;
}

TEST(AsuBenchmarkStressTest, MultiThreadSubmitWaitPressure)
{
    constexpr std::size_t kTransportNum = 4;
    constexpr std::size_t kThreadNum = 8;
    constexpr std::size_t kOpsPerThread = 512;
    constexpr std::size_t kEntriesPerOp = 2;
    constexpr std::size_t kPayloadSize = 1024;
    constexpr std::size_t kTotalOps = kThreadNum * kOpsPerThread;
    constexpr std::size_t kTotalEntries = kTotalOps * kEntriesPerOp;

    auto rdma_stats = std::make_shared<MockIoStats>();

    MemoryIoBackendOptions options;
    options.enable_rdma_stats = true;
    options.rdma_per_entry = true;
    options.fixed_latency_us = 0;
    options.bandwidth_bytes_per_sec = 0;
    options.jitter_us = 0;
    options.error_rate = 0.0;
    options.min_query_bytes = 64;
    options.min_delete_bytes = 64;

    auto client = CreateRdmaMockBenchClient(rdma_stats, options);
    ASSERT_NE(client, nullptr);

    auto status = client->Init(MakeBenchConfig(kTransportNum));
    ASSERT_TRUE(status.ok()) << status.message;

    std::atomic<std::size_t> success_ops{0};
    std::atomic<std::size_t> failed_ops{0};
    std::atomic<std::size_t> task_index{0};

    std::vector<StressTaskData> all_tasks(kTotalOps);

    auto t0 = Clock::now();

    std::vector<std::thread> threads;
    threads.reserve(kThreadNum);

    for (std::size_t tid = 0; tid < kThreadNum; ++tid) {
        threads.emplace_back([&, tid]() {
            for (std::size_t op = 0; op < kOpsPerThread; ++op) {
                const auto global_index = tid * kOpsPerThread + op;
                auto task = MakeStressStoreTask(global_index, kEntriesPerOp, kPayloadSize, "rdma-mt-");

                TaskId task_id{kInvalidTaskId};
                auto s = client->StoreAsync(task.entries, task_id);
                if (!s.ok()) {
                    failed_ops.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }

                TaskResult result;
                s = WaitWithTimeout(*client, task_id, 30000, result);
                if (!s.ok()) {
                    failed_ops.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }

                const auto idx = task_index.fetch_add(1, std::memory_order_relaxed);
                all_tasks[idx] = std::move(task);
                success_ops.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    for (auto& th : threads) {
        th.join();
    }

    auto t1 = Clock::now();

    const auto ok_ops = success_ops.load(std::memory_order_relaxed);
    const auto bad_ops = failed_ops.load(std::memory_order_relaxed);

    EXPECT_EQ(bad_ops, 0);
    EXPECT_EQ(ok_ops, kTotalOps);

    const auto expected_store_transfers = ok_ops * kEntriesPerOp;
    EXPECT_EQ(rdma_stats->store_transfers.load(std::memory_order_relaxed), expected_store_transfers);
    EXPECT_EQ(rdma_stats->store_bytes.load(std::memory_order_relaxed), expected_store_transfers * kPayloadSize);

    std::vector<std::vector<std::uint8_t>> load_payloads(kTotalEntries);
    std::vector<KVBuffer> load_entries;
    load_entries.reserve(kTotalEntries);

    for (std::size_t i = 0; i < ok_ops; ++i) {
        for (std::size_t j = 0; j < kEntriesPerOp; ++j) {
            const auto entry_idx = i * kEntriesPerOp + j;
            load_payloads[entry_idx].resize(kPayloadSize, 0);

            MemoryRegion region;
            region.memory_type = MemoryType::HOST;
            region.addr = reinterpret_cast<std::uint64_t>(load_payloads[entry_idx].data());
            region.size = kPayloadSize;

            Buffer buffer;
            buffer.region = region;

            load_entries.push_back(KVBuffer{all_tasks[i].keys[j], buffer});
        }
    }

    TaskId load_id{kInvalidTaskId};
    status = client->LoadAsync(load_entries, load_id);
    ASSERT_TRUE(status.ok()) << status.message;

    TaskResult load_result;
    status = WaitWithTimeout(*client, load_id, 30000, load_result);
    ASSERT_TRUE(status.ok()) << status.message;

    std::size_t integrity_failures = 0;
    for (std::size_t i = 0; i < ok_ops; ++i) {
        for (std::size_t j = 0; j < kEntriesPerOp; ++j) {
            const auto entry_idx = i * kEntriesPerOp + j;
            if (std::memcmp(all_tasks[i].payloads[j].data(),
                            load_payloads[entry_idx].data(),
                            kPayloadSize) != 0) {
                ++integrity_failures;
            }
        }
    }
    EXPECT_EQ(integrity_failures, 0u) << "data integrity check failed";

    const double total_ms = Duration(t1 - t0).count() * 1000.0;
    const double ops_per_sec = total_ms > 0.0 ? static_cast<double>(ok_ops) / (total_ms / 1000.0) : 0.0;

    std::printf("[STRESS][RDMA-MOCK][MT] threads=%zu ops_per_thread=%zu entries_per_op=%zu payload=%zuB\n",
                kThreadNum, kOpsPerThread, kEntriesPerOp, kPayloadSize);
    std::printf("[STRESS][RDMA-MOCK][MT] success_ops=%zu failed_ops=%zu store_transfers=%llu total=%.3fms throughput=%.1f ops/sec\n",
                ok_ops, bad_ops,
                static_cast<unsigned long long>(rdma_stats->store_transfers.load(std::memory_order_relaxed)),
                total_ms, ops_per_sec);
    std::printf("[STRESS][RDMA-MOCK][MT] integrity_check: entries=%zu failures=%zu\n",
                kTotalEntries, integrity_failures);

    status = client->Shutdown();
    ASSERT_TRUE(status.ok()) << status.message;
}


}  // namespace UC::ASU