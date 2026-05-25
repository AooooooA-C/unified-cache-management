#include "asu_client/asu_client.h"
#include "io_backend.h"
#include "memory_io_backend.h"
#include "mock_io_backend.h"
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

// std::unique_ptr<AsuClient> CreateBenchClient()
// {
//     auto factory = []() -> std::unique_ptr<AsuTransport> {
//         return std::make_unique<AsuTransportImpl>(CreateMemoryIoBackend());
//     };
//     return CreateAsuClient(factory);
// }


std::unique_ptr<AsuClient> CreateBenchClient()
{
    auto factory = []() -> std::unique_ptr<AsuTransport> {
        MockIoBackendOptions options;
        options.fixed_latency_us = 1000;  // 每次 IO 固定 1ms
        options.bandwidth_bytes_per_sec = 1024ULL * 1024 * 1024;  // 1GB/s
        options.jitter_us = 100;  // 0~100us 抖动
        options.error_rate = 0.0;  // 不注入错误

        return std::make_unique<AsuTransportImpl>(
            CreateMockIoBackend(options, CreateMemoryIoBackend()));
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

}  // namespace UC::ASU