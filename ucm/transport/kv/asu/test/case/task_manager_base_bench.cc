/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "transport_task_manager.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

namespace UC::ASU {
namespace {

using Clock = std::chrono::steady_clock;

std::size_t GetEnvSize(const char* name, std::size_t fallback)
{
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') { return fallback; }

    char* end = nullptr;
    const auto parsed = std::strtoull(value, &end, 10);
    if (end == value || parsed == 0) { return fallback; }
    return static_cast<std::size_t>(parsed);
}

double ToNsPerOp(Clock::duration elapsed, std::size_t ops)
{
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
    return static_cast<double>(ns) / static_cast<double>(std::max<std::size_t>(ops, 1));
}

void PrintBenchResult(const std::string& name, std::size_t threads, std::size_t ops,
                      Clock::duration elapsed)
{
    const auto seconds = std::chrono::duration<double>(elapsed).count();
    const auto throughput = static_cast<double>(ops) / std::max(seconds, 1e-9);
    std::cout << "[task-manager-bench] " << name << " threads=" << threads
              << " ops=" << ops << " seconds=" << seconds
              << " ns_per_op=" << ToNsPerOp(elapsed, ops)
              << " throughput_ops_per_sec=" << throughput << std::endl;
}

void RunSubmitRemoveBench(const std::string& name, std::size_t threadCount,
                          std::size_t opsPerThread)
{
    TransportTaskManager manager;
    std::atomic<std::size_t> completedOps{0};
    std::vector<std::thread> threads;
    threads.reserve(threadCount);

    const auto start = Clock::now();
    for (std::size_t tid = 0; tid < threadCount; ++tid) {
        threads.emplace_back([&]() {
            for (std::size_t i = 0; i < opsPerThread; ++i) {
                auto ctx = std::make_unique<TransportTaskContext>();
                TaskId taskId{kInvalidTaskId};
                auto status = manager.Submit(std::move(ctx), taskId);
                if (!status.ok()) { continue; }
                completedOps.fetch_add(1, std::memory_order_relaxed);

                status = manager.Remove(taskId);
                if (status.ok()) { completedOps.fetch_add(1, std::memory_order_relaxed); }
            }
        });
    }

    for (auto& thread : threads) { thread.join(); }
    const auto elapsed = Clock::now() - start;

    PrintBenchResult(name, threadCount, completedOps.load(), elapsed);
}

void RunSubmitGetRemoveBench(const std::string& name, std::size_t threadCount,
                             std::size_t opsPerThread)
{
    TransportTaskManager manager;
    std::atomic<std::size_t> completedOps{0};
    std::vector<std::thread> threads;
    threads.reserve(threadCount);

    const auto start = Clock::now();
    for (std::size_t tid = 0; tid < threadCount; ++tid) {
        threads.emplace_back([&]() {
            for (std::size_t i = 0; i < opsPerThread; ++i) {
                auto ctx = std::make_unique<TransportTaskContext>();
                TaskId taskId{kInvalidTaskId};
                auto status = manager.Submit(std::move(ctx), taskId);
                if (!status.ok()) { continue; }
                completedOps.fetch_add(1, std::memory_order_relaxed);

                auto task = manager.Get(taskId);
                if (task != nullptr) { completedOps.fetch_add(1, std::memory_order_relaxed); }

                status = manager.Remove(taskId);
                if (status.ok()) { completedOps.fetch_add(1, std::memory_order_relaxed); }
            }
        });
    }

    for (auto& thread : threads) { thread.join(); }
    const auto elapsed = Clock::now() - start;

    PrintBenchResult(name, threadCount, completedOps.load(), elapsed);
}

void RunGetAllBench(const std::string& name, std::size_t liveTasks, std::size_t iterations)
{
    TransportTaskManager manager;

    for (std::size_t i = 0; i < liveTasks; ++i) {
        auto ctx = std::make_unique<TransportTaskContext>();
        TaskId taskId{kInvalidTaskId};
        auto status = manager.Submit(std::move(ctx), taskId);
        ASSERT_TRUE(status.ok());
    }

    std::size_t observedTasks = 0;
    const auto start = Clock::now();
    for (std::size_t i = 0; i < iterations; ++i) {
        observedTasks += manager.GetAll().size();
    }
    const auto elapsed = Clock::now() - start;

    ASSERT_EQ(observedTasks, liveTasks * iterations);
    PrintBenchResult(name, 1, iterations, elapsed);
}

TEST(TaskManagerBaseBench, DISABLED_SingleThreadSubmitRemove)
{
    const auto iterations = GetEnvSize("ASU_TASK_MANAGER_BENCH_ITERS", 200000);
    RunSubmitRemoveBench("single_thread_submit_remove", 1, iterations);
}

TEST(TaskManagerBaseBench, DISABLED_MultiThreadSubmitRemove)
{
    const auto threadCount = GetEnvSize(
        "ASU_TASK_MANAGER_BENCH_THREADS",
        std::max<std::size_t>(std::thread::hardware_concurrency(), 1));
    const auto opsPerThread = GetEnvSize("ASU_TASK_MANAGER_BENCH_ITERS", 50000);
    RunSubmitRemoveBench("multi_thread_submit_remove", threadCount, opsPerThread);
}

TEST(TaskManagerBaseBench, DISABLED_MultiThreadSubmitGetRemove)
{
    const auto threadCount = GetEnvSize(
        "ASU_TASK_MANAGER_BENCH_THREADS",
        std::max<std::size_t>(std::thread::hardware_concurrency(), 1));
    const auto opsPerThread = GetEnvSize("ASU_TASK_MANAGER_BENCH_ITERS", 50000);
    RunSubmitGetRemoveBench("multi_thread_submit_get_remove", threadCount, opsPerThread);
}

TEST(TaskManagerBaseBench, DISABLED_GetAllLiveTasks)
{
    const auto liveTasks = GetEnvSize("ASU_TASK_MANAGER_BENCH_LIVE_TASKS", 4096);
    const auto iterations = GetEnvSize("ASU_TASK_MANAGER_BENCH_ITERS", 10000);
    RunGetAllBench("get_all_live_tasks", liveTasks, iterations);
}

TEST(TaskManagerBaseBench, DISABLED_ThreadCountSweepSubmitRemove)
{
    constexpr std::array<std::size_t, 6> kThreadCounts{1, 2, 4, 8, 16, 32};
    const auto opsPerThread = GetEnvSize("ASU_TASK_MANAGER_BENCH_ITERS", 50000);

    for (const auto threadCount : kThreadCounts) {
        RunSubmitRemoveBench("thread_sweep_submit_remove", threadCount, opsPerThread);
    }
}

TEST(TaskManagerBaseBench, DISABLED_ThreadCountSweepSubmitGetRemove)
{
    constexpr std::array<std::size_t, 6> kThreadCounts{1, 2, 4, 8, 16, 32};
    const auto opsPerThread = GetEnvSize("ASU_TASK_MANAGER_BENCH_ITERS", 50000);

    for (const auto threadCount : kThreadCounts) {
        RunSubmitGetRemoveBench("thread_sweep_submit_get_remove", threadCount, opsPerThread);
    }
}

TEST(TaskManagerBaseBench, DISABLED_GetAllLiveTaskCountSweep)
{
    constexpr std::array<std::size_t, 5> kLiveTaskCounts{64, 256, 1024, 4096, 8192};
    const auto iterations = GetEnvSize("ASU_TASK_MANAGER_BENCH_ITERS", 10000);

    for (const auto liveTasks : kLiveTaskCounts) {
        RunGetAllBench("get_all_live_task_sweep/live_tasks=" + std::to_string(liveTasks),
                       liveTasks, iterations);
    }
}

}  // namespace
}  // namespace UC::ASU
