#include "transport_task_manager.h"
#include "connection_internal.h"
#include "logger.h"

namespace UC::ASU {

bool TransportTaskContext::Done() const
{
    auto s = state.load(std::memory_order_acquire);
    return s == TransportTaskState::COMPLETED || s == TransportTaskState::FAILED ||
           s == TransportTaskState::CANCELED;
}

// Stub for testing, remove after real implementation
bool TransportTaskContext::StubDone()
{
    auto s = state.load(std::memory_order_acquire);
    if (s == TransportTaskState::COMPLETED || s == TransportTaskState::FAILED ||
        s == TransportTaskState::CANCELED) {
        UC_DEBUG("TransportTaskContext::StubDone taskId={} state={} (terminal)", taskId,
                 static_cast<int>(s));
        return true;
    }
    if (s == TransportTaskState::INFLIGHT) {
        if (flagbufferStatus.load(std::memory_order_acquire) >= 1) {
            UC_DEBUG(
                "TransportTaskContext::StubDone taskId={} INFLIGHT + flagbuffer ready->COMPLETED",
                taskId);
            finalStatus = Status::OK();
            if (opType == TransportOpType::QUERY) {
                queryResult.exists.assign(keys.size, 0);
                queryResult.prefixHitKeys = 0;
            }
            state.store(TransportTaskState::COMPLETED, std::memory_order_release);
            auto* ch = this->channel.load(std::memory_order_acquire);
            if (ch) {
                UC_DEBUG("TransportTaskContext::StubDone inflight-1 on ch_id={}",
                         ch->GetChannelId());
                ch->ReleaseInflight();
            }
            return true;
        } else {
            UC_DEBUG("TransportTaskContext::StubDone taskId={} INFLIGHT but flagbuffer not ready",
                     taskId);
        }
    }
    return false;
}

void TransportTaskManager::Shutdown()
{
    for (const auto& ctx : GetAll()) {
        if (ctx == nullptr) { continue; }

        std::function<void(TaskId)> callback;
        {
            std::lock_guard<std::mutex> lock(ctx->waitMu);
            if (ctx->Done()) { continue; }

            ctx->finalStatus =
                Status::Error(StatusCode::CANCELED, "transport shutdown canceled task");
            ctx->state.store(TransportTaskState::CANCELED, std::memory_order_release);
            callback = std::move(ctx->completionCallback);
            ctx->cv.notify_all();
        }
        if (callback) { callback(ctx->taskId); }
    }
}

}  // namespace UC::ASU
