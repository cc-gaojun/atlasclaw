from __future__ import annotations

import logging
import time
from typing import Any, AsyncIterator

from app.atlasclaw.agent.stream import StreamEvent


logger = logging.getLogger(__name__)


class RunnerExecutionFlowErrorMixin:
    async def _handle_loop_phase_exception(
        self,
        *,
        error: Exception,
        state: dict[str, Any],
    ) -> AsyncIterator[StreamEvent]:
        """Handle runtime exceptions within loop phase and attempt token failover."""
        logger.exception("Agent runtime exception during streaming run")
        if bool(state.get("answer_committed")):
            error_text = str(error).strip() or error.__class__.__name__
            yield StreamEvent.runtime_update(
                "warning",
                f"Post-answer side effect failed: {error_text}",
                metadata={
                    "phase": "post_answer_exception",
                    "elapsed": round(time.monotonic() - float(state.get("start_time") or 0.0), 1),
                },
            )
            return

        error_text = str(error).strip()
        if not error_text:
            if state.get("model_stream_timed_out") and state.get("model_timeout_error_message"):
                error_text = str(state.get("model_timeout_error_message"))
            else:
                error_text = error.__class__.__name__

        retry_error: Exception = error
        if not str(error).strip():
            retry_error = RuntimeError(error_text)

        hard_failure_retried = False
        async for retry_event in self._retry_after_hard_token_failure(
            error=retry_error,
            session_key=state.get("session_key"),
            user_message=state.get("user_message"),
            deps=state.get("deps"),
            selected_token_id=state.get("selected_token_id"),
            release_slot=state.get("release_slot"),
            thinking_emitter=state.get("thinking_emitter"),
            start_time=state.get("start_time"),
            max_tool_calls=state.get("max_tool_calls"),
            timeout_seconds=state.get("timeout_seconds"),
            token_failover_attempt=state.get("_token_failover_attempt"),
            emit_lifecycle_bounds=state.get("_emit_lifecycle_bounds"),
        ):
            hard_failure_retried = True
            yield retry_event

        if hard_failure_retried:
            state["release_slot"] = None
            state["selected_token_id"] = None
            state["should_stop"] = True
            return

        state["run_failed"] = True
        await self.runtime_events.trigger_llm_failed(
            session_key=state.get("session_key"),
            run_id=state.get("run_id"),
            error=error_text,
        )
        await self.runtime_events.trigger_run_failed(
            session_key=state.get("session_key"),
            run_id=state.get("run_id"),
            error=error_text,
        )
        await self.runtime_events.trigger_run_context_ready(
            session_key=state.get("session_key"),
            run_id=state.get("run_id"),
            user_message=state.get("user_message"),
            system_prompt=state.get("system_prompt"),
            message_history=state.get("context_history_for_hooks") or [],
            assistant_message=state.get("final_assistant") or "",
            tool_calls=state.get("tool_call_summaries") or [],
            run_status="failed",
            error=error_text,
            session_title=state.get("session_title"),
        )

        thinking_emitter = state.get("thinking_emitter")
        if thinking_emitter is not None:
            async for event in thinking_emitter.close_if_active():
                yield event

        yield StreamEvent.runtime_update(
            "failed",
            f"Agent runtime error: {error_text}",
            metadata={"phase": "exception", "elapsed": round(time.monotonic() - float(state.get('start_time') or 0.0), 1)},
        )
        yield StreamEvent.error_event(f"agent_error: {error_text}")

