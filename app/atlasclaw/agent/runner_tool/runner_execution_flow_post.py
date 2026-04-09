from __future__ import annotations

import logging
import time
from typing import Any, AsyncIterator

from app.atlasclaw.agent.stream import StreamEvent
from app.atlasclaw.agent.tool_gate_models import ToolPolicyMode

logger = logging.getLogger(__name__)


class RunnerExecutionFlowPostMixin:
    async def _process_agent_run_outcome(
        self,
        *,
        agent_run: Any,
        state: dict[str, Any],
        _log_step: Any,
        provider_fast_path_turn: bool,
    ) -> AsyncIterator[StreamEvent]:
        """Resolve final assistant output, persist transcript, and emit terminal events."""
        start_time = float(state.get("start_time") or 0.0)
        session_key = state.get("session_key")
        session_manager = state.get("session_manager")
        session = state.get("session")
        run_id = state.get("run_id")
        user_message = state.get("user_message")
        system_prompt = state.get("system_prompt")
        deps = state.get("deps")
        runtime_agent = state.get("runtime_agent")
        tool_gate_decision = state.get("tool_gate_decision")
        tool_match_result = state.get("tool_match_result")
        available_tools = state.get("available_tools")
        web_tool_verification_enforced = bool(state.get("web_tool_verification_enforced"))
        max_tool_calls = int(state.get("max_tool_calls") or 0)
        timeout_seconds = float(state.get("timeout_seconds") or 0.0)
        token_failover_attempt = int(state.get("_token_failover_attempt") or 0)
        emit_lifecycle_bounds = bool(state.get("_emit_lifecycle_bounds"))
        selected_token_id = state.get("selected_token_id")
        release_slot = state.get("release_slot")
        tool_policy_retry_count = int(state.get("tool_policy_retry_count") or 0)

        try:
            raw_final_messages = agent_run.all_messages()
        except Exception:
            raw_final_messages = list(state.get("message_history") or []) + [
                {"role": "user", "content": user_message}
            ]
        final_messages = self.history.normalize_messages(raw_final_messages)

        persist_override_messages = state.get("persist_override_messages")
        persist_override_base_len = int(state.get("persist_override_base_len") or 0)
        if persist_override_messages is not None:
            if len(final_messages) > persist_override_base_len > 0:
                final_messages = persist_override_messages + final_messages[persist_override_base_len:]
            else:
                final_messages = persist_override_messages
            state["run_output_start_index"] = len(persist_override_messages)

        run_output_start_index = int(state.get("run_output_start_index") or 0)
        final_assistant = self._extract_latest_assistant_from_messages(
            messages=final_messages,
            start_index=run_output_start_index,
        )

        fast_path_tool_answer = str(state.get("fast_path_tool_answer") or "")
        if fast_path_tool_answer and not final_assistant.strip():
            final_assistant = fast_path_tool_answer
            final_messages = self._replace_last_assistant_message(
                messages=final_messages,
                content=final_assistant,
            )
            _log_step(
                "provider_fast_path_short_circuit_applied",
                answer_chars=len(final_assistant),
            )

        buffered_assistant_events = state.get("buffered_assistant_events") or []
        if buffered_assistant_events and final_assistant:
            buffered_reasoning_text = self._collect_buffered_assistant_text(buffered_assistant_events)
            if buffered_reasoning_text:
                yield StreamEvent.thinking_delta(buffered_reasoning_text)
                yield StreamEvent.thinking_end(elapsed=0.0)
            buffered_assistant_events.clear()

        tool_call_summaries = state.get("tool_call_summaries") or []
        if buffered_assistant_events and not final_assistant:
            if tool_call_summaries:
                buffered_reasoning_text = self._collect_buffered_assistant_text(buffered_assistant_events)
                if buffered_reasoning_text:
                    yield StreamEvent.thinking_delta(buffered_reasoning_text)
                    yield StreamEvent.thinking_end(elapsed=0.0)
                buffered_assistant_events.clear()
            else:
                while buffered_assistant_events:
                    event = buffered_assistant_events.pop(0)
                    if event.type == "assistant":
                        final_assistant += event.content
                        state["assistant_output_streamed"] = True
                    yield event
                state.get("thinking_emitter").assistant_emitted = bool(final_assistant)

        if final_messages:
            inferred_tool_calls = self._collect_tool_call_summaries_from_messages(
                messages=final_messages,
                start_index=run_output_start_index,
            )
            if inferred_tool_calls:
                existing_signatures: set[tuple[str, str]] = set()
                for item in tool_call_summaries:
                    if not isinstance(item, dict):
                        continue
                    name = str(item.get("name", "") or "").strip()
                    if not name:
                        continue
                    args = item.get("args")
                    args_signature = str(sorted(args.items())) if isinstance(args, dict) else ""
                    existing_signatures.add((name, args_signature))
                for item in inferred_tool_calls:
                    name = str(item.get("name", "") or "").strip()
                    if not name:
                        continue
                    args = item.get("args")
                    args_signature = str(sorted(args.items())) if isinstance(args, dict) else ""
                    signature = (name, args_signature)
                    if signature in existing_signatures:
                        continue
                    existing_signatures.add(signature)
                    tool_call_summaries.append(item)

        model_stream_timed_out = bool(state.get("model_stream_timed_out"))
        if model_stream_timed_out and not final_assistant.strip() and tool_call_summaries:
            recovered_assistant = await self._build_post_tool_wrapped_message(
                runtime_agent=runtime_agent,
                deps=deps,
                user_message=user_message,
                tool_calls=tool_call_summaries,
            )
            if recovered_assistant:
                final_assistant = recovered_assistant
                final_messages = self._replace_last_assistant_message(
                    messages=final_messages,
                    content=final_assistant,
                )

        if provider_fast_path_turn and not final_assistant.strip() and tool_call_summaries:
            provider_tool_text = self._extract_tool_text_from_messages(
                messages=final_messages,
                start_index=run_output_start_index,
                max_chars=9000,
            ).strip()
            if provider_tool_text:
                compact_provider_tool_text = self._compact_tool_fallback_text(provider_tool_text)
                if compact_provider_tool_text:
                    final_assistant = compact_provider_tool_text
                    final_messages = self._replace_last_assistant_message(
                        messages=final_messages,
                        content=final_assistant,
                    )
                    _log_step(
                        "provider_fast_path_tool_text_fallback",
                        tool_text_chars=len(provider_tool_text),
                        answer_chars=len(final_assistant),
                    )

        if model_stream_timed_out and not final_assistant.strip():
            recovered_from_messages = await self._build_timeout_fallback_message_from_messages(
                runtime_agent=runtime_agent,
                deps=deps,
                user_message=user_message,
                messages=final_messages,
                start_index=run_output_start_index,
            )
            if recovered_from_messages:
                final_assistant = recovered_from_messages
                final_messages = self._replace_last_assistant_message(
                    messages=final_messages,
                    content=final_assistant,
                )

        assistant_output_streamed = bool(state.get("assistant_output_streamed"))
        if not assistant_output_streamed:
            if not final_assistant and hasattr(agent_run, "result") and agent_run.result:
                result = agent_run.result
                if hasattr(result, "response") and result.response:
                    response = result.response
                    if hasattr(response, "parts"):
                        for part in response.parts:
                            part_kind = getattr(part, "part_kind", "")
                            if part_kind != "thinking" and hasattr(part, "content") and part.content:
                                content = str(part.content)
                                if content:
                                    final_assistant = content
                                    break
                    elif hasattr(response, "content") and response.content:
                        final_assistant = str(response.content)
                if not final_assistant and hasattr(result, "data") and result.data:
                    final_assistant = str(result.data)

            if not final_assistant:
                final_assistant = self._extract_latest_assistant_from_messages(
                    messages=final_messages,
                    start_index=run_output_start_index,
                )

        missing_required_tool_names = (
            self._missing_required_tool_names(
                decision=tool_gate_decision,
                match_result=tool_match_result,
                tool_call_summaries=tool_call_summaries,
                available_tools=available_tools,
                final_messages=final_messages,
                run_output_start_index=run_output_start_index,
            )
            if web_tool_verification_enforced
            else []
        )
        strict_tool_need = self._tool_gate_has_strict_need(tool_gate_decision)
        should_fail_for_missing_evidence = False
        should_block_assistant_emit = should_fail_for_missing_evidence

        if not assistant_output_streamed and final_assistant and not should_block_assistant_emit:
            state.get("thinking_emitter").assistant_emitted = True
            assistant_output_streamed = True
            yield StreamEvent.assistant_delta(final_assistant)

        current_model_attempt = int(state.get("current_model_attempt") or 0)
        if (
            web_tool_verification_enforced
            and tool_gate_decision.policy is ToolPolicyMode.PREFER_TOOL
            and not strict_tool_need
            and missing_required_tool_names
        ):
            warning_message = self._build_tool_evidence_required_message(
                match_result=tool_match_result,
                missing_required_tools=missing_required_tool_names,
            )
            yield StreamEvent.runtime_update(
                "warning",
                warning_message,
                metadata={
                    "phase": "final",
                    "attempt": current_model_attempt,
                    "elapsed": round(time.monotonic() - start_time, 1),
                },
            )

        if should_fail_for_missing_evidence:
            failure_message = self._build_tool_evidence_required_message(
                match_result=tool_match_result,
                missing_required_tools=missing_required_tool_names,
            )
            tool_policy_retried = False
            async for retry_event in self._retry_after_tool_policy_failure(
                session_key=session_key,
                user_message=user_message,
                deps=deps,
                release_slot=release_slot,
                selected_token_id=selected_token_id,
                start_time=start_time,
                max_tool_calls=max_tool_calls,
                timeout_seconds=timeout_seconds,
                token_failover_attempt=token_failover_attempt,
                emit_lifecycle_bounds=emit_lifecycle_bounds,
                failure_message=failure_message,
                missing_required_tools=missing_required_tool_names,
                tool_policy_retry_count=tool_policy_retry_count,
                allow_retry=not bool(tool_gate_decision.needs_external_system),
            ):
                tool_policy_retried = True
                yield retry_event
            if tool_policy_retried:
                state["release_slot"] = None
                state["selected_token_id"] = None
                state["should_stop"] = True
                return

            safe_messages = self._remove_last_assistant_from_run(
                messages=final_messages,
                start_index=run_output_start_index,
            )
            await session_manager.persist_transcript(session_key, safe_messages)
            await self.runtime_events.trigger_run_context_ready(
                session_key=session_key,
                run_id=run_id,
                user_message=user_message,
                system_prompt=system_prompt,
                message_history=state.get("context_history_for_hooks") or [],
                assistant_message="",
                tool_calls=tool_call_summaries,
                run_status="failed",
                error=failure_message,
                session_title=state.get("session_title"),
            )
            state["run_failed"] = True
            await self.runtime_events.trigger_llm_failed(
                session_key=session_key,
                run_id=run_id,
                error=failure_message,
            )
            await self.runtime_events.trigger_run_failed(
                session_key=session_key,
                run_id=run_id,
                error=failure_message,
            )
            yield StreamEvent.runtime_update(
                "failed",
                failure_message,
                metadata={
                    "phase": "final",
                    "attempt": current_model_attempt,
                    "elapsed": round(time.monotonic() - start_time, 1),
                },
            )
            yield StreamEvent.error_event(failure_message)
            buffered_assistant_events.clear()
            final_assistant = ""

        else:
            if not final_assistant.strip():
                state["run_failed"] = True
                model_timeout_error_message = str(state.get("model_timeout_error_message") or "")
                failure_message = (
                    model_timeout_error_message
                    if model_stream_timed_out and model_timeout_error_message
                    else "The run ended without a usable final answer."
                )
                await self.runtime_events.trigger_llm_failed(
                    session_key=session_key,
                    run_id=run_id,
                    error=failure_message,
                )
                await self.runtime_events.trigger_run_failed(
                    session_key=session_key,
                    run_id=run_id,
                    error=failure_message,
                )
                safe_messages = self._remove_last_assistant_from_run(
                    messages=final_messages,
                    start_index=run_output_start_index,
                )
                await session_manager.persist_transcript(session_key, safe_messages)
                await self.runtime_events.trigger_run_context_ready(
                    session_key=session_key,
                    run_id=run_id,
                    user_message=user_message,
                    system_prompt=system_prompt,
                    message_history=state.get("context_history_for_hooks") or [],
                    assistant_message="",
                    tool_calls=tool_call_summaries,
                    run_status="failed",
                    error=failure_message,
                    session_title=state.get("session_title"),
                )
                yield StreamEvent.runtime_update(
                    "failed",
                    failure_message,
                    metadata={
                        "phase": "final",
                        "attempt": current_model_attempt,
                        "elapsed": round(time.monotonic() - start_time, 1),
                    },
                )
                yield StreamEvent.error_event(failure_message)
                buffered_assistant_events.clear()
                final_assistant = ""
            else:
                answered_elapsed = round(time.monotonic() - start_time, 1)
                yield StreamEvent.runtime_update(
                    "answered",
                    "Final answer ready.",
                    metadata={
                        "phase": "final",
                        "attempt": current_model_attempt,
                        "elapsed": answered_elapsed,
                    },
                )
                state["answer_committed"] = True

                _log_step("post_success_llm_completed_start")
                try:
                    await self.runtime_events.trigger_llm_completed(
                        session_key=session_key,
                        run_id=run_id,
                        assistant_message=final_assistant,
                    )
                    _log_step("post_success_llm_completed_done")
                except Exception as exc:
                    logger.exception("post_success_llm_completed failed")
                    _log_step("post_success_llm_completed_error", error=str(exc))

                _log_step("post_success_persist_transcript_start")
                try:
                    await session_manager.persist_transcript(session_key, final_messages)
                    _log_step("post_success_persist_transcript_done")
                except Exception as exc:
                    logger.exception("post_success_persist_transcript failed")
                    _log_step("post_success_persist_transcript_error", error=str(exc))

                _log_step("post_success_finalize_title_start")
                try:
                    await self._maybe_finalize_title(
                        session_manager=session_manager,
                        session_key=session_key,
                        session=session,
                        final_messages=final_messages,
                        user_message=user_message,
                    )
                    _log_step("post_success_finalize_title_done")
                    state["session_title"] = str(getattr(session, "title", "") or "")
                except Exception as exc:
                    logger.exception("post_success_finalize_title failed")
                    _log_step("post_success_finalize_title_error", error=str(exc))

                _log_step("post_success_run_context_ready_start")
                try:
                    await self.runtime_events.trigger_run_context_ready(
                        session_key=session_key,
                        run_id=run_id,
                        user_message=user_message,
                        system_prompt=system_prompt,
                        message_history=state.get("context_history_for_hooks") or [],
                        assistant_message=final_assistant,
                        tool_calls=tool_call_summaries,
                        run_status="completed",
                        session_title=state.get("session_title"),
                    )
                    _log_step("post_success_run_context_ready_done")
                except Exception as exc:
                    logger.exception("post_success_run_context_ready failed")
                    _log_step("post_success_run_context_ready_error", error=str(exc))

        state["assistant_output_streamed"] = assistant_output_streamed
        state["final_assistant"] = final_assistant
        state["tool_call_summaries"] = tool_call_summaries
        state["buffered_assistant_events"] = buffered_assistant_events
        state["message_history"] = final_messages

