from __future__ import annotations

import time
from typing import Any, AsyncIterator

from app.atlasclaw.agent.context_pruning import prune_context_messages, should_apply_context_pruning
from app.atlasclaw.agent.runner_tool.runner_execution_runtime import _ModelNodeTimeout
from app.atlasclaw.agent.stream import StreamEvent
from app.atlasclaw.agent.tool_gate_models import ToolPolicyMode


class RunnerExecutionFlowStreamMixin:
    async def _run_agent_node_stream(
        self,
        *,
        agent_run: Any,
        state: dict[str, Any],
        _log_step: Any,
        provider_fast_path_turn: bool,
    ) -> AsyncIterator[StreamEvent]:
        """Stream model/tool nodes and update run state in-place."""
        deps = state.get("deps")
        start_time = float(state.get("start_time") or 0.0)
        timeout_seconds = float(state.get("timeout_seconds") or 0.0)
        session = state.get("session")
        session_key = state.get("session_key")
        session_manager = state.get("session_manager")
        run_id = state.get("run_id")
        user_message = state.get("user_message")
        system_prompt = state.get("system_prompt")
        max_tool_calls = int(state.get("max_tool_calls") or 0)
        runtime_context_window = state.get("runtime_context_window")
        flushed_memory_signatures = state.get("flushed_memory_signatures")

        try:
            first_node_timeout = (
                self.PROVIDER_FAST_PATH_FIRST_NODE_TIMEOUT_SECONDS
                if provider_fast_path_turn
                else self.MODEL_FIRST_NODE_TIMEOUT_SECONDS
            )
            next_node_timeout = (
                self.PROVIDER_FAST_PATH_NEXT_NODE_TIMEOUT_SECONDS
                if provider_fast_path_turn
                else self.MODEL_NEXT_NODE_TIMEOUT_SECONDS
            )
            async for node in self._iter_agent_nodes_with_timeout(
                agent_run,
                first_node_timeout_seconds=first_node_timeout,
                next_node_timeout_seconds=next_node_timeout,
            ):
                if deps.is_aborted():
                    yield StreamEvent.lifecycle_aborted()
                    break

                if time.monotonic() - start_time > timeout_seconds:
                    yield StreamEvent.error_event("timeout")
                    break

                current_messages = self.history.normalize_messages(agent_run.all_messages())
                current_messages = self.history.prune_summary_messages(current_messages)
                if should_apply_context_pruning(
                    settings=self.context_pruning_settings,
                    session=session,
                ):
                    current_messages = prune_context_messages(
                        messages=current_messages,
                        settings=self.context_pruning_settings,
                        context_window_tokens=runtime_context_window,
                    )
                current_messages = self._deduplicate_message_history(current_messages)
                state["context_history_for_hooks"] = list(current_messages)

                if self.compaction.should_memory_flush(
                    current_messages,
                    session,
                    context_window_override=runtime_context_window,
                ):
                    await self.history.flush_history_to_timestamped_memory(
                        session_key=session_key,
                        messages=current_messages,
                        deps=deps,
                        session=session,
                        context_window=runtime_context_window,
                        flushed_signatures=flushed_memory_signatures,
                    )

                if self.compaction.should_compact(
                    current_messages,
                    session,
                    context_window_override=runtime_context_window,
                ):
                    if self.hooks:
                        await self.hooks.trigger(
                            "before_compaction",
                            {
                                "session_key": session_key,
                                "message_count": len(current_messages),
                            },
                        )
                    yield StreamEvent.compaction_start()
                    compressed = await self.compaction.compact(current_messages, session)
                    persist_override_messages = self.history.normalize_messages(compressed)
                    persist_override_messages = await self.history.inject_memory_recall(
                        persist_override_messages,
                        deps,
                    )
                    state["context_history_for_hooks"] = list(persist_override_messages)
                    state["persist_override_messages"] = persist_override_messages
                    state["persist_override_base_len"] = len(current_messages)
                    await session_manager.mark_compacted(session_key)
                    state["compaction_applied"] = True
                    yield StreamEvent.compaction_end()
                    if self.hooks:
                        await self.hooks.trigger(
                            "after_compaction",
                            {
                                "session_key": session_key,
                                "message_count": len(persist_override_messages),
                            },
                        )

                if self._is_model_request_node(node):
                    current_model_attempt = int(state.get("current_model_attempt") or 0) + 1
                    state["current_model_attempt"] = current_model_attempt
                    state["current_attempt_started_at"] = time.monotonic()
                    state["current_attempt_has_text"] = False
                    state["current_attempt_has_tool"] = False
                    thinking_emitter = state.get("thinking_emitter")
                    thinking_emitter.reset_cycle_flags()
                    await self.runtime_events.trigger_llm_input(
                        session_key=session_key,
                        run_id=run_id,
                        user_message=user_message,
                        system_prompt=system_prompt,
                        message_history=current_messages,
                    )
                    payload_profile = self._build_llm_payload_profile(
                        system_prompt=system_prompt,
                        user_message=user_message,
                        message_history=current_messages,
                    )
                    _log_step(
                        "llm_payload_profile",
                        stage=f"attempt_{current_model_attempt}",
                        attempt=current_model_attempt,
                        **payload_profile,
                    )
                    if isinstance(deps.extra, dict):
                        existing_profiles = deps.extra.get("_llm_payload_profiles")
                        entry = {
                            "stage": f"attempt_{current_model_attempt}",
                            "attempt": current_model_attempt,
                            **payload_profile,
                        }
                        if isinstance(existing_profiles, list):
                            existing_profiles.append(entry)
                        else:
                            deps.extra["_llm_payload_profiles"] = [entry]
                    yield StreamEvent.runtime_update(
                        "reasoning",
                        (
                            "Analyzing request."
                            if current_model_attempt == 1
                            else (
                                "Preparing final response from tool results."
                                if (state.get("tool_call_summaries") or provider_fast_path_turn)
                                else "Continuing reasoning."
                            )
                        ),
                        metadata={
                            "phase": "model_request",
                            "attempt": current_model_attempt,
                            "elapsed": round(time.monotonic() - start_time, 1),
                        },
                    )

                thinking_emitter = state.get("thinking_emitter")
                tool_gate_decision = state.get("tool_gate_decision")
                tool_call_summaries = state.get("tool_call_summaries") or []
                if hasattr(node, "model_response") and node.model_response:
                    async for event in thinking_emitter.emit_from_model_response(
                        model_response=node.model_response,
                        hooks=self.hooks,
                        session_key=session_key,
                    ):
                        if (
                            event.type == "assistant"
                            and (
                                tool_gate_decision.policy in {
                                    ToolPolicyMode.MUST_USE_TOOL,
                                    ToolPolicyMode.PREFER_TOOL,
                                }
                                and not tool_call_summaries
                            )
                        ):
                            state.get("buffered_assistant_events").append(event)
                        else:
                            if event.type == "assistant":
                                state["current_attempt_has_text"] = True
                                state["assistant_output_streamed"] = True
                            yield event
                elif hasattr(node, "content") and node.content:
                    content = str(node.content)
                    async for event in thinking_emitter.emit_plain_content(
                        content=content,
                        hooks=self.hooks,
                        session_key=session_key,
                    ):
                        if (
                            event.type == "assistant"
                            and (
                                tool_gate_decision.policy in {
                                    ToolPolicyMode.MUST_USE_TOOL,
                                    ToolPolicyMode.PREFER_TOOL,
                                }
                                and not tool_call_summaries
                            )
                        ):
                            state.get("buffered_assistant_events").append(event)
                        else:
                            if event.type == "assistant":
                                state["current_attempt_has_text"] = True
                                state["assistant_output_streamed"] = True
                            yield event

                tool_calls_in_node = self.runtime_events.collect_tool_calls(node)
                for tool_call in tool_calls_in_node:
                    if isinstance(tool_call, dict):
                        tool_name = tool_call.get("name", tool_call.get("tool_name", "unknown_tool"))
                        raw_args = tool_call.get("args", tool_call.get("arguments"))
                    else:
                        tool_name = getattr(tool_call, "tool_name", getattr(tool_call, "name", "unknown_tool"))
                        raw_args = getattr(tool_call, "args", getattr(tool_call, "arguments", None))
                    normalized_tool_name = str(tool_name)
                    parsed_args = self._extract_tool_call_arguments(raw_args)
                    summary: dict[str, Any] = {"name": normalized_tool_name}
                    if parsed_args:
                        summary["args"] = parsed_args
                    tool_call_summaries.append(summary)
                state["tool_call_summaries"] = tool_call_summaries

                if tool_calls_in_node:
                    state["current_attempt_has_tool"] = True
                    yield StreamEvent.runtime_update(
                        "waiting_for_tool",
                        "Preparing tool execution.",
                        metadata={
                            "phase": "planned",
                            "attempt": state.get("current_model_attempt"),
                            "elapsed": round(time.monotonic() - start_time, 1),
                            "tools": [
                                (
                                    tool_call.get("name", tool_call.get("tool_name", "unknown_tool"))
                                    if isinstance(tool_call, dict)
                                    else getattr(
                                        tool_call,
                                        "tool_name",
                                        getattr(tool_call, "name", "unknown_tool"),
                                    )
                                )
                                for tool_call in tool_calls_in_node
                            ],
                        },
                    )

                tool_dispatch = await self.runtime_events.dispatch_tool_calls(
                    tool_calls_in_node,
                    tool_calls_count=int(state.get("tool_calls_count") or 0),
                    max_tool_calls=max_tool_calls,
                    deps=deps,
                    session_key=session_key,
                    run_id=run_id,
                )
                state["tool_calls_count"] = tool_dispatch.tool_calls_count
                for event in tool_dispatch.events:
                    if event.type == "assistant":
                        state["assistant_output_streamed"] = True
                    yield event

                if provider_fast_path_turn and self._is_call_tools_node(node):
                    post_tool_messages = self.history.normalize_messages(agent_run.all_messages())
                    if not tool_call_summaries:
                        inferred = self._collect_tool_call_summaries_from_messages(
                            messages=post_tool_messages,
                            start_index=state.get("run_output_start_index"),
                        )
                        if inferred:
                            tool_call_summaries.extend(inferred)
                            state["tool_call_summaries"] = tool_call_summaries
                    post_tool_text = self._extract_tool_text_from_messages(
                        messages=post_tool_messages,
                        start_index=state.get("run_output_start_index"),
                        max_chars=9000,
                    ).strip()
                    if post_tool_text:
                        compact_tool_answer = self._compact_tool_fallback_text(post_tool_text)
                        if compact_tool_answer:
                            state["fast_path_tool_answer"] = compact_tool_answer
                            _log_step(
                                "provider_fast_path_short_circuit",
                                tool_calls=len(tool_call_summaries),
                                tool_text_chars=len(post_tool_text),
                                has_compact_answer=True,
                            )
                            break

                if (
                    self._is_call_tools_node(node)
                    and not state.get("current_attempt_has_text")
                    and not state.get("current_attempt_has_tool")
                    and thinking_emitter.current_cycle_had_thinking
                ):
                    elapsed_total = round(time.monotonic() - start_time, 1)
                    current_attempt_started_at = state.get("current_attempt_started_at")
                    attempt_elapsed = (
                        round(time.monotonic() - current_attempt_started_at, 1)
                        if current_attempt_started_at is not None
                        else elapsed_total
                    )
                    reasoning_retry_count = int(state.get("reasoning_retry_count") or 0)
                    reasoning_retry_limit = int(state.get("reasoning_retry_limit") or 0)
                    should_escalate = (
                        provider_fast_path_turn
                        or elapsed_total >= self.REASONING_ONLY_ESCALATION_SECONDS
                        or reasoning_retry_count >= reasoning_retry_limit
                    )
                    if should_escalate:
                        if state.get("web_tool_verification_enforced") and tool_gate_decision.policy in {
                            ToolPolicyMode.MUST_USE_TOOL,
                            ToolPolicyMode.PREFER_TOOL,
                        }:
                            yield StreamEvent.runtime_update(
                                "warning",
                                "Verification did not produce a usable tool-backed answer in this cycle.",
                                metadata={
                                    "phase": "verification",
                                    "attempt": state.get("current_model_attempt"),
                                    "elapsed": elapsed_total,
                                    "attempt_elapsed": attempt_elapsed,
                                },
                            )
                            break
                        raise RuntimeError(
                            "The model did not produce a usable answer after bounded reasoning retries."
                        )

                    reasoning_retry_count += 1
                    state["reasoning_retry_count"] = reasoning_retry_count
                    yield StreamEvent.runtime_update(
                        "retrying",
                        "Reasoning finished without a usable answer. Retrying with a stricter response policy.",
                        metadata={
                            "phase": "retry",
                            "attempt": reasoning_retry_count,
                            "elapsed": elapsed_total,
                            "attempt_elapsed": attempt_elapsed,
                            "reason": "reasoning_only",
                        },
                    )
                    if tool_dispatch.should_break:
                        break

        except _ModelNodeTimeout as timeout_exc:
            state["model_stream_timed_out"] = True
            state["model_timeout_error_message"] = "The model stream timed out before producing a usable response."
            yield StreamEvent.runtime_update(
                "warning",
                "Model stream timed out in this cycle. Attempting to recover from available tool output.",
                metadata={
                    "phase": "model_timeout",
                    "attempt": state.get("current_model_attempt"),
                    "elapsed": round(time.monotonic() - start_time, 1),
                    "timeout_seconds": timeout_exc.timeout_seconds,
                },
            )
            if not state.get("tool_call_summaries"):
                raise RuntimeError(state.get("model_timeout_error_message"))

