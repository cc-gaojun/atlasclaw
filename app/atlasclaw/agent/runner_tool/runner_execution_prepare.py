from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncIterator, Optional

from app.atlasclaw.agent.context_pruning import prune_context_messages, should_apply_context_pruning
from app.atlasclaw.agent.context_window_guard import evaluate_context_window_guard
from app.atlasclaw.agent.runner_prompt_context import build_system_prompt, collect_tool_groups_snapshot, collect_tools_snapshot
from app.atlasclaw.agent.stream import StreamEvent
from app.atlasclaw.agent.thinking_stream import ThinkingStreamEmitter
from app.atlasclaw.agent.tool_gate import CapabilityMatcher
from app.atlasclaw.agent.tool_gate_models import CapabilityMatchResult, ToolGateDecision, ToolPolicyMode
from app.atlasclaw.core.deps import SkillDeps
from app.atlasclaw.agent.runner_tool.runner_execution_runtime import _ModelNodeTimeout


logger = logging.getLogger(__name__)


class RunnerExecutionPreparePhaseMixin:
    async def _run_prepare_phase(self, *, state: dict[str, Any], _log_step: Any) -> AsyncIterator[StreamEvent]:
        """Prepare runtime/session/prompt/tool-gate phase before model loop."""
        session_key = state.get("session_key")
        user_message = state.get("user_message")
        deps = state.get("deps")
        max_tool_calls = state.get("max_tool_calls")
        timeout_seconds = state.get("timeout_seconds")
        _token_failover_attempt = state.get("_token_failover_attempt")
        _emit_lifecycle_bounds = state.get("_emit_lifecycle_bounds")
        start_time = state.get("start_time")
        tool_calls_count = state.get("tool_calls_count")
        compaction_applied = state.get("compaction_applied")
        thinking_emitter = state.get("thinking_emitter")
        persist_override_messages = state.get("persist_override_messages")
        persist_override_base_len = state.get("persist_override_base_len")
        runtime_agent = state.get("runtime_agent")
        selected_token_id = state.get("selected_token_id")
        release_slot = state.get("release_slot")
        flushed_memory_signatures = state.get("flushed_memory_signatures")
        extra = state.get("extra")
        run_id = state.get("run_id")
        tool_policy_retry_count = state.get("tool_policy_retry_count")
        run_failed = state.get("run_failed")
        message_history = state.get("message_history")
        system_prompt = state.get("system_prompt")
        final_assistant = state.get("final_assistant")
        context_history_for_hooks = state.get("context_history_for_hooks")
        tool_call_summaries = state.get("tool_call_summaries")
        session_title = state.get("session_title")
        buffered_assistant_events = state.get("buffered_assistant_events")
        assistant_output_streamed = state.get("assistant_output_streamed")
        tool_request_message = state.get("tool_request_message")
        tool_gate_decision = state.get("tool_gate_decision")
        tool_match_result = state.get("tool_match_result")
        current_model_attempt = state.get("current_model_attempt")
        current_attempt_started_at = state.get("current_attempt_started_at")
        current_attempt_has_text = state.get("current_attempt_has_text")
        current_attempt_has_tool = state.get("current_attempt_has_tool")
        reasoning_retry_count = state.get("reasoning_retry_count")
        run_output_start_index = state.get("run_output_start_index")
        web_tool_verification_enforced = state.get("web_tool_verification_enforced")
        reasoning_retry_limit = state.get("reasoning_retry_limit")
        model_stream_timed_out = state.get("model_stream_timed_out")
        model_timeout_error_message = state.get("model_timeout_error_message")
        fast_path_tool_answer = state.get("fast_path_tool_answer")
        runtime_context_window_info = state.get("runtime_context_window_info")
        runtime_context_guard = state.get("runtime_context_guard")
        runtime_context_window = state.get("runtime_context_window")
        session_manager = state.get("session_manager")
        session = state.get("session")
        transcript = state.get("transcript")
        all_available_tools = state.get("all_available_tools")
        tool_groups_snapshot = state.get("tool_groups_snapshot")
        available_tools = state.get("available_tools")
        toolset_filter_trace = state.get("toolset_filter_trace")
        used_toolset_fallback = state.get("used_toolset_fallback")
        provider_hint_docs = state.get("provider_hint_docs")
        skill_hint_docs = state.get("skill_hint_docs")
        metadata_candidates = state.get("metadata_candidates")
        ranking_trace = state.get("ranking_trace")
        try:
            if _emit_lifecycle_bounds:
                yield StreamEvent.lifecycle_start()
            _log_step("lifecycle_start")
            yield StreamEvent.runtime_update(
                "reasoning",
                "Starting response analysis.",
                metadata={"phase": "start", "attempt": 0, "elapsed": 0.0},
            )

            runtime_agent, selected_token_id, release_slot = await self._resolve_runtime_agent(session_key, deps)
            logger.warning(
                "runtime token resolved: session=%s selected_token_id=%s managed_tokens=%s",
                session_key,
                selected_token_id,
                len(self.token_policy.token_pool.tokens) if self.token_policy is not None else 0,
            )
            runtime_context_window_info = self._resolve_runtime_context_window_info(selected_token_id, deps)
            runtime_context_guard = evaluate_context_window_guard(
                tokens=runtime_context_window_info.tokens,
                source=runtime_context_window_info.source,
            )
            runtime_context_window = runtime_context_guard.tokens
            _log_step(
                "context_guard_evaluated",
                tokens=runtime_context_guard.tokens,
                source=runtime_context_guard.source,
                should_warn=runtime_context_guard.should_warn,
                should_block=runtime_context_guard.should_block,
            )
            if runtime_context_guard.should_warn:
                yield StreamEvent.runtime_update(
                    "warning",
                    (
                        "Model context window is below the warning threshold. "
                        f"tokens={runtime_context_guard.tokens}, source={runtime_context_guard.source}"
                    ),
                    metadata={
                        "phase": "context_guard",
                        "tokens": runtime_context_guard.tokens,
                        "source": runtime_context_guard.source,
                        "guard": "warn",
                        "elapsed": round(time.monotonic() - start_time, 1),
                    },
                )
            if runtime_context_guard.should_block:
                failure_message = (
                    "Model context window is below the minimum safety threshold. "
                    f"tokens={runtime_context_guard.tokens}, source={runtime_context_guard.source}"
                )
                run_failed = True
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
                        "phase": "context_guard",
                        "tokens": runtime_context_guard.tokens,
                        "source": runtime_context_guard.source,
                        "guard": "block",
                        "elapsed": round(time.monotonic() - start_time, 1),
                    },
                )
                yield StreamEvent.error_event(failure_message)
                state["should_stop"] = True
                return
            session_manager = self._resolve_session_manager(session_key, deps)

            # --:session + build prompt --

            session = await session_manager.get_or_create(session_key)
            _log_step("session_get_or_create_done")
            transcript = await session_manager.load_transcript(session_key)
            _log_step("session_load_transcript_done", transcript_entries=len(transcript))
            message_history = self.history.build_message_history(transcript)
            message_history = self.history.prune_summary_messages(message_history)
            if should_apply_context_pruning(settings=self.context_pruning_settings, session=session):
                message_history = prune_context_messages(
                    messages=message_history,
                    settings=self.context_pruning_settings,
                    context_window_tokens=runtime_context_window,
                )
            message_history = self._deduplicate_message_history(message_history)
            context_history_for_hooks = list(message_history)
            session_title = str(getattr(session, "title", "") or "")
            await self.runtime_events.trigger_message_received(
                session_key=session_key,
                run_id=run_id,
                user_message=user_message,
            )
            _log_step("hook_message_received_dispatched")
            await self.runtime_events.trigger_run_started(
                session_key=session_key,
                run_id=run_id,
                user_message=user_message,
            )
            _log_step("hook_run_started_dispatched")
            await self._maybe_set_draft_title(
                session_manager=session_manager,
                session_key=session_key,
                session=session,
                transcript=transcript,
                user_message=user_message,
            )
            _log_step("session_draft_title_done")
            all_available_tools = collect_tools_snapshot(agent=runtime_agent, deps=deps)
            _log_step("tools_snapshot_collected", all_tools_count=len(all_available_tools))
            tool_groups_snapshot = collect_tool_groups_snapshot(deps)
            _log_step("tool_groups_snapshot_collected", group_count=len(tool_groups_snapshot))
            available_tools, toolset_filter_trace, used_toolset_fallback = self._build_turn_toolset(
                deps=deps,
                session_key=session_key,
                all_tools=all_available_tools,
                tool_groups=tool_groups_snapshot,
            )
            _log_step(
                "toolset_policy_applied",
                total_tools=len(all_available_tools),
                filtered_tools=len(available_tools),
                used_fallback=used_toolset_fallback,
                policy_layers=len(toolset_filter_trace),
            )
            if used_toolset_fallback:
                yield StreamEvent.runtime_update(
                    "warning",
                    "Tool policy filtering produced an empty set; reverted to a safe fallback toolset.",
                    metadata={
                        "phase": "toolset_policy",
                        "elapsed": round(time.monotonic() - start_time, 1),
                    },
                )
            if isinstance(deps.extra, dict):
                deps.extra["tools_snapshot"] = list(available_tools)
                deps.extra["tools_snapshot_authoritative"] = True
                deps.extra["toolset_policy_trace"] = list(toolset_filter_trace)
                deps.extra["tool_groups_snapshot"] = self._build_filtered_group_map(
                    tool_groups_snapshot,
                    available_tools,
                )
            provider_hint_docs = self._build_provider_hint_docs(
                deps=deps,
                available_tools=available_tools,
            )
            skill_hint_docs = self._build_skill_hint_docs(
                deps=deps,
                available_tools=available_tools,
            )
            if isinstance(deps.extra, dict):
                deps.extra["provider_hint_docs"] = provider_hint_docs
                deps.extra["skill_hint_docs"] = skill_hint_docs
            _log_step(
                "hint_docs_built",
                provider_hint_count=len(provider_hint_docs),
                skill_hint_count=len(skill_hint_docs),
            )
            metadata_candidates = self._recall_provider_skill_candidates_from_metadata(
                user_message=user_message,
                recent_history=message_history,
                available_tools=available_tools,
                provider_hint_docs=provider_hint_docs,
                skill_hint_docs=skill_hint_docs,
                top_k_provider=int(getattr(self, "TOOL_METADATA_PROVIDER_TOP_K", 3) or 3),
                top_k_skill=int(getattr(self, "TOOL_METADATA_SKILL_TOP_K", 6) or 6),
            )
            if isinstance(deps.extra, dict):
                deps.extra["tool_metadata_candidates"] = dict(metadata_candidates)
            logger.warning(
                "tool_gate metadata_recall: session=%s provider_top=%s skill_top=%s preferred_providers=%s preferred_capabilities=%s preferred_tools=%s confidence=%.3f",
                session_key,
                [
                    str(item.get("provider_type", "") or "").strip()
                    for item in metadata_candidates.get("provider_candidates", [])[:3]
                    if isinstance(item, dict)
                ],
                [
                    str(item.get("hint_id", "") or "").strip()
                    for item in metadata_candidates.get("skill_candidates", [])[:3]
                    if isinstance(item, dict)
                ],
                list(metadata_candidates.get("preferred_provider_types", [])),
                list(metadata_candidates.get("preferred_capability_classes", [])),
                list(metadata_candidates.get("preferred_tool_names", [])),
                float(metadata_candidates.get("confidence", 0.0) or 0.0),
            )

            ranking_trace: dict[str, Any] = {}
            if self._should_attempt_hint_ranking(
                available_tools=available_tools,
                provider_hint_docs=provider_hint_docs,
                skill_hint_docs=skill_hint_docs,
                metadata_candidates=metadata_candidates,
                min_confidence=float(
                    getattr(self, "TOOL_HINT_RANKER_MIN_METADATA_CONFIDENCE", 0.3) or 0.3
                ),
            ):
                _log_step(
                    "hint_ranking_started",
                    candidate_count=len(available_tools),
                    provider_hint_count=len(provider_hint_docs),
                    skill_hint_count=len(skill_hint_docs),
                )
                ranking_started_at = time.monotonic()
                await self.runtime_events.trigger_hint_ranking_started(
                    session_key=session_key,
                    run_id=run_id,
                    candidate_count=len(available_tools),
                    provider_hint_count=len(provider_hint_docs),
                    skill_hint_count=len(skill_hint_docs),
                )
                ranking_result, ranking_fallback_reason = await self._rank_tools_with_hint_docs(
                    agent=runtime_agent,
                    deps=deps,
                    user_message=user_message,
                    recent_history=message_history,
                    available_tools=available_tools,
                    provider_hint_docs=provider_hint_docs,
                    skill_hint_docs=skill_hint_docs,
                )
                ranking_elapsed_ms = int((time.monotonic() - ranking_started_at) * 1000)
                if ranking_result is not None:
                    available_tools, ranking_trace = self._reorder_tools_by_hint_ranking(
                        available_tools=available_tools,
                        ranking=ranking_result,
                    )
                    if isinstance(deps.extra, dict):
                        deps.extra["tools_snapshot"] = list(available_tools)
                        deps.extra["tools_snapshot_authoritative"] = True
                        deps.extra["tool_groups_snapshot"] = self._build_filtered_group_map(
                            tool_groups_snapshot,
                            available_tools,
                        )
                        deps.extra["tool_ranking_trace"] = dict(ranking_trace)
                    await self.runtime_events.trigger_hint_ranking_completed(
                        session_key=session_key,
                        run_id=run_id,
                        preferred_provider_types=list(
                            ranking_result.get("preferred_provider_types", [])
                        ),
                        preferred_capability_classes=list(
                            ranking_result.get("preferred_capability_classes", [])
                        ),
                        preferred_tool_names=list(
                            ranking_result.get("preferred_tool_names", [])
                        ),
                        confidence=float(ranking_result.get("confidence", 0.0) or 0.0),
                        reason=str(ranking_result.get("reason", "") or ""),
                        elapsed_ms=ranking_elapsed_ms,
                    )
                    _log_step(
                        "hint_ranking_completed",
                        elapsed_ms=ranking_elapsed_ms,
                        preferred_provider_types=list(
                            ranking_result.get("preferred_provider_types", [])
                        ),
                        preferred_capability_classes=list(
                            ranking_result.get("preferred_capability_classes", [])
                        ),
                        preferred_tool_names=list(ranking_result.get("preferred_tool_names", [])),
                        confidence=float(ranking_result.get("confidence", 0.0) or 0.0),
                        reason=str(ranking_result.get("reason", "") or ""),
                    )
                else:
                    ranking_trace = {
                        "status": "fallback",
                        "reason": ranking_fallback_reason,
                        "top_tool_hints": [],
                    }
                    if isinstance(deps.extra, dict):
                        deps.extra["tool_ranking_trace"] = dict(ranking_trace)
                    await self.runtime_events.trigger_hint_ranking_fallback(
                        session_key=session_key,
                        run_id=run_id,
                        reason=ranking_fallback_reason,
                        elapsed_ms=ranking_elapsed_ms,
                    )
                    _log_step(
                        "hint_ranking_fallback",
                        elapsed_ms=ranking_elapsed_ms,
                        reason=ranking_fallback_reason,
                    )
            else:
                _log_step(
                    "hint_ranking_skipped",
                    candidate_count=len(available_tools),
                    provider_hint_count=len(provider_hint_docs),
                    skill_hint_count=len(skill_hint_docs),
                    metadata_confidence=float(metadata_candidates.get("confidence", 0.0) or 0.0),
                    metadata_reason=str(metadata_candidates.get("reason", "") or ""),
                )
            tool_request_message, used_follow_up_context = self._resolve_contextual_tool_request(
                user_message=user_message,
                recent_history=message_history,
            )
            _log_step(
                "tool_request_resolved",
                used_follow_up_context=used_follow_up_context,
                raw_user_message=user_message,
                resolved_tool_request=tool_request_message,
            )
            classifier_history = self._build_classifier_history(
                recent_history=message_history,
                used_follow_up_context=used_follow_up_context,
            )
            tool_gate_classifier = self._resolve_tool_gate_classifier(
                agent=runtime_agent,
                deps=deps,
                available_tools=available_tools,
            )
            _log_step(
                "tool_gate_classifier_resolved",
                classifier_enabled=bool(tool_gate_classifier is not None),
            )
            tool_gate_cache_key = self._build_tool_gate_cache_key(
                session_key=session_key,
                resolved_tool_request=tool_request_message,
                used_follow_up_context=used_follow_up_context,
                recent_history=classifier_history,
                available_tools=available_tools,
                metadata_candidates=metadata_candidates,
            )
            cached_tool_gate_decision = self._get_cached_tool_gate_decision(tool_gate_cache_key)
            if cached_tool_gate_decision is not None:
                tool_gate_decision = cached_tool_gate_decision
                _log_step(
                    "tool_gate_cache_hit",
                    cache_key=tool_gate_cache_key[:12],
                )
            else:
                _log_step(
                    "tool_gate_cache_miss",
                    cache_key=tool_gate_cache_key[:12],
                )
                short_circuit_decision = self._build_metadata_short_circuit_decision(
                    user_message=tool_request_message,
                    recent_history=message_history,
                    available_tools=available_tools,
                    metadata_candidates=metadata_candidates,
                    deps=deps,
                )
                if short_circuit_decision is not None:
                    tool_gate_decision = short_circuit_decision
                    _log_step(
                        "tool_gate_short_circuit",
                        source="metadata_active_provider",
                        policy=tool_gate_decision.policy.value,
                        suggested_classes=list(tool_gate_decision.suggested_tool_classes),
                    )
                else:
                    tool_gate_decision = await self.tool_gate.classify_async(
                        tool_request_message,
                        classifier_history,
                        classifier=tool_gate_classifier,
                    )
                tool_gate_decision = self._normalize_tool_gate_decision(tool_gate_decision)
                tool_gate_decision = self._apply_no_classifier_follow_up_fallback(
                    decision=tool_gate_decision,
                    used_follow_up_context=used_follow_up_context,
                    available_tools=available_tools,
                )
                tool_gate_decision = self._apply_provider_skill_intent_fallback(
                    decision=tool_gate_decision,
                    user_message=tool_request_message,
                    recent_history=message_history,
                    available_tools=available_tools,
                    deps=deps,
                )
                tool_gate_decision = self._apply_tool_gate_consistency_guard(
                    decision=tool_gate_decision,
                    user_message=tool_request_message,
                    recent_history=classifier_history,
                    available_tools=available_tools,
                    deps=deps,
                    metadata_candidates=metadata_candidates,
                )
                self._store_tool_gate_decision_cache(
                    cache_key=tool_gate_cache_key,
                    decision=tool_gate_decision,
                )
            tool_match_result = CapabilityMatcher(available_tools=available_tools).match(
                tool_gate_decision.suggested_tool_classes
            )
            tool_gate_decision, tool_match_result = self._align_external_system_intent(
                decision=tool_gate_decision,
                match_result=tool_match_result,
                available_tools=available_tools,
                user_message=tool_request_message,
                recent_history=message_history,
                deps=deps,
            )
            logger.warning(
                "tool_gate decision: session=%s policy=%s needs_external=%s needs_live_data=%s suggested=%s candidates=%s",
                session_key,
                tool_gate_decision.policy.value,
                bool(tool_gate_decision.needs_external_system),
                bool(tool_gate_decision.needs_live_data),
                list(tool_gate_decision.suggested_tool_classes),
                [
                    str(getattr(candidate, "name", "") or "").strip()
                    for candidate in tool_match_result.tool_candidates
                    if str(getattr(candidate, "name", "") or "").strip()
                ],
            )
            _log_step(
                "tool_gate_decided",
                policy=tool_gate_decision.policy.value,
                needs_tool=bool(tool_gate_decision.needs_tool),
                needs_external=bool(tool_gate_decision.needs_external_system),
                needs_live_data=bool(tool_gate_decision.needs_live_data),
                suggested_classes=list(tool_gate_decision.suggested_tool_classes),
                candidate_count=len(tool_match_result.tool_candidates),
                missing_capabilities=list(tool_match_result.missing_capabilities),
            )
            available_tools, provider_prefilter_trace = self._apply_provider_hard_prefilter(
                decision=tool_gate_decision,
                match_result=tool_match_result,
                available_tools=available_tools,
                deps=deps,
            )
            if isinstance(deps.extra, dict):
                deps.extra["tool_provider_prefilter_trace"] = dict(provider_prefilter_trace)
                deps.extra["tools_snapshot"] = list(available_tools)
                deps.extra["tools_snapshot_authoritative"] = True
                deps.extra["tool_groups_snapshot"] = self._build_filtered_group_map(
                    tool_groups_snapshot,
                    available_tools,
                )
            if provider_prefilter_trace:
                logger.warning(
                    "tool_gate provider_prefilter: session=%s enabled=%s before=%s after=%s providers=%s capabilities=%s matched_provider_tools=%s retained_builtin=%s",
                    session_key,
                    bool(provider_prefilter_trace.get("enabled")),
                    int(provider_prefilter_trace.get("before_count", 0) or 0),
                    int(provider_prefilter_trace.get("after_count", 0) or 0),
                    list(provider_prefilter_trace.get("target_provider_types", [])),
                    list(provider_prefilter_trace.get("target_capability_classes", [])),
                    int(provider_prefilter_trace.get("matched_provider_tool_count", 0) or 0),
                    list(provider_prefilter_trace.get("retained_builtin_tools", [])),
                )
            _log_step(
                "provider_prefilter_applied",
                before_count=int(provider_prefilter_trace.get("before_count", 0) or 0),
                after_count=int(provider_prefilter_trace.get("after_count", 0) or 0),
                enabled=bool(provider_prefilter_trace.get("enabled")),
                reason=str(provider_prefilter_trace.get("reason", "") or ""),
                target_provider_types=list(provider_prefilter_trace.get("target_provider_types", [])),
                target_capability_classes=list(
                    provider_prefilter_trace.get("target_capability_classes", [])
                ),
            )
            web_tool_verification_enforced = self._should_enforce_web_tool_verification(
                decision=tool_gate_decision,
                match_result=tool_match_result,
                available_tools=available_tools,
            )
            reasoning_retry_limit = self.REASONING_ONLY_MAX_RETRIES
            if tool_gate_decision.needs_external_system:
                reasoning_retry_limit = 0
            self._inject_tool_policy(
                deps=deps,
                decision=tool_gate_decision,
                match_result=tool_match_result,
            )
            _log_step(
                "tool_policy_injected",
                enforce_verification=web_tool_verification_enforced,
                reasoning_retry_limit=reasoning_retry_limit,
            )
            await self.runtime_events.trigger_tool_gate_evaluated(
                session_key=session_key,
                run_id=run_id,
                decision=tool_gate_decision,
            )
            await self.runtime_events.trigger_tool_matcher_resolved(
                session_key=session_key,
                run_id=run_id,
                decision=tool_gate_decision,
                match_result=tool_match_result,
            )

            if (
                tool_gate_decision.policy is ToolPolicyMode.MUST_USE_TOOL
                and tool_match_result.missing_capabilities
            ):
                warning_message = self._build_missing_capability_message(tool_match_result)
                yield StreamEvent.runtime_update(
                    "warning",
                    warning_message,
                    metadata={"phase": "gate", "elapsed": round(time.monotonic() - start_time, 1)},
                )

            if tool_gate_decision.needs_external_system and isinstance(deps.extra, dict):
                # Keep provider turns lean: avoid injecting broad skill indexes when
                # runtime has already narrowed tools to provider/skill candidates.
                deps.extra["skills_snapshot"] = []
                deps.extra["md_skills_snapshot"] = []
                deps.extra.pop("target_md_skill", None)

            system_prompt = build_system_prompt(
                self.prompt_builder,
                session=session,
                deps=deps,
                agent=runtime_agent or self.agent,
                context_window_tokens=runtime_context_window,
            )
            consume_prompt_warnings = getattr(self.prompt_builder, "consume_warnings", None)
            if callable(consume_prompt_warnings):
                for warning_message in consume_prompt_warnings():
                    if not self._should_surface_prompt_warning(warning_message):
                        logger.debug("Suppressing prompt-context warning: %s", warning_message)
                        continue
                    yield StreamEvent.runtime_update(
                        "warning",
                        warning_message,
                        metadata={
                            "phase": "prompt_context",
                            "elapsed": round(time.monotonic() - start_time, 1),
                        },
                    )

            if self.hooks:
                prompt_ctx = await self.hooks.trigger(
                    "before_prompt_build",
                    {
                        "session_key": session_key,
                        "user_message": user_message,
                        "system_prompt": system_prompt,
                    },
                )
                system_prompt = prompt_ctx.get("system_prompt", system_prompt)

            # at iter,.
            if self.compaction.should_memory_flush(
                message_history,
                session,
                context_window_override=runtime_context_window,
            ):
                await self.history.flush_history_to_timestamped_memory(
                    session_key=session_key,
                    messages=message_history,
                    deps=deps,
                    session=session,
                    context_window=runtime_context_window,
                    flushed_signatures=flushed_memory_signatures,
                )

            if message_history and self.compaction.should_compact(
                message_history,
                session,
                context_window_override=runtime_context_window,
            ):
                if self.hooks:
                    await self.hooks.trigger(
                        "before_compaction",
                        {
                            "session_key": session_key,
                            "message_count": len(message_history),
                        },
                    )
                yield StreamEvent.compaction_start()
                compressed_history = await self.compaction.compact(message_history, session)
                message_history = self.history.normalize_messages(compressed_history)
                message_history = await self.history.inject_memory_recall(message_history, deps)
                context_history_for_hooks = list(message_history)
                await session_manager.mark_compacted(session_key)
                compaction_applied = True
                yield StreamEvent.compaction_end()
                if self.hooks:
                    await self.hooks.trigger(
                        "after_compaction",
                        {
                            "session_key": session_key,
                            "message_count": len(message_history),
                        },
                    )

            # -- hook:before_agent_start --
            if self.hooks:
                start_ctx = await self.hooks.trigger(
                    "before_agent_start",
                    {
                        "session_key": session_key,
                        "user_message": user_message,
                    },
                )
                user_message = start_ctx.get("user_message", user_message)
            await self.runtime_events.trigger_llm_input(
                session_key=session_key,
                run_id=run_id,
                user_message=user_message,
                system_prompt=system_prompt,
                message_history=message_history,
            )
            payload_profile = self._build_llm_payload_profile(
                system_prompt=system_prompt,
                user_message=user_message,
                message_history=message_history,
            )
            _log_step(
                "llm_payload_profile",
                stage="pre_iter",
                **payload_profile,
            )
            if isinstance(deps.extra, dict):
                existing_profiles = deps.extra.get("_llm_payload_profiles")
                entry = {"stage": "pre_iter", **payload_profile}
                if isinstance(existing_profiles, list):
                    existing_profiles.append(entry)
                else:
                    deps.extra["_llm_payload_profiles"] = [entry]
        finally:
            state.update({
                "session_key": session_key,
                "user_message": user_message,
                "deps": deps,
                "max_tool_calls": max_tool_calls,
                "timeout_seconds": timeout_seconds,
                "_token_failover_attempt": _token_failover_attempt,
                "_emit_lifecycle_bounds": _emit_lifecycle_bounds,
                "start_time": start_time,
                "tool_calls_count": tool_calls_count,
                "compaction_applied": compaction_applied,
                "thinking_emitter": thinking_emitter,
                "persist_override_messages": persist_override_messages,
                "persist_override_base_len": persist_override_base_len,
                "runtime_agent": runtime_agent,
                "selected_token_id": selected_token_id,
                "release_slot": release_slot,
                "flushed_memory_signatures": flushed_memory_signatures,
                "extra": extra,
                "run_id": run_id,
                "tool_policy_retry_count": tool_policy_retry_count,
                "run_failed": run_failed,
                "message_history": message_history,
                "system_prompt": system_prompt,
                "final_assistant": final_assistant,
                "context_history_for_hooks": context_history_for_hooks,
                "tool_call_summaries": tool_call_summaries,
                "session_title": session_title,
                "buffered_assistant_events": buffered_assistant_events,
                "assistant_output_streamed": assistant_output_streamed,
                "tool_request_message": tool_request_message,
                "tool_gate_decision": tool_gate_decision,
                "tool_match_result": tool_match_result,
                "current_model_attempt": current_model_attempt,
                "current_attempt_started_at": current_attempt_started_at,
                "current_attempt_has_text": current_attempt_has_text,
                "current_attempt_has_tool": current_attempt_has_tool,
                "reasoning_retry_count": reasoning_retry_count,
                "run_output_start_index": run_output_start_index,
                "web_tool_verification_enforced": web_tool_verification_enforced,
                "reasoning_retry_limit": reasoning_retry_limit,
                "model_stream_timed_out": model_stream_timed_out,
                "model_timeout_error_message": model_timeout_error_message,
                "fast_path_tool_answer": fast_path_tool_answer,
                "runtime_context_window_info": runtime_context_window_info,
                "runtime_context_guard": runtime_context_guard,
                "runtime_context_window": runtime_context_window,
                "session_manager": session_manager,
                "session": session,
                "transcript": transcript,
                "all_available_tools": all_available_tools,
                "tool_groups_snapshot": tool_groups_snapshot,
                "available_tools": available_tools,
                "toolset_filter_trace": toolset_filter_trace,
                "used_toolset_fallback": used_toolset_fallback,
                "provider_hint_docs": provider_hint_docs,
                "skill_hint_docs": skill_hint_docs,
                "metadata_candidates": metadata_candidates,
                "ranking_trace": ranking_trace,
            })

