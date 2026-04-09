from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict
from contextlib import asynccontextmanager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, AsyncIterator, Optional

from app.atlasclaw.agent.context_pruning import prune_context_messages, should_apply_context_pruning
from app.atlasclaw.agent.context_window_guard import (
    ContextWindowInfo,
    evaluate_context_window_guard,
    resolve_context_window_info,
)
from app.atlasclaw.agent.runner_prompt_context import (
    build_system_prompt,
    collect_tool_groups_snapshot,
    collect_tools_snapshot,
)
from app.atlasclaw.agent.stream import StreamEvent
from app.atlasclaw.agent.thinking_stream import ThinkingStreamEmitter
from app.atlasclaw.agent.tool_gate import CapabilityMatcher
from app.atlasclaw.agent.tool_gate_models import CapabilityMatchResult, ToolGateDecision, ToolPolicyMode
from app.atlasclaw.core.deps import SkillDeps
from app.atlasclaw.session.context import SessionKey
from app.atlasclaw.tools.policy_pipeline import ToolPolicyPipeline, build_ordered_policy_layers

if TYPE_CHECKING:
    from app.atlasclaw.agent.agent_pool import AgentInstancePool
    from app.atlasclaw.agent.token_policy import DynamicTokenPolicy
    from app.atlasclaw.core.token_interceptor import TokenHealthInterceptor
    from app.atlasclaw.hooks.system import HookSystem
    from app.atlasclaw.session.manager import SessionManager
    from app.atlasclaw.session.queue import SessionQueue
    from app.atlasclaw.session.router import SessionManagerRouter


logger = logging.getLogger(__name__)


@dataclass
class _ModelNodeTimeout(RuntimeError):
    """Raised when the model stream stalls waiting for next node."""

    first_node: bool
    timeout_seconds: float


class RunnerExecutionMixin:
    async def run(
        self,
        session_key: str,
        user_message: str,
        deps: SkillDeps,
        *,
        max_tool_calls: int = 50,
        timeout_seconds: int = 600,
        _token_failover_attempt: int = 0,
        _emit_lifecycle_bounds: bool = True,
    ) -> AsyncIterator[StreamEvent]:
        """Execute one agent turn as a stream of runtime events."""
        start_time = time.monotonic()
        tool_calls_count = 0
        compaction_applied = False
        thinking_emitter = ThinkingStreamEmitter()
        persist_override_messages: Optional[list[dict]] = None
        persist_override_base_len: int = 0
        runtime_agent: Any = self.agent
        selected_token_id: Optional[str] = None
        release_slot: Optional[Any] = None
        flushed_memory_signatures: set[str] = set()
        extra = deps.extra if isinstance(deps.extra, dict) else {}
        run_id = str(extra.get("run_id", "") or "")
        tool_policy_retry_count = int(extra.get("_tool_policy_retry_count", 0) or 0)
        run_failed = False
        message_history: list[dict] = []
        system_prompt = ""
        final_assistant = ""
        context_history_for_hooks: list[dict] = []
        tool_call_summaries: list[dict[str, Any]] = []
        session_title = ""
        buffered_assistant_events: list[StreamEvent] = []
        assistant_output_streamed = False
        tool_request_message = user_message
        tool_gate_decision = ToolGateDecision(reason="Tool gate not evaluated yet.")
        tool_match_result = CapabilityMatchResult(
            resolved_policy=ToolPolicyMode.ANSWER_DIRECT,
            tool_candidates=[],
            missing_capabilities=[],
            reason="Tool matcher not evaluated yet.",
        )
        current_model_attempt = 0
        current_attempt_started_at: float | None = None
        current_attempt_has_text = False
        current_attempt_has_tool = False
        reasoning_retry_count = 0
        run_output_start_index = 0
        web_tool_verification_enforced = False
        reasoning_retry_limit = self.REASONING_ONLY_MAX_RETRIES
        model_stream_timed_out = False
        model_timeout_error_message = ""
        fast_path_tool_answer = ""

        def _log_step(step: str, **data: Any) -> None:
            payload: dict[str, Any] = {
                "session": session_key,
                "run_id": run_id,
                "step": step,
                "elapsed": round(time.monotonic() - start_time, 3),
            }
            payload.update(data)
            logger.warning("run_step %s", payload)

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

            # -- inject user_message to deps, for Skills --
            deps.user_message = user_message
            run_output_start_index = len(message_history)

            # ========================================
            # :PydanticAI iter()
            # ========================================
            try:
                model_message_history = self.history.to_model_message_history(message_history)
                provider_fast_path_turn = bool(tool_gate_decision.needs_external_system)
                async with self._run_iter_with_optional_override(
                    agent=runtime_agent,
                    user_message=user_message,
                    deps=deps,
                    message_history=model_message_history,
                    system_prompt=system_prompt,
                ) as agent_run:

                    node_count = 0
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
                            node_count += 1
                            # -- checkpoint 1:abort_signal --
                            if deps.is_aborted():
                                yield StreamEvent.lifecycle_aborted()
                                break

                            # -- checkpoint 2:--
                            if time.monotonic() - start_time > timeout_seconds:
                                yield StreamEvent.error_event("timeout")
                                break

                            # -- checkpoint 3:context -> trigger --
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
                            context_history_for_hooks = list(current_messages)
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
                                context_history_for_hooks = list(persist_override_messages)
                                persist_override_base_len = len(current_messages)
                                await session_manager.mark_compacted(session_key)
                                compaction_applied = True
                                yield StreamEvent.compaction_end()
                                if self.hooks:
                                    await self.hooks.trigger(
                                        "after_compaction",
                                        {
                                            "session_key": session_key,
                                            "message_count": len(persist_override_messages),
                                        },
                                    )
    
                            # -- hook:llm_input() --
                            if self._is_model_request_node(node):
                                current_model_attempt += 1
                                current_attempt_started_at = time.monotonic()
                                current_attempt_has_text = False
                                current_attempt_has_tool = False
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
                                            if (tool_call_summaries or provider_fast_path_turn)
                                            else "Continuing reasoning."
                                        )
                                    ),
                                    metadata={
                                        "phase": "model_request",
                                        "attempt": current_model_attempt,
                                        "elapsed": round(time.monotonic() - start_time, 1),
                                    },
                                )
    
                            # Emit model output chunks as assistant deltas.
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
                                        buffered_assistant_events.append(event)
                                    else:
                                        if event.type == "assistant":
                                            current_attempt_has_text = True
                                            assistant_output_streamed = True
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
                                        buffered_assistant_events.append(event)
                                    else:
                                        if event.type == "assistant":
                                            current_attempt_has_text = True
                                            assistant_output_streamed = True
                                        yield event
    
                            # Surface tool activity in the event stream.
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
                            if tool_calls_in_node:
                                current_attempt_has_tool = True
                                yield StreamEvent.runtime_update(
                                    "waiting_for_tool",
                                    "Preparing tool execution.",
                                    metadata={
                                        "phase": "planned",
                                        "attempt": current_model_attempt,
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
                                tool_calls_count=tool_calls_count,
                                max_tool_calls=max_tool_calls,
                                deps=deps,
                                session_key=session_key,
                                run_id=run_id,
                            )
                            tool_calls_count = tool_dispatch.tool_calls_count
                            for event in tool_dispatch.events:
                                if event.type == "assistant":
                                    assistant_output_streamed = True
                                yield event
                            if (
                                provider_fast_path_turn
                                and self._is_call_tools_node(node)
                            ):
                                post_tool_messages = self.history.normalize_messages(
                                    agent_run.all_messages()
                                )
                                if not tool_call_summaries:
                                    inferred = self._collect_tool_call_summaries_from_messages(
                                        messages=post_tool_messages,
                                        start_index=run_output_start_index,
                                    )
                                    if inferred:
                                        tool_call_summaries.extend(inferred)
                                post_tool_text = self._extract_tool_text_from_messages(
                                    messages=post_tool_messages,
                                    start_index=run_output_start_index,
                                    max_chars=9000,
                                ).strip()
                                compact_tool_answer = ""
                                if post_tool_text:
                                    compact_tool_answer = self._compact_tool_fallback_text(post_tool_text)
                                    if compact_tool_answer:
                                        fast_path_tool_answer = compact_tool_answer
                                        _log_step(
                                            "provider_fast_path_short_circuit",
                                            tool_calls=len(tool_call_summaries),
                                            tool_text_chars=len(post_tool_text),
                                            has_compact_answer=True,
                                        )
                                        break
                            if (
                                self._is_call_tools_node(node)
                                and not current_attempt_has_text
                                and not current_attempt_has_tool
                                and thinking_emitter.current_cycle_had_thinking
                            ):
                                elapsed_total = round(time.monotonic() - start_time, 1)
                                attempt_elapsed = round(
                                    time.monotonic() - current_attempt_started_at,
                                    1,
                                ) if current_attempt_started_at is not None else elapsed_total
                                should_escalate = (
                                    provider_fast_path_turn
                                    or (
                                    elapsed_total >= self.REASONING_ONLY_ESCALATION_SECONDS
                                    or reasoning_retry_count >= reasoning_retry_limit
                                    )
                                )
                                if should_escalate:
                                    if web_tool_verification_enforced and tool_gate_decision.policy in {
                                        ToolPolicyMode.MUST_USE_TOOL,
                                        ToolPolicyMode.PREFER_TOOL,
                                    }:
                                        yield StreamEvent.runtime_update(
                                            "warning",
                                            "Verification did not produce a usable tool-backed answer in this cycle.",
                                            metadata={
                                                "phase": "verification",
                                                "attempt": current_model_attempt,
                                                "elapsed": elapsed_total,
                                                "attempt_elapsed": attempt_elapsed,
                                            },
                                        )
                                        break
                                    raise RuntimeError(
                                        "The model did not produce a usable answer after bounded reasoning retries."
                                    )
                                reasoning_retry_count += 1
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
                        model_stream_timed_out = True
                        model_timeout_error_message = (
                            "The model stream timed out before producing a usable response."
                        )
                        yield StreamEvent.runtime_update(
                            "warning",
                            (
                                "Model stream timed out in this cycle. "
                                "Attempting to recover from available tool output."
                            ),
                            metadata={
                                "phase": "model_timeout",
                                "attempt": current_model_attempt,
                                "elapsed": round(time.monotonic() - start_time, 1),
                                "timeout_seconds": timeout_exc.timeout_seconds,
                            },
                        )
                        if not tool_call_summaries:
                            raise RuntimeError(model_timeout_error_message)

                    # Ensure thinking phase is properly closed if still active.
                    async for event in thinking_emitter.close_if_active():
                        yield event

                    # Persist the final normalized transcript.
                    try:
                        raw_final_messages = agent_run.all_messages()
                    except Exception:
                        raw_final_messages = list(message_history) + [
                            {"role": "user", "content": user_message}
                        ]
                    final_messages = self.history.normalize_messages(raw_final_messages)
                    if persist_override_messages is not None:
                        if len(final_messages) > persist_override_base_len > 0:
                            # Preserve override messages and append new run output.
                            final_messages = persist_override_messages + final_messages[persist_override_base_len:]
                        else:
                            final_messages = persist_override_messages
                        run_output_start_index = len(persist_override_messages)

                    final_assistant = self._extract_latest_assistant_from_messages(
                        messages=final_messages,
                        start_index=run_output_start_index,
                    )
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
                    if buffered_assistant_events and final_assistant:
                        buffered_reasoning_text = self._collect_buffered_assistant_text(
                            buffered_assistant_events
                        )
                        if buffered_reasoning_text:
                            yield StreamEvent.thinking_delta(buffered_reasoning_text)
                            yield StreamEvent.thinking_end(elapsed=0.0)
                        buffered_assistant_events.clear()
                    if buffered_assistant_events and not final_assistant:
                        # If this run executed tools, treat pre-tool assistant chatter as
                        # reasoning-only context and keep it out of the final assistant body.
                        if tool_call_summaries:
                            buffered_reasoning_text = self._collect_buffered_assistant_text(
                                buffered_assistant_events
                            )
                            if buffered_reasoning_text:
                                yield StreamEvent.thinking_delta(buffered_reasoning_text)
                                yield StreamEvent.thinking_end(elapsed=0.0)
                            buffered_assistant_events.clear()
                        else:
                            while buffered_assistant_events:
                                event = buffered_assistant_events.pop(0)
                                if event.type == "assistant":
                                    final_assistant += event.content
                                    assistant_output_streamed = True
                                yield event
                            thinking_emitter.assistant_emitted = bool(final_assistant)

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
                                if isinstance(args, dict):
                                    args_signature = str(sorted(args.items()))
                                else:
                                    args_signature = ""
                                existing_signatures.add((name, args_signature))
                            for item in inferred_tool_calls:
                                name = str(item.get("name", "") or "").strip()
                                if not name:
                                    continue
                                args = item.get("args")
                                if isinstance(args, dict):
                                    args_signature = str(sorted(args.items()))
                                else:
                                    args_signature = ""
                                signature = (name, args_signature)
                                if signature in existing_signatures:
                                    continue
                                existing_signatures.add(signature)
                                tool_call_summaries.append(item)

                    if (
                        model_stream_timed_out
                        and not final_assistant.strip()
                        and tool_call_summaries
                    ):
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

                    if (
                        provider_fast_path_turn
                        and not final_assistant.strip()
                        and tool_call_summaries
                    ):
                        provider_tool_text = self._extract_tool_text_from_messages(
                            messages=final_messages,
                            start_index=run_output_start_index,
                            max_chars=9000,
                        ).strip()
                        if provider_tool_text:
                            compact_provider_tool_text = self._compact_tool_fallback_text(
                                provider_tool_text
                            )
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

                    if not assistant_output_streamed:
                        # Try to get response from agent_run.result first (pydantic-ai structure)
                        if not final_assistant and hasattr(agent_run, "result") and agent_run.result:
                            result = agent_run.result
                            # Try response property first
                            if hasattr(result, "response") and result.response:
                                response = result.response
                                # Extract text content from response parts, excluding thinking parts
                                if hasattr(response, "parts"):
                                    for part in response.parts:
                                        part_kind = getattr(part, "part_kind", "")
                                        # Skip thinking parts, only extract text parts
                                        if part_kind != "thinking" and hasattr(part, "content") and part.content:
                                            content = str(part.content)
                                            if content:
                                                final_assistant = content
                                                break
                                elif hasattr(response, "content") and response.content:
                                    final_assistant = str(response.content)
                            # Try data property as fallback
                            if not final_assistant and hasattr(result, "data") and result.data:
                                final_assistant = str(result.data)
                        
                        # Fallback: search in final_messages
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
                    # Keep unified loop non-blocking: missing evidence emits warnings only.
                    should_fail_for_missing_evidence = False
                    should_block_assistant_emit = (
                        should_fail_for_missing_evidence
                    )
                    if (
                        not assistant_output_streamed
                        and final_assistant
                        and not should_block_assistant_emit
                    ):
                        thinking_emitter.assistant_emitted = True
                        assistant_output_streamed = True
                        yield StreamEvent.assistant_delta(final_assistant)
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
                            token_failover_attempt=_token_failover_attempt,
                            emit_lifecycle_bounds=_emit_lifecycle_bounds,
                            failure_message=failure_message,
                            missing_required_tools=missing_required_tool_names,
                            tool_policy_retry_count=tool_policy_retry_count,
                            allow_retry=not bool(tool_gate_decision.needs_external_system),
                        ):
                            tool_policy_retried = True
                            yield retry_event
                        if tool_policy_retried:
                            release_slot = None
                            selected_token_id = None
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
                            message_history=context_history_for_hooks,
                            assistant_message="",
                            tool_calls=tool_call_summaries,
                            run_status="failed",
                            error=failure_message,
                            session_title=session_title,
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
                            run_failed = True
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
                                message_history=context_history_for_hooks,
                                assistant_message="",
                                tool_calls=tool_call_summaries,
                                run_status="failed",
                                error=failure_message,
                                session_title=session_title,
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
                            _log_step("post_success_llm_completed_start")
                            await self.runtime_events.trigger_llm_completed(
                                session_key=session_key,
                                run_id=run_id,
                                assistant_message=final_assistant,
                            )
                            _log_step("post_success_llm_completed_done")
                            _log_step("post_success_persist_transcript_start")
                            await session_manager.persist_transcript(session_key, final_messages)
                            _log_step("post_success_persist_transcript_done")
                            _log_step("post_success_finalize_title_start")
                            await self._maybe_finalize_title(
                                session_manager=session_manager,
                                session_key=session_key,
                                session=session,
                                final_messages=final_messages,
                                user_message=user_message,
                            )
                            _log_step("post_success_finalize_title_done")
                            session_title = str(getattr(session, "title", "") or "")
                            _log_step("post_success_run_context_ready_start")
                            await self.runtime_events.trigger_run_context_ready(
                                session_key=session_key,
                                run_id=run_id,
                                user_message=user_message,
                                system_prompt=system_prompt,
                                message_history=context_history_for_hooks,
                                assistant_message=final_assistant,
                                tool_calls=tool_call_summaries,
                                run_status="completed",
                                session_title=session_title,
                            )
                            _log_step("post_success_run_context_ready_done")

            except Exception as e:
                logger.exception("Agent runtime exception during streaming run")
                error_text = str(e).strip()
                if not error_text:
                    if model_stream_timed_out and model_timeout_error_message:
                        error_text = model_timeout_error_message
                    else:
                        error_text = e.__class__.__name__
                retry_error: Exception = e
                if not str(e).strip():
                    retry_error = RuntimeError(error_text)
                hard_failure_retried = False
                async for retry_event in self._retry_after_hard_token_failure(
                    error=retry_error,
                    session_key=session_key,
                    user_message=user_message,
                    deps=deps,
                    selected_token_id=selected_token_id,
                    release_slot=release_slot,
                    thinking_emitter=thinking_emitter,
                    start_time=start_time,
                    max_tool_calls=max_tool_calls,
                    timeout_seconds=timeout_seconds,
                    token_failover_attempt=_token_failover_attempt,
                    emit_lifecycle_bounds=_emit_lifecycle_bounds,
                ):
                    hard_failure_retried = True
                    yield retry_event
                if hard_failure_retried:
                    release_slot = None
                    selected_token_id = None
                    return
                run_failed = True
                await self.runtime_events.trigger_llm_failed(
                    session_key=session_key,
                    run_id=run_id,
                    error=error_text,
                )
                await self.runtime_events.trigger_run_failed(
                    session_key=session_key,
                    run_id=run_id,
                    error=error_text,
                )
                await self.runtime_events.trigger_run_context_ready(
                    session_key=session_key,
                    run_id=run_id,
                    user_message=user_message,
                    system_prompt=system_prompt,
                    message_history=context_history_for_hooks,
                    assistant_message=final_assistant,
                    tool_calls=tool_call_summaries,
                    run_status="failed",
                    error=error_text,
                    session_title=session_title,
                )
                # Close thinking phase on exception to maintain contract
                async for event in thinking_emitter.close_if_active():
                    yield event
                        
                # Surface agent runtime errors as stream events.
                yield StreamEvent.runtime_update(
                    "failed",
                    f"Agent runtime error: {error_text}",
                    metadata={"phase": "exception", "elapsed": round(time.monotonic() - start_time, 1)},
                )
                yield StreamEvent.error_event(f"agent_error: {error_text}")

            # -- hook:agent_end --
            if not run_failed:
                await self.runtime_events.trigger_agent_end(
                    session_key=session_key,
                    run_id=run_id,
                    tool_calls_count=tool_calls_count,
                    compaction_applied=compaction_applied,
                )

            if _emit_lifecycle_bounds:
                yield StreamEvent.lifecycle_end()

        except Exception as e:
            logger.exception("Agent runtime outer exception")
            await self.runtime_events.trigger_run_failed(
                session_key=session_key,
                run_id=run_id,
                error=str(e),
            )
            await self.runtime_events.trigger_run_context_ready(
                session_key=session_key,
                run_id=run_id,
                user_message=user_message,
                system_prompt=system_prompt,
                message_history=context_history_for_hooks,
                assistant_message=final_assistant,
                tool_calls=tool_call_summaries,
                run_status="failed",
                error=str(e),
                session_title=session_title,
            )
            # Close thinking phase on exception to maintain contract
            async for event in thinking_emitter.close_if_active():
                yield event
                
            yield StreamEvent.runtime_update(
                "failed",
                str(e),
                metadata={"phase": "exception", "elapsed": round(time.monotonic() - start_time, 1)},
            )
            yield StreamEvent.error_event(str(e))
        finally:
            if selected_token_id and self.token_interceptor is not None:
                headers = self._extract_rate_limit_headers(deps)
                if headers:
                    self.token_interceptor.on_response(selected_token_id, headers)
            if release_slot is not None:
                release_slot()

    async def _retry_after_hard_token_failure(
        self,
        *,
        error: Exception,
        session_key: str,
        user_message: str,
        deps: SkillDeps,
        selected_token_id: Optional[str],
        release_slot: Optional[Any],
        thinking_emitter: ThinkingStreamEmitter,
        start_time: float,
        max_tool_calls: int,
        timeout_seconds: int,
        token_failover_attempt: int,
        emit_lifecycle_bounds: bool,
    ) -> AsyncIterator[StreamEvent]:
        """Rotate away from a hard-failed token and retry the same run once."""
        if (
            self.token_policy is None
            or not self._is_hard_token_failure(error)
        ):
            logger.warning(
                "token failover skipped: token_policy=%s hard_failure=%s error=%s",
                self.token_policy is not None,
                self._is_hard_token_failure(error),
                str(error),
            )
            return
        pool_max_attempts = max(len(self.token_policy.token_pool.tokens) - 1, 0)
        configured_max_attempts = max(
            int(getattr(self, "TOKEN_FAILOVER_MAX_ATTEMPTS", pool_max_attempts) or 0),
            0,
        )
        max_failover_attempts = min(pool_max_attempts, configured_max_attempts)
        if token_failover_attempt >= max_failover_attempts:
            logger.warning(
                "token failover exhausted: attempt=%s max_attempts=%s",
                token_failover_attempt,
                max_failover_attempts,
            )
            return

        extra = deps.extra if isinstance(deps.extra, dict) else {}
        provider = extra.get("provider") if isinstance(extra.get("provider"), str) else None
        model = extra.get("model") if isinstance(extra.get("model"), str) else None
        error_text = str(error)
        next_token = None
        if selected_token_id:
            if self.token_interceptor is not None:
                self.token_interceptor.on_hard_failure(selected_token_id, error_text)
            next_token = self.token_policy.mark_session_token_unhealthy(
                session_key,
                reason=error_text,
                provider=provider,
                model=model,
            )
            if next_token is None and provider:
                next_token = self.token_policy.mark_session_token_unhealthy(
                    session_key,
                    reason=error_text,
                    provider=provider,
                    model=None,
                )
            if next_token is None:
                next_token = self.token_policy.mark_session_token_unhealthy(
                    session_key,
                    reason=error_text,
                    provider=None,
                    model=None,
                )
        else:
            next_token = self.token_policy.get_or_select_session_token(
                session_key,
                provider=provider,
                model=model,
            )
            if next_token is None and provider:
                next_token = self.token_policy.get_or_select_session_token(
                    session_key,
                    provider=provider,
                    model=None,
                )
            if next_token is None:
                next_token = self.token_policy.get_or_select_session_token(
                    session_key,
                    provider=None,
                    model=None,
                )

        if next_token is None or (selected_token_id and next_token.token_id == selected_token_id):
            logger.warning(
                "token failover unavailable: selected_token_id=%s next_token=%s",
                selected_token_id,
                None if next_token is None else next_token.token_id,
            )
            return

        async for event in thinking_emitter.close_if_active():
            yield event
        if release_slot is not None:
            release_slot()

        yield StreamEvent.runtime_update(
            "retrying",
            (
                (
                    "Current model token failed with a provider/model-side error or stream stall. "
                    f"Switching to fallback model token `{next_token.token_id}`."
                )
                if selected_token_id
                else (
                    "Current run failed before a managed token was pinned. "
                    f"Switching to managed fallback token `{next_token.token_id}`."
                )
            ),
            metadata={
                "phase": "token_failover",
                "elapsed": round(time.monotonic() - start_time, 1),
                "attempt": token_failover_attempt + 1,
                "failed_token_id": selected_token_id,
                "fallback_token_id": next_token.token_id,
            },
        )
        async for event in self.run(
            session_key=session_key,
            user_message=user_message,
            deps=deps,
            max_tool_calls=max_tool_calls,
            timeout_seconds=timeout_seconds,
            _token_failover_attempt=token_failover_attempt + 1,
            _emit_lifecycle_bounds=False,
        ):
            yield event
        if emit_lifecycle_bounds:
            yield StreamEvent.lifecycle_end()
        return

    async def _retry_after_tool_policy_failure(
        self,
        *,
        session_key: str,
        user_message: str,
        deps: SkillDeps,
        release_slot: Optional[Any],
        selected_token_id: Optional[str],
        start_time: float,
        max_tool_calls: int,
        timeout_seconds: int,
        token_failover_attempt: int,
        emit_lifecycle_bounds: bool,
        failure_message: str,
        missing_required_tools: list[str],
        tool_policy_retry_count: int,
        allow_retry: bool,
    ) -> AsyncIterator[StreamEvent]:
        """Retry once when must-use-tool policy produced no usable tool evidence."""
        if not allow_retry:
            return
        if tool_policy_retry_count >= self.TOOL_POLICY_MAX_RETRIES:
            return

        if release_slot is not None:
            release_slot()

        if not isinstance(deps.extra, dict):
            deps.extra = {}
        deps.extra["_tool_policy_retry_count"] = tool_policy_retry_count + 1
        deps.extra["tool_policy_retry_reason"] = "missing_required_tool_evidence"
        deps.extra["tool_policy_retry_missing_tools"] = list(missing_required_tools)

        yield StreamEvent.runtime_update(
            "retrying",
            (
                "Tool-backed verification did not produce usable evidence for required tools. "
                "Retrying once with stricter tool-policy guidance."
            ),
            metadata={
                "phase": "tool_policy_retry",
                "elapsed": round(time.monotonic() - start_time, 1),
                "attempt": tool_policy_retry_count + 1,
                "failed_token_id": selected_token_id,
                "missing_required_tools": list(missing_required_tools),
                "failure_message": failure_message,
            },
        )

        async for event in self.run(
            session_key=session_key,
            user_message=user_message,
            deps=deps,
            max_tool_calls=max_tool_calls,
            timeout_seconds=timeout_seconds,
            _token_failover_attempt=token_failover_attempt,
            _emit_lifecycle_bounds=False,
        ):
            yield event
        if emit_lifecycle_bounds:
            yield StreamEvent.lifecycle_end()
        return

    def _is_hard_token_failure(self, error: Exception) -> bool:
        """Return true when an error indicates the current token should be evicted."""
        lowered = str(error).lower()
        hard_markers = (
            "status_code: 401",
            "status_code: 403",
            "status_code: 429",
            "authenticationerror",
            "accountoverdueerror",
            "forbidden",
            "invalid api key",
            "insufficient_quota",
            "api key format is incorrect",
            "provider returned error', 'code': 429",
            '"code": 429',
            "rate-limited upstream",
            "too many requests",
            "rate limit",
            "model stream timed out before producing a usable response",
            "model stream timed out",
            "stream timed out",
        )
        return any(marker in lowered for marker in hard_markers)

    async def _resolve_runtime_agent(
        self,
        session_key: str,
        deps: SkillDeps,
    ) -> tuple[Any, Optional[str], Optional[Any]]:
        """Resolve runtime agent instance and optional semaphore release callback."""
        if self.token_policy is None or self.agent_pool is None or self.agent_factory is None:
            return self.agent, None, None

        extra = deps.extra if isinstance(deps.extra, dict) else {}
        provider = extra.get("provider") if isinstance(extra.get("provider"), str) else None
        model = extra.get("model") if isinstance(extra.get("model"), str) else None

        token = self.token_policy.get_or_select_session_token(
            session_key,
            provider=provider,
            model=model,
        )
        if token is None and provider:
            token = self.token_policy.get_or_select_session_token(
                session_key,
                provider=provider,
                model=None,
            )
        if token is None:
            token = self.token_policy.get_or_select_session_token(
                session_key,
                provider=None,
                model=None,
            )
        if token is None:
            return self.agent, None, None

        instance = await self.agent_pool.get_or_create(
            self.agent_id,
            token,
            self.agent_factory,
        )
        await instance.concurrency_sem.acquire()
        return instance.agent, token.token_id, instance.concurrency_sem.release

    def _extract_rate_limit_headers(self, deps: SkillDeps) -> dict[str, str]:
        """Best-effort extraction of ratelimit headers from deps.extra."""
        extra = deps.extra if isinstance(deps.extra, dict) else {}
        candidates = [
            extra.get("rate_limit_headers"),
            extra.get("response_headers"),
            extra.get("llm_response_headers"),
        ]
        for candidate in candidates:
            if isinstance(candidate, dict):
                return {str(k): str(v) for k, v in candidate.items()}
        return {}

    def _resolve_runtime_context_window_info(
        self,
        selected_token_id: Optional[str],
        deps: SkillDeps,
    ) -> ContextWindowInfo:
        """Resolve context window info with source tags for runtime guard checks."""
        selected_token_window: Optional[int] = None
        if selected_token_id and self.token_policy is not None:
            token = self.token_policy.token_pool.tokens.get(selected_token_id)
            context_window = getattr(token, "context_window", None) if token else None
            if isinstance(context_window, int) and context_window > 0:
                selected_token_window = context_window

        extra = deps.extra if isinstance(deps.extra, dict) else {}
        runtime_override = extra.get("context_window") or extra.get("model_context_window")
        models_config_window = (
            extra.get("models_config_context_window")
            or extra.get("configured_context_window")
            or extra.get("provider_config_context_window")
        )
        default_window = self.compaction.config.context_window

        return resolve_context_window_info(
            selected_token_window=selected_token_window,
            models_config_window=models_config_window if isinstance(models_config_window, int) else None,
            runtime_override_window=runtime_override if isinstance(runtime_override, int) else None,
            default_window=default_window,
        )

    def _resolve_runtime_context_window(
        self,
        selected_token_id: Optional[str],
        deps: SkillDeps,
    ) -> Optional[int]:
        """Backward-compatible helper returning only resolved token count."""
        return self._resolve_runtime_context_window_info(selected_token_id, deps).tokens

    def _resolve_session_manager(self, session_key: str, deps: SkillDeps) -> Any:
        """Resolve the correct per-user session manager for the active session."""
        parsed = SessionKey.from_string(session_key)
        scoped_manager = getattr(deps, "session_manager", None)
        scoped_user_id = getattr(scoped_manager, "user_id", None)
        if scoped_manager is not None and scoped_user_id == parsed.user_id:
            return scoped_manager
        if self.session_manager_router is not None:
            return self.session_manager_router.for_session_key(session_key)
        return self.sessions

    async def _maybe_set_draft_title(
        self,
        *,
        session_manager: Any,
        session_key: str,
        session: Any,
        transcript: list[Any],
        user_message: str,
    ) -> None:
        """Create a draft title for brand-new chat threads."""
        if getattr(session, "title_status", "empty") not in {"", "empty"}:
            return
        if transcript:
            return
        draft_title = self.title_generator.build_draft_title(user_message)
        await session_manager.update_title(
            session_key,
            title=draft_title,
            title_status="draft",
        )
        session.title = draft_title
        session.title_status = "draft"

    async def _maybe_finalize_title(
        self,
        *,
        session_manager: Any,
        session_key: str,
        session: Any,
        final_messages: list[dict],
        user_message: str,
    ) -> None:
        """Promote a draft title to a stable final title after the first assistant reply."""
        if getattr(session, "title_status", "empty") == "final":
            return
        assistant_message = next(
            (
                msg.get("content", "")
                for msg in final_messages
                if msg.get("role") == "assistant" and msg.get("content")
            ),
            "",
        )
        final_title = self.title_generator.build_final_title(
            first_user_message=user_message,
            first_assistant_message=assistant_message,
            existing_title=getattr(session, "title", ""),
        )
        await session_manager.update_title(
            session_key,
            title=final_title,
            title_status="final",
        )
        session.title = final_title
        session.title_status = "final"

    @asynccontextmanager

    async def _run_iter_with_optional_override(
        self,
        *,
        agent: Any,
        user_message: str,
        deps: SkillDeps,
        message_history: list[dict],
        system_prompt: str,
    ):

        """Run `agent.iter()` with optional system-prompt overrides."""
        override_factory = getattr(agent, "override", None)

        if callable(override_factory) and system_prompt:
            override_cm = nullcontext()
            override_candidates = (
                {"instructions": system_prompt},
                {"system_prompt": system_prompt},
            )
            for override_kwargs in override_candidates:
                try:
                    override_cm = override_factory(**override_kwargs)
                    break
                except TypeError:
                    continue
        else:
            override_cm = nullcontext()

        if hasattr(override_cm, "__aenter__"):
            async with override_cm:
                async with agent.iter(
                    user_message,
                    deps=deps,
                    message_history=message_history,
                ) as agent_run:
                    yield agent_run
            return

        with override_cm:
            async with agent.iter(
                user_message,
                deps=deps,
                message_history=message_history,
            ) as agent_run:

                yield agent_run

    async def _iter_agent_nodes_with_timeout(
        self,
        agent_run: Any,
        *,
        first_node_timeout_seconds: Optional[float] = None,
        next_node_timeout_seconds: Optional[float] = None,
    ) -> AsyncIterator[Any]:
        iterator = agent_run.__aiter__()
        waiting_for_first_node = True
        first_timeout = (
            float(first_node_timeout_seconds)
            if first_node_timeout_seconds is not None
            else float(self.MODEL_FIRST_NODE_TIMEOUT_SECONDS)
        )
        next_timeout = (
            float(next_node_timeout_seconds)
            if next_node_timeout_seconds is not None
            else float(self.MODEL_NEXT_NODE_TIMEOUT_SECONDS)
        )
        while True:
            timeout_seconds = (
                first_timeout
                if waiting_for_first_node
                else next_timeout
            )
            try:
                node = await asyncio.wait_for(iterator.__anext__(), timeout=timeout_seconds)
            except StopAsyncIteration:
                return
            except asyncio.TimeoutError as exc:
                raise _ModelNodeTimeout(
                    first_node=waiting_for_first_node,
                    timeout_seconds=float(timeout_seconds),
                ) from exc
            waiting_for_first_node = False
            yield node

    def _is_model_request_node(self, node: Any) -> bool:
        """Return whether a node represents a model request boundary."""
        node_type = type(node).__name__.lower()
        return "modelrequest" in node_type or node_type.endswith("requestnode")

    def _is_call_tools_node(self, node: Any) -> bool:
        """Return whether a node represents the tool-dispatch boundary."""
        node_type = type(node).__name__.lower()
        return "calltools" in node_type

    def _build_turn_toolset(
        self,
        *,
        deps: SkillDeps,
        session_key: str,
        all_tools: list[dict[str, Any]],
        tool_groups: dict[str, list[str]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
        """Filter runtime tools through an ordered allow/deny policy pipeline."""
        if not all_tools:
            return [], [], False

        policy = self._resolve_toolset_policy_payload(deps)
        provider_type = ""
        if isinstance(getattr(deps, "extra", None), dict):
            provider_instance = deps.extra.get("provider_instance")
            if isinstance(provider_instance, dict):
                provider_type = str(provider_instance.get("provider_type", "")).strip()
        cache_key = self._build_turn_toolset_cache_key(
            session_key=session_key,
            policy=policy,
            provider_type=provider_type,
            all_tools=all_tools,
            tool_groups=tool_groups,
        )
        cached = self._get_cached_turn_toolset(cache_key)
        if cached is not None:
            return cached

        layers = build_ordered_policy_layers(
            policy=policy,
            provider_type=provider_type,
            agent_id=getattr(self, "agent_id", "") or "",
            channel=str(getattr(deps, "channel", "") or ""),
            session_key=session_key,
        )
        pipeline = ToolPolicyPipeline(
            tools=all_tools,
            group_map=tool_groups,
            aliases=self._build_tool_aliases(all_tools),
        )
        result = pipeline.run(layers)

        filtered_names = set(result.tool_names)
        filtered_tools = [
            tool
            for tool in all_tools
            if str(tool.get("name", "")).strip() in filtered_names
        ]
        if filtered_tools:
            payload = (filtered_tools, result.trace, False)
            self._store_turn_toolset_cache(cache_key=cache_key, payload=payload)
            return payload

        fallback_tools = self._safe_fallback_toolset(all_tools)
        if fallback_tools:
            payload = (fallback_tools, result.trace, True)
            self._store_turn_toolset_cache(cache_key=cache_key, payload=payload)
            return payload
        payload = (list(all_tools), result.trace, True)
        self._store_turn_toolset_cache(cache_key=cache_key, payload=payload)
        return payload

    def _build_turn_toolset_cache_key(
        self,
        *,
        session_key: str,
        policy: dict[str, Any],
        provider_type: str,
        all_tools: list[dict[str, Any]],
        tool_groups: dict[str, list[str]],
    ) -> str:
        policy_blob = json.dumps(policy, ensure_ascii=False, sort_keys=True)
        toolset_signature = self._build_toolset_signature(all_tools)
        group_rows = []
        for group_id in sorted((tool_groups or {}).keys()):
            members = sorted(
                str(item).strip()
                for item in (tool_groups.get(group_id) or [])
                if str(item).strip()
            )
            group_rows.append(f"{group_id}:{','.join(members)}")
        payload = "\n".join(
            [
                str(session_key or "").strip(),
                str(provider_type or "").strip().lower(),
                policy_blob,
                toolset_signature,
                "\n".join(group_rows),
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _get_cached_turn_toolset(
        self,
        cache_key: str,
    ) -> Optional[tuple[list[dict[str, Any]], list[dict[str, Any]], bool]]:
        cache = getattr(self, "_turn_toolset_cache", None)
        if not isinstance(cache, OrderedDict):
            return None
        entry = cache.get(cache_key)
        if not entry:
            return None
        expires_at, tools, trace, used_fallback = entry
        if float(expires_at) <= time.monotonic():
            cache.pop(cache_key, None)
            return None
        cache.move_to_end(cache_key)
        return (
            [dict(item) for item in tools if isinstance(item, dict)],
            [dict(item) for item in trace if isinstance(item, dict)],
            bool(used_fallback),
        )

    def _store_turn_toolset_cache(
        self,
        *,
        cache_key: str,
        payload: tuple[list[dict[str, Any]], list[dict[str, Any]], bool],
    ) -> None:
        cache = getattr(self, "_turn_toolset_cache", None)
        if not isinstance(cache, OrderedDict):
            return
        ttl_seconds = max(
            1.0,
            float(getattr(self, "TURN_TOOLSET_CACHE_TTL_SECONDS", 300.0) or 300.0),
        )
        max_entries = max(
            32,
            int(getattr(self, "TURN_TOOLSET_CACHE_MAX_ENTRIES", 256) or 256),
        )
        expires_at = time.monotonic() + ttl_seconds
        tools, trace, used_fallback = payload
        cache[cache_key] = (
            expires_at,
            [dict(item) for item in tools if isinstance(item, dict)],
            [dict(item) for item in trace if isinstance(item, dict)],
            bool(used_fallback),
        )
        cache.move_to_end(cache_key)
        now = time.monotonic()
        stale_keys = [key for key, (expire_ts, *_rest) in list(cache.items()) if float(expire_ts) <= now]
        for key in stale_keys:
            cache.pop(key, None)
        while len(cache) > max_entries:
            cache.popitem(last=False)

    @staticmethod
    def _resolve_toolset_policy_payload(deps: SkillDeps) -> dict[str, Any]:
        extra = deps.extra if isinstance(getattr(deps, "extra", None), dict) else {}
        payload = extra.get("toolset_policy")
        if isinstance(payload, dict):
            return payload
        return {}

    @staticmethod
    def _build_tool_aliases(tools: list[dict[str, Any]]) -> dict[str, list[str]]:
        aliases: dict[str, list[str]] = {}
        capability_map: dict[str, list[str]] = {}
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            name = str(tool.get("name", "")).strip()
            if not name:
                continue
            capability = str(tool.get("capability_class", "")).strip()
            if capability:
                capability_map.setdefault(capability, []).append(name)
        for capability, members in capability_map.items():
            aliases[capability] = members
            if capability.startswith("provider:"):
                provider_type = capability.split(":", 1)[1].strip()
                if provider_type:
                    aliases.setdefault(f"group:{provider_type}", members)
        return aliases

    @staticmethod
    def _safe_fallback_toolset(all_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        preferred_names = {
            "session_status",
            "web_search",
            "web_fetch",
            "openmeteo_weather",
            "list_provider_instances",
            "select_provider_instance",
        }
        preferred: list[dict[str, Any]] = []
        provider_or_skill: list[dict[str, Any]] = []
        for tool in all_tools:
            if not isinstance(tool, dict):
                continue
            name = str(tool.get("name", "")).strip()
            if not name:
                continue
            capability = str(tool.get("capability_class", "")).strip()
            if name in preferred_names:
                preferred.append(tool)
                continue
            if capability.startswith("provider:") or capability == "skill":
                provider_or_skill.append(tool)

        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in [*provider_or_skill, *preferred]:
            name = str(item.get("name", "")).strip()
            if not name or name in seen:
                continue
            seen.add(name)
            merged.append(item)
        return merged

    @staticmethod
    def _build_filtered_group_map(
        original_groups: dict[str, list[str]],
        filtered_tools: list[dict[str, Any]],
    ) -> dict[str, list[str]]:
        allowed = {
            str(tool.get("name", "")).strip()
            for tool in filtered_tools
            if isinstance(tool, dict) and str(tool.get("name", "")).strip()
        }
        filtered: dict[str, list[str]] = {}
        for group_id, members in (original_groups or {}).items():
            normalized_group = str(group_id or "").strip()
            if not normalized_group:
                continue
            kept = [
                str(member).strip()
                for member in (members or [])
                if str(member).strip() in allowed
            ]
            if kept:
                filtered[normalized_group] = kept
        return filtered

    @staticmethod
    def _should_surface_prompt_warning(warning_message: Any) -> bool:
        normalized = str(warning_message or "").strip().lower()
        if not normalized:
            return False
        if normalized.startswith("missing bootstrap file:"):
            return False
        return True

    @classmethod
    def _build_llm_payload_profile(
        cls,
        *,
        system_prompt: str,
        user_message: str,
        message_history: list[dict],
    ) -> dict[str, Any]:
        system_text = str(system_prompt or "")
        user_text = str(user_message or "")
        history_rows = [cls._normalize_payload_message(row) for row in (message_history or [])]

        system_chars = len(system_text)
        user_chars = len(user_text)
        history_chars = sum(len(row) for row in history_rows)

        system_bytes = len(system_text.encode("utf-8", errors="ignore"))
        user_bytes = len(user_text.encode("utf-8", errors="ignore"))
        history_bytes = sum(len(row.encode("utf-8", errors="ignore")) for row in history_rows)

        total_chars = system_chars + user_chars + history_chars
        total_bytes = system_bytes + user_bytes + history_bytes
        estimated_tokens = cls._estimate_tokens_by_chars(total_chars)

        duplicate_message_count, duplicate_group_count = cls._count_duplicate_history_messages(
            history_rows
        )
        history_count = len(history_rows)
        duplicate_ratio = (
            round(float(duplicate_message_count) / float(history_count), 4)
            if history_count > 0
            else 0.0
        )
        max_history_message_chars = max((len(row) for row in history_rows), default=0)
        user_repeated_in_history = cls._has_user_message_duplicate_in_history(
            user_text,
            history_rows,
        )

        return {
            "system_prompt_chars": system_chars,
            "system_prompt_bytes": system_bytes,
            "history_message_count": history_count,
            "history_chars": history_chars,
            "history_bytes": history_bytes,
            "history_max_message_chars": max_history_message_chars,
            "history_duplicate_messages": duplicate_message_count,
            "history_duplicate_groups": duplicate_group_count,
            "history_duplicate_ratio": duplicate_ratio,
            "user_message_chars": user_chars,
            "user_message_bytes": user_bytes,
            "user_message_repeated_in_history": user_repeated_in_history,
            "total_chars": total_chars,
            "total_bytes": total_bytes,
            "estimated_tokens": estimated_tokens,
        }

    @staticmethod
    def _normalize_payload_message(message: Any) -> str:
        if not isinstance(message, dict):
            return str(message or "")
        role = str(message.get("role", "") or "").strip()
        content = message.get("content", "")
        if isinstance(content, (dict, list)):
            content_text = json.dumps(content, ensure_ascii=False, sort_keys=True)
        else:
            content_text = str(content or "")
        name = str(message.get("name", "") or "").strip()
        if name:
            return f"{role}:{name}:{content_text}"
        return f"{role}:{content_text}"

    @staticmethod
    def _estimate_tokens_by_chars(char_count: int) -> int:
        if char_count <= 0:
            return 0
        # Rough multilingual estimate for runtime observability.
        return max(1, int((char_count + 3) / 4))

    @staticmethod
    def _count_duplicate_history_messages(history_rows: list[str]) -> tuple[int, int]:
        if not history_rows:
            return 0, 0
        counts: dict[str, int] = {}
        for row in history_rows:
            normalized = " ".join(str(row or "").split()).strip()
            if not normalized:
                continue
            digest = hashlib.sha1(normalized.encode("utf-8", errors="ignore")).hexdigest()
            counts[digest] = counts.get(digest, 0) + 1
        duplicate_messages = sum(max(0, count - 1) for count in counts.values() if count > 1)
        duplicate_groups = sum(1 for count in counts.values() if count > 1)
        return duplicate_messages, duplicate_groups

    @staticmethod
    def _has_user_message_duplicate_in_history(user_message: str, history_rows: list[str]) -> bool:
        normalized_user = " ".join(str(user_message or "").split()).strip()
        if not normalized_user:
            return False
        user_entry = f"user:{normalized_user}"
        for row in history_rows:
            normalized_row = " ".join(str(row or "").split()).strip()
            if normalized_row == user_entry:
                return True
        return False

    @staticmethod
    def _deduplicate_message_history(messages: list[dict]) -> list[dict]:
        if len(messages) <= 1:
            return messages

        head_system: Optional[dict] = None
        core_messages = messages
        first = messages[0]
        if isinstance(first, dict) and str(first.get("role", "")).strip().lower() == "system":
            head_system = first
            core_messages = messages[1:]

        seen_signatures: set[str] = set()
        dedup_reversed: list[dict] = []
        for msg in reversed(core_messages):
            if not isinstance(msg, dict):
                dedup_reversed.append(msg)
                continue
            role = str(msg.get("role", "")).strip().lower()
            if role != "user":
                dedup_reversed.append(msg)
                continue
            if msg.get("tool_calls") or msg.get("tool_name") or msg.get("tool_call_id"):
                dedup_reversed.append(msg)
                continue
            normalized_content = " ".join(str(msg.get("content", "") or "").split()).strip()
            if not normalized_content:
                dedup_reversed.append(msg)
                continue
            user_identity = str(
                msg.get("user_id")
                or msg.get("name")
                or msg.get("sender_id")
                or "current_user"
            ).strip().lower()
            signature = f"{role}:{user_identity}:{normalized_content}"
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            dedup_reversed.append(msg)

        deduped = list(reversed(dedup_reversed))
        if head_system is not None:
            return [head_system, *deduped]
        return deduped

    async def run_single(
        self,
        user_message: str,
        deps: SkillDeps,
        *,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Run a single non-streaming agent call."""
        # Simplified helper that bypasses the streaming session pipeline.
        try:
            result = await self.agent.run(
                user_message,
                deps=deps,
            )
            return result.output if hasattr(result, "output") else str(result)
        except Exception as e:
            return f"[Error: {str(e)}]"

