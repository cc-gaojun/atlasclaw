from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
import inspect
import json
import logging
import re
import time
from collections import OrderedDict
from contextlib import asynccontextmanager, nullcontext
from typing import Any, Optional

from app.atlasclaw.agent.stream import StreamEvent
from app.atlasclaw.agent.tool_gate import CapabilityMatcher
from app.atlasclaw.agent.tool_gate_models import CapabilityMatchResult, ToolGateDecision, ToolPolicyMode
from app.atlasclaw.core.deps import SkillDeps


logger = logging.getLogger(__name__)


class _ModelToolGateClassifier:
    """Model-backed classifier used by the runtime when a direct model call is available."""

    def __init__(
        self,
        *,
        runner: "AgentRunner",
        deps: SkillDeps,
        available_tools: list[dict[str, Any]],
        agent: Optional[Any] = None,
        agent_resolver: Optional[Any] = None,
    ) -> None:
        self._runner = runner
        self._agent = agent
        self._agent_resolver = agent_resolver
        self._deps = deps
        self._available_tools = available_tools

    async def _resolve_agent(self) -> Optional[Any]:
        if self._agent is not None:
            return self._agent
        if self._agent_resolver is None:
            return None
        resolved = self._agent_resolver()
        if inspect.isawaitable(resolved):
            resolved = await resolved
        self._agent = resolved
        return resolved

    async def classify(
        self,
        user_message: str,
        recent_history: list[dict[str, Any]],
    ) -> Optional[ToolGateDecision]:
        classifier_agent = await self._resolve_agent()
        if classifier_agent is None:
            return None
        return await self._runner._classify_tool_gate_with_model(
            agent=classifier_agent,
            deps=self._deps,
            user_message=user_message,
            recent_history=recent_history,
            available_tools=self._available_tools,
        )


class RunnerToolGateMixin:
    @staticmethod
    def _build_toolset_signature(available_tools: list[dict[str, Any]]) -> str:
        signatures: list[str] = []
        for tool in available_tools:
            if not isinstance(tool, dict):
                continue
            name = str(tool.get("name", "") or "").strip()
            if not name:
                continue
            capability = str(tool.get("capability_class", "") or "").strip().lower()
            provider_type = str(tool.get("provider_type", "") or "").strip().lower()
            signatures.append(f"{name}|{capability}|{provider_type}")
        signatures.sort()
        return "\n".join(signatures)

    def _build_tool_gate_cache_key(
        self,
        *,
        session_key: str,
        resolved_tool_request: str,
        used_follow_up_context: bool,
        recent_history: list[dict[str, Any]],
        available_tools: list[dict[str, Any]],
        metadata_candidates: Optional[dict[str, Any]] = None,
    ) -> str:
        history_parts: list[str] = []
        if used_follow_up_context:
            for item in recent_history[-4:]:
                if not isinstance(item, dict):
                    continue
                role = str(item.get("role", "") or "").strip()
                content = " ".join(str(item.get("content", "") or "").split()).strip()
                if role and content:
                    history_parts.append(f"{role}:{content}")
        metadata_signature = ""
        if isinstance(metadata_candidates, dict):
            metadata_signature = json.dumps(
                {
                    "providers": list(metadata_candidates.get("preferred_provider_types", []) or []),
                    "capabilities": list(
                        metadata_candidates.get("preferred_capability_classes", []) or []
                    ),
                    "tools": list(metadata_candidates.get("preferred_tool_names", []) or []),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        payload = "\n".join(
            [
                str(session_key or "").strip(),
                " ".join(str(resolved_tool_request or "").split()).strip(),
                "1" if used_follow_up_context else "0",
                "\n".join(history_parts),
                self._build_toolset_signature(available_tools),
                metadata_signature,
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _get_cached_tool_gate_decision(self, cache_key: str) -> Optional[ToolGateDecision]:
        cache = getattr(self, "_tool_gate_decision_cache", None)
        if not isinstance(cache, OrderedDict):
            return None
        entry = cache.get(cache_key)
        if not entry:
            return None
        expires_at, payload = entry
        if float(expires_at) <= time.monotonic():
            cache.pop(cache_key, None)
            return None
        cache.move_to_end(cache_key)
        try:
            return ToolGateDecision.model_validate(dict(payload))
        except Exception:
            cache.pop(cache_key, None)
            return None

    def _store_tool_gate_decision_cache(
        self,
        *,
        cache_key: str,
        decision: ToolGateDecision,
    ) -> None:
        cache = getattr(self, "_tool_gate_decision_cache", None)
        if not isinstance(cache, OrderedDict):
            return
        ttl_seconds = max(
            1.0,
            float(getattr(self, "TOOL_GATE_DECISION_CACHE_TTL_SECONDS", 300.0) or 300.0),
        )
        max_entries = max(
            32,
            int(getattr(self, "TOOL_GATE_DECISION_CACHE_MAX_ENTRIES", 512) or 512),
        )
        now = time.monotonic()
        expires_at = now + ttl_seconds
        cache[cache_key] = (expires_at, decision.model_dump(mode="python"))
        cache.move_to_end(cache_key)
        stale_keys = [key for key, (expire_ts, _) in list(cache.items()) if float(expire_ts) <= now]
        for key in stale_keys:
            cache.pop(key, None)
        while len(cache) > max_entries:
            cache.popitem(last=False)

    def _resolve_tool_gate_classifier(
        self,
        *,
        agent: Any,
        deps: SkillDeps,
        available_tools: list[dict[str, Any]],
    ) -> Optional[Any]:
        extra = deps.extra if isinstance(deps.extra, dict) else {}
        explicit_classifier = extra.get("tool_gate_classifier")
        if explicit_classifier is not None:
            return explicit_classifier
        if not self.tool_gate_model_classifier_enabled:
            return None
        classifier_agent = self._select_tool_gate_classifier_agent(agent)
        if classifier_agent is None:
            return None
        return _ModelToolGateClassifier(
            runner=self,
            deps=deps,
            available_tools=available_tools,
            agent=classifier_agent if not callable(classifier_agent) else None,
            agent_resolver=classifier_agent if callable(classifier_agent) else None,
        )

    def _select_tool_gate_classifier_agent(self, runtime_agent: Any) -> Optional[Any]:
        if self.agent_factory is not None and self.token_policy is not None:
            classifier_token = self._select_tool_gate_classifier_token()
            if classifier_token is not None:
                async def _resolver() -> Any:
                    built = self.agent_factory(self.agent_id, classifier_token)
                    if inspect.isawaitable(built):
                        built = await built
                    return built if hasattr(built, "run") else None

                return _resolver
        if hasattr(runtime_agent, "run"):
            return runtime_agent
        return None

    def _build_metadata_short_circuit_decision(
        self,
        *,
        user_message: str,
        recent_history: list[dict[str, Any]],
        available_tools: list[dict[str, Any]],
        metadata_candidates: Optional[dict[str, Any]] = None,
        deps: Optional[SkillDeps] = None,
    ) -> Optional[ToolGateDecision]:
        """Build a direct provider/skill gate decision from metadata when confidence is contextual.

        This path is intentionally constrained to sessions that already have an active
        provider capability selected, so generic questions do not get over-routed.
        """
        if not isinstance(metadata_candidates, dict):
            return None
        provider_skill_classes = self._collect_provider_skill_capability_classes(available_tools)
        if not provider_skill_classes:
            return None

        active_provider_class = self._resolve_active_provider_capability_class(
            deps=deps,
            provider_skill_classes=provider_skill_classes,
        )
        if not active_provider_class:
            return None

        active_provider_type = active_provider_class.split(":", 1)[1].strip().lower()
        preferred_provider_types = [
            str(item).strip().lower()
            for item in (metadata_candidates.get("preferred_provider_types") or [])
            if str(item).strip()
        ]
        preferred_capability_classes = [
            str(item).strip().lower()
            for item in (metadata_candidates.get("preferred_capability_classes") or [])
            if str(item).strip()
        ]
        provider_candidates = [
            item
            for item in (metadata_candidates.get("provider_candidates") or [])
            if isinstance(item, dict)
        ]

        has_active_provider_match = (
            active_provider_type in preferred_provider_types
            or active_provider_class in preferred_capability_classes
            or any(
                str(item.get("provider_type", "") or "").strip().lower() == active_provider_type
                for item in provider_candidates
            )
        )
        if not has_active_provider_match:
            return None

        if not self._looks_provider_or_skill_related(
            user_message=user_message,
            recent_history=recent_history,
            available_tools=available_tools,
            provider_hint_tokens=self._collect_provider_hint_tokens_from_deps(deps),
        ):
            return None

        requested_classes = [
            capability
            for capability in preferred_capability_classes
            if capability == "skill" or capability.startswith("provider:")
        ]
        if active_provider_class not in requested_classes:
            requested_classes.insert(0, active_provider_class)

        selected_classes = self._select_external_system_capability_classes(
            requested_provider_skill_classes=requested_classes,
            provider_skill_classes=provider_skill_classes,
            available_tools=available_tools,
            user_message=user_message,
            recent_history=recent_history,
            preferred_provider_class=active_provider_class,
        )
        if not selected_classes:
            return None
        if active_provider_class not in selected_classes:
            selected_classes = [active_provider_class, *selected_classes]

        metadata_confidence = float(metadata_candidates.get("confidence", 0.0) or 0.0)
        return ToolGateDecision(
            needs_tool=True,
            needs_external_system=True,
            needs_grounded_verification=True,
            suggested_tool_classes=self._dedupe_preserve_order(selected_classes),
            confidence=max(metadata_confidence, self.TOOL_GATE_SHORT_CIRCUIT_MIN_CONFIDENCE),
            reason=(
                "Provider/skill gate short-circuited from metadata recall using "
                "the active provider context."
            ),
            policy=ToolPolicyMode.MUST_USE_TOOL,
        )

    def _select_tool_gate_classifier_token(self) -> Optional[Any]:
        if self.token_policy is None:
            return None
        pool = self.token_policy.token_pool
        ranked: list[tuple[int, int, int, Any]] = []
        for token_id, token in pool.tokens.items():
            health = pool.get_token_health(token_id)
            is_healthy = 1 if (health is None or health.is_healthy) else 0
            ranked.append((is_healthy, int(getattr(token, "priority", 0) or 0), int(getattr(token, "weight", 0) or 0), token))
        if not ranked:
            return None
        ranked.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        return ranked[0][3]

    def _normalize_tool_gate_decision(self, decision: ToolGateDecision) -> ToolGateDecision:
        """Normalize gate output and avoid over-aggressive mandatory-tool enforcement."""
        if not isinstance(decision, ToolGateDecision):
            return ToolGateDecision(
                reason="Tool gate decision is invalid; fallback to direct-answer mode.",
                confidence=0.0,
                policy=ToolPolicyMode.ANSWER_DIRECT,
            )

        normalized = decision.model_copy(deep=True)
        normalized.suggested_tool_classes = [
            item.strip()
            for item in normalized.suggested_tool_classes
            if isinstance(item, str) and item.strip()
        ]

        has_provider_skill_hint = any(
            item == "skill" or item.startswith("provider:")
            for item in normalized.suggested_tool_classes
        )
        strict_web_grounding = bool(normalized.needs_live_data)
        strict_provider_or_skill = bool(normalized.needs_external_system) or has_provider_skill_hint
        strict_tool_enforcement = strict_web_grounding or strict_provider_or_skill

        if strict_provider_or_skill:
            normalized.needs_external_system = True
            normalized.needs_tool = True
            normalized.policy = ToolPolicyMode.MUST_USE_TOOL
            normalized.confidence = max(
                normalized.confidence,
                self.TOOL_GATE_SHORT_CIRCUIT_MIN_CONFIDENCE,
            )
            if "provider/skill intent" not in normalized.reason.lower():
                normalized.reason = (
                    f"{normalized.reason} External-system/provider-skill intent detected from tool metadata."
                ).strip()

        if strict_web_grounding and normalized.policy is ToolPolicyMode.ANSWER_DIRECT:
            normalized.policy = ToolPolicyMode.PREFER_TOOL
            normalized.needs_tool = True
            normalized.confidence = max(
                normalized.confidence,
                self.TOOL_GATE_SHORT_CIRCUIT_MIN_CONFIDENCE,
            )
            normalized.reason = (
                f"{normalized.reason} Live grounded requests require tool-backed verification."
            ).strip()

        has_tool_hints = bool(normalized.suggested_tool_classes)
        strict_need = self._tool_gate_has_strict_need(normalized)
        expects_tool = normalized.needs_tool or has_tool_hints or strict_need

        if normalized.policy is ToolPolicyMode.MUST_USE_TOOL and (
            (not strict_tool_enforcement and normalized.confidence < self.TOOL_GATE_MUST_USE_MIN_CONFIDENCE)
            or not expects_tool
            or not strict_need
        ):
            normalized.policy = (
                ToolPolicyMode.PREFER_TOOL
                if expects_tool
                else ToolPolicyMode.ANSWER_DIRECT
            )
            normalized.reason = (
                f"{normalized.reason} Downgraded from must_use_tool due to insufficient confidence or strict-need signals."
            ).strip()

        if normalized.policy is ToolPolicyMode.ANSWER_DIRECT and expects_tool:
            normalized.policy = ToolPolicyMode.PREFER_TOOL

        return normalized

    def _align_external_system_intent(
        self,
        *,
        decision: ToolGateDecision,
        match_result: CapabilityMatchResult,
        available_tools: list[dict[str, Any]],
        user_message: str,
        recent_history: list[dict[str, Any]],
        deps: Optional[SkillDeps] = None,
    ) -> tuple[ToolGateDecision, CapabilityMatchResult]:
        """Prioritize provider/skill tool classes for external-system requests."""
        if not decision.needs_external_system:
            return decision, match_result

        provider_skill_classes = self._collect_provider_skill_capability_classes(available_tools)
        if not provider_skill_classes:
            return decision, match_result

        requested_provider_skill_classes = [
            capability
            for capability in decision.suggested_tool_classes
            if capability == "skill" or capability.startswith("provider:")
        ]
        selected_classes = self._select_external_system_capability_classes(
            requested_provider_skill_classes=requested_provider_skill_classes,
            provider_skill_classes=provider_skill_classes,
            available_tools=available_tools,
            user_message=user_message,
            recent_history=recent_history,
            preferred_provider_class=self._resolve_active_provider_capability_class(
                deps=deps,
                provider_skill_classes=provider_skill_classes,
            ),
        )

        rewritten = decision.model_copy(deep=True)
        rewritten.needs_tool = True
        rewritten.policy = ToolPolicyMode.MUST_USE_TOOL
        rewritten.confidence = max(rewritten.confidence, self.TOOL_GATE_SHORT_CIRCUIT_MIN_CONFIDENCE)
        rewritten.suggested_tool_classes = selected_classes
        rewritten.reason = (
            f"{rewritten.reason} External-system intent was mapped to provider/skill direct tools."
        ).strip()

        refreshed_match = CapabilityMatcher(available_tools=available_tools).match(
            rewritten.suggested_tool_classes
        )
        return rewritten, refreshed_match

    @staticmethod
    def _collect_provider_skill_capability_classes(available_tools: list[dict[str, Any]]) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()

        for tool in available_tools:
            capability = RunnerToolGateMixin._resolve_provider_skill_capability(tool)

            if not capability:
                continue
            if capability.startswith("provider:") or capability == "skill":
                if capability in seen:
                    continue
                seen.add(capability)
                ordered.append(capability)
        return ordered

    @staticmethod
    def _has_provider_or_skill_candidates(match_result: CapabilityMatchResult) -> bool:
        for candidate in match_result.tool_candidates:
            capability = str(getattr(candidate, "capability_class", "") or "").strip()
            if capability.startswith("provider:") or capability == "skill":
                return True
        return False

    @staticmethod
    def _resolve_provider_skill_capability(tool: dict[str, Any]) -> str:
        capability = str(tool.get("capability_class", "") or "").strip().lower()
        lowered_name = str(tool.get("name", "") or "").strip().lower()
        lowered_description = str(tool.get("description", "") or "").strip().lower()
        provider_type = str(tool.get("provider_type", "") or "").strip().lower()
        category = str(tool.get("category", "") or "").strip().lower()

        if capability.startswith("provider:") or capability == "skill":
            return capability
        if provider_type:
            return f"provider:{provider_type}"
        if "jira" in lowered_name or "jira" in lowered_description:
            return "provider:jira"
        if category.startswith("provider") or "provider:" in lowered_description:
            return "provider:generic"
        if "skill" in category or (
            "skill" in lowered_description and lowered_name not in {"web_search", "web_fetch"}
        ):
            return "skill"
        return ""

    def _select_external_system_capability_classes(
        self,
        *,
        requested_provider_skill_classes: list[str],
        provider_skill_classes: list[str],
        available_tools: list[dict[str, Any]],
        user_message: str,
        recent_history: list[dict[str, Any]],
        preferred_provider_class: Optional[str] = None,
    ) -> list[str]:
        requested = [
            capability
            for capability in requested_provider_skill_classes
            if capability in provider_skill_classes
        ]
        if requested:
            return requested
        if preferred_provider_class and preferred_provider_class in provider_skill_classes:
            return [preferred_provider_class]

        history_text = " ".join(
            str(item.get("content", "") or "").strip()
            for item in recent_history[-4:]
            if isinstance(item, dict)
        )
        request_text = f"{user_message} {history_text}".strip()
        request_text_lower = request_text.lower()
        request_tokens = self._tokenize_classifier_fallback_text(request_text)
        if not request_tokens:
            return provider_skill_classes

        class_scores: dict[str, int] = {}
        for capability in provider_skill_classes:
            class_tokens = self._tokenize_classifier_fallback_text(capability.replace("provider:", ""))
            score = len(request_tokens.intersection(class_tokens))
            if capability.startswith("provider:"):
                provider_key = capability.split(":", 1)[1].strip().lower()
                if provider_key:
                    if provider_key in request_text_lower:
                        score += 6
                    for token in request_tokens:
                        if len(token) < 3:
                            continue
                        if token in provider_key or provider_key in token:
                            score += 2
            for tool in available_tools:
                if self._resolve_provider_skill_capability(tool) != capability:
                    continue
                metadata_text = " ".join(
                    [
                        str(tool.get("name", "") or ""),
                        str(tool.get("description", "") or ""),
                        str(tool.get("provider_type", "") or ""),
                        str(tool.get("category", "") or ""),
                    ]
                ).strip().lower()
                metadata_tokens = self._tokenize_classifier_fallback_text(metadata_text)
                score += len(request_tokens.intersection(metadata_tokens))
                for token in request_tokens:
                    if len(token) < 3:
                        continue
                    if token in metadata_text:
                        score += 1
            class_scores[capability] = score

        if not class_scores:
            return provider_skill_classes
        top_score = max(class_scores.values())
        if top_score <= 0:
            return provider_skill_classes

        selected = [
            capability
            for capability in provider_skill_classes
            if class_scores.get(capability, 0) == top_score
        ]
        return selected or provider_skill_classes

    @staticmethod
    def _resolve_active_provider_capability_class(
        *,
        deps: Optional[SkillDeps],
        provider_skill_classes: list[str],
    ) -> Optional[str]:
        if deps is None or not isinstance(getattr(deps, "extra", None), dict):
            return None
        extra = deps.extra
        provider_type = ""
        provider_instance = extra.get("provider_instance")
        if isinstance(provider_instance, dict):
            provider_type = str(provider_instance.get("provider_type", "") or "").strip().lower()
        if not provider_type:
            provider_type = str(extra.get("provider_type", "") or "").strip().lower()
        if not provider_type:
            provider_type = str(extra.get("provider", "") or "").strip().lower()
        if not provider_type:
            provider_instances = extra.get("provider_instances")
            if isinstance(provider_instances, dict):
                for key in sorted(provider_instances.keys()):
                    capability = f"provider:{str(key).strip().lower()}"
                    if capability in provider_skill_classes:
                        provider_type = str(key).strip().lower()
                        break
        if not provider_type:
            return None
        capability = f"provider:{provider_type}"
        if capability in provider_skill_classes:
            return capability
        return None

    @staticmethod
    def _tool_gate_has_strict_need(decision: ToolGateDecision) -> bool:
        return any(
            [
                bool(decision.needs_live_data),
                bool(decision.needs_grounded_verification),
                bool(decision.needs_external_system),
                bool(decision.needs_browser_interaction),
                bool(decision.needs_private_context),
            ]
        )

    def _resolve_contextual_tool_request(
        self,
        *,
        user_message: str,
        recent_history: list[dict[str, Any]],
    ) -> tuple[str, bool]:
        normalized_user_message = " ".join((user_message or "").split()).strip()
        if not normalized_user_message:
            return user_message, False
        if len(re.sub(r"\s+", "", normalized_user_message)) > 32:
            return normalized_user_message, False

        last_assistant_index: Optional[int] = None
        last_assistant_message = ""
        for index in range(len(recent_history) - 1, -1, -1):
            item = recent_history[index]
            if str(item.get("role", "")).strip() != "assistant":
                continue
            content = " ".join(str(item.get("content", "") or "").split()).strip()
            if not content:
                continue
            last_assistant_index = index
            last_assistant_message = content
            break

        if last_assistant_index is None or not self._looks_like_follow_up_request(last_assistant_message):
            return normalized_user_message, False

        previous_user_message = ""
        for index in range(last_assistant_index - 1, -1, -1):
            item = recent_history[index]
            if str(item.get("role", "")).strip() != "user":
                continue
            content = " ".join(str(item.get("content", "") or "").split()).strip()
            if not content:
                continue
            previous_user_message = content
            break

        if not previous_user_message:
            return normalized_user_message, False

        current_tokens = self._tokenize_classifier_fallback_text(normalized_user_message)
        compact_current_len = len(re.sub(r"\s+", "", normalized_user_message))
        low_information_follow_up = compact_current_len <= 8 or len(current_tokens) <= 1
        if not low_information_follow_up:
            return normalized_user_message, False

        combined = f"{previous_user_message} {normalized_user_message}".strip()
        return combined, combined != normalized_user_message

    @staticmethod
    def _build_classifier_history(
        *,
        recent_history: list[dict[str, Any]],
        used_follow_up_context: bool,
        max_messages: int = 4,
        max_chars_per_message: int = 240,
    ) -> list[dict[str, Any]]:
        """Build a compact history slice for gate classification.

        The classifier should always receive a small amount of session context so
        follow-up requests (for example "show details for this ticket") can stay
        on the same provider path without shipping the full transcript.
        """
        if not isinstance(recent_history, list) or not recent_history:
            return []
        tail_count = max(2, int(max_messages or 4))
        char_limit = max(80, int(max_chars_per_message or 240))
        selected = recent_history[-tail_count:]

        compact: list[dict[str, Any]] = []
        for item in selected:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "") or "").strip()
            if role not in {"user", "assistant"}:
                continue
            content = " ".join(str(item.get("content", "") or "").split()).strip()
            if not content:
                continue
            if len(content) > char_limit:
                content = content[:char_limit].rstrip() + " ..."
            compact.append({"role": role, "content": content})

        if used_follow_up_context:
            return compact
        # Even when not an explicit follow-up, keep minimal context to reduce
        # false answer_direct classification on short continuation turns.
        return compact

    def _apply_no_classifier_follow_up_fallback(
        self,
        *,
        decision: ToolGateDecision,
        used_follow_up_context: bool,
        available_tools: list[dict[str, Any]],
    ) -> ToolGateDecision:
        # Keep follow-up turns on the same LLM-driven gate path.
        # Do not inject runtime web defaults here.
        return decision

    def _apply_provider_skill_intent_fallback(
        self,
        *,
        decision: ToolGateDecision,
        user_message: str,
        recent_history: list[dict[str, Any]],
        available_tools: list[dict[str, Any]],
        deps: Optional[SkillDeps] = None,
    ) -> ToolGateDecision:
        if decision.needs_external_system:
            return decision
        metadata_confidence = 0.0
        has_metadata_provider_candidates = False
        if deps is not None and isinstance(getattr(deps, "extra", None), dict):
            metadata = deps.extra.get("tool_metadata_candidates")
            if isinstance(metadata, dict):
                metadata_confidence = float(metadata.get("confidence", 0.0) or 0.0)
                has_metadata_provider_candidates = bool(
                    (metadata.get("preferred_provider_types") or [])
                    or (metadata.get("preferred_capability_classes") or [])
                    or (metadata.get("preferred_tool_names") or [])
                )
        has_contextual_provider_signal = bool(recent_history) and has_metadata_provider_candidates
        if (
            metadata_confidence < self.TOOL_GATE_SHORT_CIRCUIT_MIN_CONFIDENCE
            and not has_contextual_provider_signal
        ):
            return decision
        provider_skill_classes = self._collect_provider_skill_capability_classes(available_tools)
        if not provider_skill_classes:
            return decision
        if not self._looks_provider_or_skill_related(
            user_message=user_message,
            recent_history=recent_history,
            available_tools=available_tools,
            provider_hint_tokens=self._collect_provider_hint_tokens_from_deps(deps),
        ):
            return decision

        rewritten = decision.model_copy(deep=True)
        rewritten.needs_tool = True
        rewritten.needs_external_system = True
        rewritten.policy = ToolPolicyMode.MUST_USE_TOOL
        rewritten.confidence = max(
            rewritten.confidence,
            self.TOOL_GATE_SHORT_CIRCUIT_MIN_CONFIDENCE,
        )
        requested_provider_skill_classes = [
            item
            for item in rewritten.suggested_tool_classes
            if item == "skill" or item.startswith("provider:")
        ]
        rewritten.suggested_tool_classes = self._select_external_system_capability_classes(
            requested_provider_skill_classes=requested_provider_skill_classes,
            provider_skill_classes=provider_skill_classes,
            available_tools=available_tools,
            user_message=user_message,
            recent_history=recent_history,
            preferred_provider_class=self._resolve_active_provider_capability_class(
                deps=deps,
                provider_skill_classes=provider_skill_classes,
            ),
        )
        rewritten.reason = (
            f"{rewritten.reason} Runtime mapped request to provider/skill intent using tool metadata."
        ).strip()
        return rewritten

    def _apply_tool_gate_consistency_guard(
        self,
        *,
        decision: ToolGateDecision,
        user_message: str,
        recent_history: list[dict[str, Any]],
        available_tools: list[dict[str, Any]],
        deps: Optional[SkillDeps] = None,
        metadata_candidates: Optional[dict[str, Any]] = None,
    ) -> ToolGateDecision:
        has_provider_skill_class = any(
            str(item).strip() == "skill" or str(item).strip().startswith("provider:")
            for item in decision.suggested_tool_classes
        )
        if not decision.needs_external_system and not has_provider_skill_class:
            return decision

        metadata_confidence = 0.0
        if isinstance(metadata_candidates, dict):
            metadata_confidence = float(metadata_candidates.get("confidence", 0.0) or 0.0)
        meaningful_provider_overlap = self._has_meaningful_provider_overlap(
            decision=decision,
            user_message=user_message,
            recent_history=recent_history,
            available_tools=available_tools,
        )
        provider_hint_tokens = self._collect_provider_hint_tokens_from_deps(deps)
        looks_related = self._looks_provider_or_skill_related(
            user_message=user_message,
            recent_history=recent_history,
            available_tools=available_tools,
            provider_hint_tokens=provider_hint_tokens,
        )
        if (
            meaningful_provider_overlap
            or looks_related
            or metadata_confidence >= self.TOOL_GATE_SHORT_CIRCUIT_MIN_CONFIDENCE
        ):
            return decision

        rewritten = decision.model_copy(deep=True)
        rewritten.needs_external_system = False
        rewritten.suggested_tool_classes = [
            item
            for item in rewritten.suggested_tool_classes
            if not (item == "skill" or item.startswith("provider:"))
        ]
        if rewritten.needs_live_data or rewritten.needs_grounded_verification:
            rewritten.needs_tool = True
            if rewritten.policy is ToolPolicyMode.ANSWER_DIRECT:
                rewritten.policy = ToolPolicyMode.PREFER_TOOL
        elif not rewritten.suggested_tool_classes:
            rewritten.needs_tool = False
            rewritten.policy = ToolPolicyMode.ANSWER_DIRECT
        rewritten.confidence = min(float(rewritten.confidence), 0.49)
        rewritten.reason = (
            f"{rewritten.reason} Consistency guard removed unsupported provider/skill routing "
            "because request-to-provider relevance was too weak."
        ).strip()
        return rewritten

    def _has_meaningful_provider_overlap(
        self,
        *,
        decision: ToolGateDecision,
        user_message: str,
        recent_history: list[dict[str, Any]],
        available_tools: list[dict[str, Any]],
    ) -> bool:
        request_text = " ".join(
            [
                str(user_message or "").strip(),
                " ".join(
                    str(item.get("content", "") or "").strip()
                    for item in recent_history[-4:]
                    if isinstance(item, dict)
                ),
            ]
        ).strip()
        request_tokens = self._tokenize_classifier_fallback_text(request_text)
        if not request_tokens:
            return False

        targeted_classes = {
            item
            for item in decision.suggested_tool_classes
            if item == "skill" or item.startswith("provider:")
        }
        provider_tools: list[dict[str, Any]] = []
        for tool in available_tools:
            capability = self._resolve_provider_skill_capability(tool)
            if not (capability == "skill" or capability.startswith("provider:")):
                continue
            if targeted_classes and capability not in targeted_classes:
                continue
            provider_tools.append(tool)
        if not provider_tools:
            return False

        score = 0
        for tool in provider_tools:
            token_bag = " ".join(
                [
                    str(tool.get("name", "") or ""),
                    str(tool.get("description", "") or ""),
                    str(tool.get("provider_type", "") or ""),
                    str(tool.get("category", "") or ""),
                    str(tool.get("capability_class", "") or ""),
                ]
            ).strip()
            tool_tokens = self._tokenize_classifier_fallback_text(token_bag)
            overlap = request_tokens.intersection(tool_tokens)
            if not overlap:
                continue
            score += sum(2 if len(token) >= 3 else 1 for token in overlap)
            if score >= 3:
                return True
        return False

    def _inject_tool_policy(
        self,
        *,
        deps: SkillDeps,
        decision: ToolGateDecision,
        match_result: CapabilityMatchResult,
    ) -> None:
        """Inject per-run tool-policy context for prompt building."""
        if not isinstance(deps.extra, dict):
            deps.extra = {}
        retry_count = int(deps.extra.get("_tool_policy_retry_count", 0) or 0)
        retry_missing_tools = deps.extra.get("tool_policy_retry_missing_tools")
        if not isinstance(retry_missing_tools, list):
            retry_missing_tools = []

        required_tools = self._required_tool_names_for_decision(
            decision=decision,
            match_result=match_result,
        )
        preferred_tools = self._preferred_tool_names_for_prompt(
            decision=decision,
            match_result=match_result,
            required_tools=required_tools,
        )

        provider_skill_fast_path = bool(
            decision.needs_external_system and self._has_provider_or_skill_candidates(match_result)
        )
        top_tool_hints: list[str] = []
        if isinstance(getattr(deps, "extra", None), dict):
            ranking_trace = deps.extra.get("tool_ranking_trace")
            if isinstance(ranking_trace, dict):
                hints = ranking_trace.get("top_tool_hints")
                if isinstance(hints, list):
                    top_tool_hints = [
                        str(item).strip()
                        for item in hints
                        if str(item).strip()
                    ][: max(1, int(getattr(self, "TOOL_HINT_TOP_K", 3) or 3))]

        deps.extra["tool_policy"] = {
            "mode": decision.policy.value,
            "reason": decision.reason,
            "required_tools": preferred_tools,
            "missing_capabilities": list(match_result.missing_capabilities),
            "confidence": float(decision.confidence),
            "execution_hint": "provider_tool_first" if provider_skill_fast_path else "default",
            "retry_count": retry_count,
            "retry_missing_tools": [
                str(name).strip() for name in retry_missing_tools if str(name).strip()
            ],
            "top_tool_hints": top_tool_hints,
        }

    @staticmethod
    def _preferred_tool_names_for_prompt(
        *,
        decision: ToolGateDecision,
        match_result: CapabilityMatchResult,
        required_tools: list[str],
    ) -> list[str]:
        if decision.needs_external_system:
            provider_skill_candidates: list[str] = []
            for candidate in match_result.tool_candidates:
                capability = str(getattr(candidate, "capability_class", "") or "").strip()
                name = str(getattr(candidate, "name", "") or "").strip()
                if not name:
                    continue
                if not (capability.startswith("provider:") or capability == "skill"):
                    continue
                if name in provider_skill_candidates:
                    continue
                provider_skill_candidates.append(name)
                if len(provider_skill_candidates) >= 5:
                    break
            if provider_skill_candidates:
                return provider_skill_candidates
        return list(required_tools)

    @staticmethod
    def _build_missing_capability_message(match_result: CapabilityMatchResult) -> str:
        missing = [item for item in match_result.missing_capabilities if item]
        if missing:
            joined = ", ".join(sorted(set(missing)))
            return (
                "Verification requires tools that are not available. Missing capabilities: "
                f"{joined}. Please enable the corresponding tools and retry."
            )
        return (
            "Verification requires tools that are not available. "
            "Please enable the required tool and retry."
        )

    @staticmethod
    def _collect_buffered_assistant_text(buffered_events: list[StreamEvent]) -> str:
        chunks: list[str] = []
        for event in buffered_events:
            if event.type != "assistant":
                continue
            content = str(getattr(event, "content", "") or "")
            if content:
                chunks.append(content)
        return "".join(chunks).strip()

    @staticmethod
    def _called_tool_names(tool_call_summaries: list[dict[str, Any]]) -> set[str]:
        called: set[str] = set()
        for item in tool_call_summaries:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "") or "").strip()
            if name:
                called.add(name)
        return called

    @staticmethod
    def _required_tool_names_for_decision(
        *,
        decision: ToolGateDecision,
        match_result: CapabilityMatchResult,
    ) -> list[str]:
        required: list[str] = []
        if decision.needs_external_system:
            required_by_capability: dict[str, str] = {}
            for candidate in match_result.tool_candidates:
                capability = str(getattr(candidate, "capability_class", "") or "").strip()
                name = str(getattr(candidate, "name", "") or "").strip()
                if not name:
                    continue
                if not (capability.startswith("provider:") or capability == "skill"):
                    continue
                if capability in required_by_capability:
                    continue
                required_by_capability[capability] = name
            if required_by_capability:
                return list(required_by_capability.values())

        for candidate in match_result.tool_candidates:
            capability = str(getattr(candidate, "capability_class", "") or "").strip()
            name = str(getattr(candidate, "name", "") or "").strip()
            if not name:
                continue
            if decision.needs_live_data and decision.needs_grounded_verification:
                if capability in {"weather", "web_search", "web_fetch", "browser"}:
                    required.append(name)
                continue
            required.append(name)

        deduped: list[str] = []
        seen: set[str] = set()
        for name in required:
            if name in seen:
                continue
            seen.add(name)
            deduped.append(name)
        return deduped

    def _missing_required_tool_names(
        self,
        *,
        decision: ToolGateDecision,
        match_result: CapabilityMatchResult,
        tool_call_summaries: list[dict[str, Any]],
        available_tools: Optional[list[dict[str, Any]]] = None,
        final_messages: Optional[list[dict[str, Any]]] = None,
        run_output_start_index: int = 0,
    ) -> list[str]:
        required = self._required_tool_names_for_decision(
            decision=decision,
            match_result=match_result,
        )
        if not required:
            return []
        called = self._called_tool_names(tool_call_summaries)
        if final_messages:
            called.update(
                self._collect_called_tool_names_from_messages(
                    messages=final_messages,
                    start_index=run_output_start_index,
                )
            )
        missing = [name for name in required if name not in called]
        if not missing:
            return []
        if not called:
            return missing

        capability_map = self._build_tool_capability_map(available_tools or [])
        called_capabilities = {
            self._resolve_tool_capability(name=name, capability_map=capability_map)
            for name in called
        }
        required_capabilities = {
            self._resolve_tool_capability(name=name, capability_map=capability_map)
            for name in required
        }

        if self._called_capabilities_satisfy_required(
            decision=decision,
            called_capabilities=called_capabilities,
            required_capabilities=required_capabilities,
        ):
            missing = []

        if final_messages:
            successful_tools = self._collect_successful_tool_names(
                messages=final_messages,
                start_index=run_output_start_index,
            )
            if required:
                missing_success = [name for name in required if name not in successful_tools]
                if missing_success:
                    successful_capabilities = {
                        self._resolve_tool_capability(name=name, capability_map=capability_map)
                        for name in successful_tools
                    }
                    if not self._called_capabilities_satisfy_required(
                        decision=decision,
                        called_capabilities=successful_capabilities,
                        required_capabilities=required_capabilities,
                    ):
                        return missing_success

        return missing

    def _collect_called_tool_names_from_messages(
        self,
        *,
        messages: list[dict[str, Any]],
        start_index: int = 0,
    ) -> set[str]:
        names: set[str] = set()
        safe_start = max(0, min(int(start_index), len(messages)))
        for message in messages[safe_start:]:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "")).strip().lower()
            if role in {"tool", "toolresult", "tool_result"}:
                tool_name = str(
                    message.get("tool_name", "") or message.get("name", "")
                ).strip()
                if tool_name:
                    names.add(tool_name)
                continue
            if role != "assistant":
                continue
            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                tool_name = str(
                    call.get("name", "") or call.get("tool_name", "")
                ).strip()
                if tool_name:
                    names.add(tool_name)
        return names

    def _should_enforce_web_tool_verification(
        self,
        *,
        decision: ToolGateDecision,
        match_result: CapabilityMatchResult,
        available_tools: list[dict[str, Any]],
    ) -> bool:
        """Return whether hard/soft tool verification should be enforced for this run.

        Enforcement stays on the unified model tool loop for all tool classes.
        """
        if decision.policy not in {ToolPolicyMode.MUST_USE_TOOL, ToolPolicyMode.PREFER_TOOL}:
            return False
        strict_need = self._tool_gate_has_strict_need(decision)
        required_tools = self._required_tool_names_for_decision(
            decision=decision,
            match_result=match_result,
        )
        if decision.policy is ToolPolicyMode.MUST_USE_TOOL and required_tools:
            return True
        if decision.policy is ToolPolicyMode.PREFER_TOOL:
            return strict_need and bool(required_tools)

        suggested = [
            str(item or "").strip()
            for item in decision.suggested_tool_classes
            if isinstance(item, str) and str(item or "").strip()
        ]
        if decision.policy is ToolPolicyMode.MUST_USE_TOOL and suggested:
            return True
        if decision.policy is ToolPolicyMode.PREFER_TOOL:
            return strict_need and bool(suggested)

        if decision.policy is ToolPolicyMode.MUST_USE_TOOL:
            return bool(
                decision.needs_tool
                or decision.needs_external_system
                or decision.needs_live_data
                or decision.needs_grounded_verification
            )
        return strict_need and bool(decision.needs_tool)

    def _collect_successful_tool_names(
        self,
        *,
        messages: list[dict[str, Any]],
        start_index: int = 0,
    ) -> set[str]:
        successful: set[str] = set()
        safe_start = max(0, min(int(start_index), len(messages)))
        for message in messages[safe_start:]:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "")).strip().lower()
            if role not in {"tool", "toolresult", "tool_result"}:
                continue
            tool_name = str(message.get("tool_name", "") or message.get("name", "")).strip()
            if not tool_name:
                continue
            if self._is_tool_result_success(message):
                successful.add(tool_name)
        return successful

    @staticmethod
    def _is_tool_result_success(message: dict[str, Any]) -> bool:
        if bool(message.get("is_error")):
            return False
        content = message.get("content")
        if isinstance(content, str):
            payload = content.strip()
            if not payload:
                return False
            if payload.startswith("{") or payload.startswith("["):
                try:
                    parsed = json.loads(payload)
                except Exception:
                    return True
                return RunnerToolGateMixin._is_tool_payload_success(parsed)
            return True
        return RunnerToolGateMixin._is_tool_payload_success(content)

    @staticmethod
    def _is_tool_payload_success(payload: Any) -> bool:
        if payload is None:
            return False
        if isinstance(payload, dict):
            if bool(payload.get("is_error")):
                return False
            error_value = payload.get("error")
            if isinstance(error_value, str) and error_value.strip():
                return False
            if isinstance(error_value, dict) and error_value:
                return False
            if isinstance(error_value, list) and error_value:
                return False
            if "content" in payload:
                return RunnerToolGateMixin._is_tool_payload_success(payload.get("content"))
            if "details" in payload:
                return True
            if "results" in payload:
                return RunnerToolGateMixin._is_tool_payload_success(payload.get("results"))
            if "data" in payload:
                return RunnerToolGateMixin._is_tool_payload_success(payload.get("data"))
            if "summary" in payload and str(payload.get("summary", "")).strip():
                return True
            if "text" in payload and str(payload.get("text", "")).strip():
                return True
            return bool(payload)
        if isinstance(payload, list):
            if not payload:
                return False
            for item in payload:
                if RunnerToolGateMixin._is_tool_payload_success(item):
                    return True
            return False
        if isinstance(payload, (int, float, bool)):
            return True
        if isinstance(payload, str):
            return bool(payload.strip())
        return bool(payload)

    @staticmethod
    def _build_tool_capability_map(available_tools: list[dict[str, Any]]) -> dict[str, str]:
        capability_map: dict[str, str] = {}
        for tool in available_tools:
            if not isinstance(tool, dict):
                continue
            name = str(tool.get("name", "") or "").strip()
            if not name:
                continue
            capability = str(tool.get("capability_class", "") or "").strip()
            if capability:
                capability_map[name] = capability
        return capability_map

    def _resolve_tool_capability(self, *, name: str, capability_map: dict[str, str]) -> str:
        direct = capability_map.get(name, "")
        if direct:
            return direct
        lowered = str(name or "").strip().lower()
        if lowered in {"web_search", "web_fetch", "browser", "openmeteo_weather"}:
            return {
                "openmeteo_weather": "weather",
            }.get(lowered, lowered)
        if lowered.startswith("provider_") or lowered.startswith("smartcmp") or lowered.startswith("jira"):
            if "jira" in lowered:
                return "provider:jira"
            return "provider:generic"
        return ""

    @staticmethod
    def _called_capabilities_satisfy_required(
        *,
        decision: ToolGateDecision,
        called_capabilities: set[str],
        required_capabilities: set[str],
    ) -> bool:
        normalized_called = {item for item in called_capabilities if item}
        normalized_required = {item for item in required_capabilities if item}
        if not normalized_called:
            return False

        provider_skill_called = any(
            capability.startswith("provider:") or capability == "skill"
            for capability in normalized_called
        )
        web_required_only = bool(normalized_required) and all(
            capability in {"web_search", "web_fetch", "browser", "weather"}
            for capability in normalized_required
        )

        if decision.needs_external_system and provider_skill_called:
            return True
        if normalized_required.intersection(normalized_called):
            return True
        if provider_skill_called and web_required_only:
            # Guard against classifier mismatch: when provider/skill tools actually ran,
            # do not force an unrelated web-search requirement.
            return True
        return False

    @staticmethod
    def _build_tool_evidence_required_message(
        *,
        match_result: CapabilityMatchResult,
        missing_required_tools: list[str],
    ) -> str:
        candidate_names = []
        for candidate in match_result.tool_candidates:
            name = str(getattr(candidate, "name", "") or "").strip()
            if name and name not in candidate_names:
                candidate_names.append(name)
        if missing_required_tools:
            display_missing = list(missing_required_tools)
            if len(display_missing) > 5:
                extra_count = len(display_missing) - 5
                display_missing = [*display_missing[:5], f"...({extra_count} more)"]
            return (
                "A grounded tool-backed answer is required for this request, but required tools were not executed: "
                f"{', '.join(display_missing)}."
            )
        if candidate_names:
            return (
                "A grounded tool-backed answer is required for this request, but no usable tool "
                f"evidence was produced in this run. Required tools: {', '.join(candidate_names)}."
            )
        return (
            "A grounded tool-backed answer is required for this request, but no usable tool "
            "evidence was produced in this run."
        )

    @staticmethod
    def _looks_like_follow_up_request(message: str) -> bool:
        text = " ".join((message or "").split())
        if not text:
            return False
        lowered = text.lower()
        question_count = text.count("?") + text.count("？")
        numbered_choices = len(re.findall(r"(?:^|[\s\n])(?:1[\)\.]|2[\)\.]|3[\)\.])", text))
        interaction_markers = (
            "please reply",
            "reply with",
            "choose",
            "confirm",
            "clarify",
            "specify",
            "select",
            "tell me",
            "provide",
            "\u8bf7\u56de\u590d",
            "\u56de\u590d\u6211",
            "\u8bf7\u786e\u8ba4",
            "\u786e\u8ba4\u4e00\u4e0b",
            "\u8865\u5145",
            "\u544a\u8bc9\u6211",
            "\u9009\u62e9",
            "\u6307\u5b9a",
            "\u9009\u9879",
            "\u4efb\u9009",
        )
        marker_hits = sum(1 for marker in interaction_markers if marker in lowered or marker in text)
        if numbered_choices >= 2 and marker_hits >= 1:
            return True
        if question_count >= 2 and marker_hits >= 1:
            return True
        if question_count >= 1 and marker_hits >= 2:
            return True
        return False

    async def _classify_tool_gate_with_model(
        self,
        *,
        agent: Any,
        deps: SkillDeps,
        user_message: str,
        recent_history: list[dict[str, Any]],
        available_tools: list[dict[str, Any]],
    ) -> Optional[ToolGateDecision]:
        classifier_prompt = self._build_tool_gate_classifier_prompt(available_tools)
        metadata_candidates: dict[str, Any] = {}
        if isinstance(getattr(deps, "extra", None), dict):
            raw_candidates = deps.extra.get("tool_metadata_candidates")
            if isinstance(raw_candidates, dict):
                metadata_candidates = dict(raw_candidates)
        classifier_message = self._build_tool_gate_classifier_message(
            user_message=user_message,
            recent_history=recent_history,
            metadata_candidates=metadata_candidates,
        )
        try:
            raw_output = await asyncio.wait_for(
                self._run_single_with_optional_override(
                    agent=agent,
                    user_message=classifier_message,
                    deps=deps,
                    system_prompt=classifier_prompt,
                ),
                timeout=self.TOOL_GATE_CLASSIFIER_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            return self._build_classifier_timeout_fallback_decision(
                deps=deps,
                user_message=user_message,
                recent_history=recent_history,
                available_tools=available_tools,
            )
        parsed = self._extract_json_object(raw_output)
        if not parsed:
            return None
        try:
            payload = json.loads(parsed)
            if not isinstance(payload, dict):
                return None
            coerced = self._coerce_tool_gate_payload(payload)
            return ToolGateDecision.model_validate(coerced)
        except Exception:
            return None

    def _build_classifier_timeout_fallback_decision(
        self,
        *,
        deps: SkillDeps,
        user_message: str,
        recent_history: list[dict[str, Any]],
        available_tools: list[dict[str, Any]],
    ) -> Optional[ToolGateDecision]:
        provider_skill_classes = self._collect_provider_skill_capability_classes(available_tools)
        if not provider_skill_classes:
            return None
        if not self._looks_provider_or_skill_related(
            user_message=user_message,
            recent_history=recent_history,
            available_tools=available_tools,
            provider_hint_tokens=self._collect_provider_hint_tokens_from_deps(deps),
        ):
            return None
        selected_classes = self._select_external_system_capability_classes(
            requested_provider_skill_classes=[],
            provider_skill_classes=provider_skill_classes,
            available_tools=available_tools,
            user_message=user_message,
            recent_history=recent_history,
            preferred_provider_class=self._resolve_active_provider_capability_class(
                deps=deps,
                provider_skill_classes=provider_skill_classes,
            ),
        )
        return ToolGateDecision(
            needs_tool=True,
            needs_external_system=True,
            needs_grounded_verification=True,
            suggested_tool_classes=selected_classes,
            confidence=max(self.TOOL_GATE_SHORT_CIRCUIT_MIN_CONFIDENCE, 0.7),
            reason=(
                "Tool-gate classifier timed out; runtime routed to provider/skill fast-path "
                "using available provider capability metadata."
            ),
            policy=ToolPolicyMode.PREFER_TOOL,
        )

    @staticmethod
    def _looks_provider_or_skill_related(
        *,
        user_message: str,
        recent_history: list[dict[str, Any]],
        available_tools: list[dict[str, Any]],
        provider_hint_tokens: Optional[set[str]] = None,
    ) -> bool:
        history_text = " ".join(
            str(item.get("content", "") or "").strip()
            for item in recent_history[-4:]
            if isinstance(item, dict)
        )
        request_text = f"{user_message} {history_text}".strip()
        request_tokens = RunnerToolGateMixin._tokenize_classifier_fallback_text(request_text)
        if not request_tokens:
            return False

        metadata_tokens: set[str] = set()
        metadata_blob_parts: list[str] = []
        for tool in available_tools:
            if not isinstance(tool, dict):
                continue
            capability = str(tool.get("capability_class", "") or "").strip()
            if not (capability.startswith("provider:") or capability == "skill"):
                continue
            metadata_text = " ".join(
                [
                    str(tool.get("name", "") or ""),
                    str(tool.get("description", "") or ""),
                    str(tool.get("provider_type", "") or ""),
                    str(tool.get("category", "") or ""),
                    capability,
                ]
            ).strip()
            metadata_tokens.update(
                RunnerToolGateMixin._tokenize_classifier_fallback_text(metadata_text)
            )
            if metadata_text:
                metadata_blob_parts.append(metadata_text.lower())

        if provider_hint_tokens:
            metadata_tokens.update(provider_hint_tokens)
            metadata_blob_parts.extend(str(token).lower() for token in provider_hint_tokens if token)

        if not metadata_tokens:
            return False
        if request_tokens.intersection(metadata_tokens):
            return True
        metadata_blob = " ".join(metadata_blob_parts)
        if not metadata_blob:
            return False
        for token in request_tokens:
            if len(token) < 3:
                continue
            if token in metadata_blob:
                return True
        return False

    @staticmethod
    def _tokenize_classifier_fallback_text(text: str) -> set[str]:
        normalized = " ".join((text or "").split()).strip().lower()
        if not normalized:
            return set()
        tokens: set[str] = set()
        for token in re.findall(r"[a-z0-9_:-]{2,}", normalized):
            tokens.add(token)
        for chunk in re.findall(r"[\u4e00-\u9fff]{2,}", normalized):
            chunk = chunk.strip()
            if not chunk:
                continue
            tokens.add(chunk)
            if len(chunk) <= 2:
                continue
            # Add CJK bigrams for robust overlap checks in mixed/long Chinese queries.
            for idx in range(0, len(chunk) - 1):
                tokens.add(chunk[idx : idx + 2])
        return tokens

    def _collect_provider_hint_tokens_from_deps(self, deps: Optional[SkillDeps]) -> set[str]:
        contexts = self._collect_provider_contexts_from_deps(deps)
        tokens: set[str] = set()
        for provider_type, ctx in contexts.items():
            tokens.update(self._tokenize_classifier_fallback_text(str(provider_type or "")))
            parts = [
                str(ctx.get("display_name", "") or ""),
                str(ctx.get("description", "") or ""),
                " ".join(str(item) for item in (ctx.get("keywords", []) or [])),
                " ".join(str(item) for item in (ctx.get("capabilities", []) or [])),
                " ".join(str(item) for item in (ctx.get("use_when", []) or [])),
                " ".join(str(item) for item in (ctx.get("avoid_when", []) or [])),
            ]
            for part in parts:
                tokens.update(self._tokenize_classifier_fallback_text(part))
        return tokens

    @staticmethod
    def _collect_provider_contexts_from_deps(deps: Optional[SkillDeps]) -> dict[str, dict[str, Any]]:
        if deps is None or not isinstance(getattr(deps, "extra", None), dict):
            return {}
        extra = deps.extra
        registry = extra.get("_service_provider_registry")
        if registry is None:
            return {}
        get_contexts = getattr(registry, "get_all_provider_contexts", None)
        if not callable(get_contexts):
            return {}
        try:
            contexts = get_contexts()
        except Exception:
            return {}
        if not isinstance(contexts, dict):
            return {}

        normalized: dict[str, dict[str, Any]] = {}
        for provider_type, ctx in contexts.items():
            provider_key = str(provider_type or "").strip().lower()
            if not provider_key:
                continue
            if hasattr(ctx, "__dict__"):
                payload = {
                    "display_name": str(getattr(ctx, "display_name", "") or ""),
                    "description": str(getattr(ctx, "description", "") or ""),
                    "keywords": list(getattr(ctx, "keywords", []) or []),
                    "capabilities": list(getattr(ctx, "capabilities", []) or []),
                    "use_when": list(getattr(ctx, "use_when", []) or []),
                    "avoid_when": list(getattr(ctx, "avoid_when", []) or []),
                }
            elif isinstance(ctx, dict):
                payload = {
                    "display_name": str(ctx.get("display_name", "") or ""),
                    "description": str(ctx.get("description", "") or ""),
                    "keywords": list(ctx.get("keywords", []) or []),
                    "capabilities": list(ctx.get("capabilities", []) or []),
                    "use_when": list(ctx.get("use_when", []) or []),
                    "avoid_when": list(ctx.get("avoid_when", []) or []),
                }
            else:
                payload = {}
            normalized[provider_key] = payload
        return normalized

    def _build_provider_hint_docs(
        self,
        *,
        deps: Optional[SkillDeps],
        available_tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        contexts = self._collect_provider_contexts_from_deps(deps)
        if not contexts:
            return []

        docs: list[dict[str, Any]] = []
        for provider_type in sorted(contexts.keys()):
            ctx = contexts.get(provider_type, {})
            matched_tools = [
                tool
                for tool in available_tools
                if str(tool.get("provider_type", "") or "").strip().lower() == provider_type
                or str(tool.get("capability_class", "") or "").strip().lower() == f"provider:{provider_type}"
            ]
            if not matched_tools:
                continue

            tool_names = sorted(
                {
                    str(tool.get("name", "") or "").strip()
                    for tool in matched_tools
                    if str(tool.get("name", "") or "").strip()
                }
            )
            capability_classes = sorted(
                {
                    str(tool.get("capability_class", "") or "").strip()
                    for tool in matched_tools
                    if str(tool.get("capability_class", "") or "").strip()
                }
            )
            group_ids = sorted(
                {
                    str(group_id or "").strip()
                    for tool in matched_tools
                    for group_id in (tool.get("group_ids", []) or [])
                    if str(group_id or "").strip()
                }
            )
            provider_group = f"group:{provider_type}"
            if provider_group not in group_ids:
                group_ids.append(provider_group)

            priority_values = []
            for tool in matched_tools:
                try:
                    priority_values.append(int(tool.get("priority", 100) or 100))
                except (TypeError, ValueError):
                    continue
            priority = max(priority_values) if priority_values else 100

            hint_text = self._build_hint_text(
                display_name=str(ctx.get("display_name", "") or provider_type),
                description=str(ctx.get("description", "") or ""),
                keywords=ctx.get("keywords", []),
                capabilities=ctx.get("capabilities", []),
                use_when=ctx.get("use_when", []),
                avoid_when=ctx.get("avoid_when", []),
            )
            docs.append(
                {
                    "hint_id": f"provider:{provider_type}",
                    "hint_type": "provider",
                    "provider_type": provider_type,
                    "tool_names": tool_names,
                    "group_ids": group_ids,
                    "capability_classes": capability_classes,
                    "hint_text": hint_text,
                    "priority": priority,
                }
            )
        return docs

    def _build_skill_hint_docs(
        self,
        *,
        deps: Optional[SkillDeps],
        available_tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if deps is None or not isinstance(getattr(deps, "extra", None), dict):
            return []
        md_skills = deps.extra.get("md_skills_snapshot")
        if not isinstance(md_skills, list):
            return []

        available_tool_by_name: dict[str, dict[str, Any]] = {}
        for tool in available_tools:
            name = str(tool.get("name", "") or "").strip()
            if not name:
                continue
            available_tool_by_name[name] = tool

        docs: list[dict[str, Any]] = []
        for entry in md_skills:
            if not isinstance(entry, dict):
                continue
            metadata = entry.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}

            declared_names = self._extract_md_declared_tool_names(metadata)
            matched_names = [
                name
                for name in declared_names
                if name in available_tool_by_name
            ]
            if not matched_names:
                continue
            matched_tools = [available_tool_by_name[name] for name in matched_names]
            provider_type = str(
                metadata.get("provider_type", "")
                or entry.get("provider", "")
                or ""
            ).strip().lower()
            capability_classes = sorted(
                {
                    str(tool.get("capability_class", "") or "").strip()
                    for tool in matched_tools
                    if str(tool.get("capability_class", "") or "").strip()
                }
            )
            group_ids = sorted(
                {
                    str(group_id or "").strip()
                    for tool in matched_tools
                    for group_id in (tool.get("group_ids", []) or [])
                    if str(group_id or "").strip()
                }
            )
            priority_values = []
            for tool in matched_tools:
                try:
                    priority_values.append(int(tool.get("priority", 100) or 100))
                except (TypeError, ValueError):
                    continue
            priority = max(priority_values) if priority_values else 100
            qualified_name = str(entry.get("qualified_name", "") or "").strip()
            skill_name = str(entry.get("name", "") or "").strip() or qualified_name or "skill"

            hint_text = self._build_hint_text(
                display_name=skill_name,
                description=str(entry.get("description", "") or ""),
                keywords=metadata.get("triggers", []),
                capabilities=metadata.get("examples", []),
                use_when=metadata.get("use_when", []),
                avoid_when=metadata.get("avoid_when", []),
            )
            docs.append(
                {
                    "hint_id": f"skill:{qualified_name or skill_name}",
                    "hint_type": "skill",
                    "provider_type": provider_type,
                    "tool_names": sorted(matched_names),
                    "group_ids": group_ids,
                    "capability_classes": capability_classes,
                    "hint_text": hint_text,
                    "priority": priority,
                }
            )
        return docs

    @staticmethod
    def _extract_md_declared_tool_names(metadata: dict[str, Any]) -> list[str]:
        names: list[str] = []
        single_name = str(metadata.get("tool_name", "") or "").strip()
        if single_name:
            names.append(single_name)
        for key, value in metadata.items():
            key_text = str(key or "").strip()
            if not key_text.startswith("tool_") or not key_text.endswith("_name"):
                continue
            tool_name = str(value or "").strip()
            if tool_name:
                names.append(tool_name)
        deduped: list[str] = []
        seen: set[str] = set()
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            deduped.append(name)
        return deduped

    @staticmethod
    def _build_hint_text(
        *,
        display_name: str,
        description: str,
        keywords: list[Any],
        capabilities: list[Any],
        use_when: list[Any],
        avoid_when: list[Any],
    ) -> str:
        blocks: list[str] = []
        title = str(display_name or "").strip()
        if title:
            blocks.append(f"name: {title}")
        desc = str(description or "").strip()
        if desc:
            blocks.append(f"description: {desc}")

        def _append_list(label: str, values: list[Any]) -> None:
            normalized = [str(item).strip() for item in values if str(item).strip()]
            if normalized:
                blocks.append(f"{label}: " + "; ".join(normalized))

        _append_list("keywords", keywords if isinstance(keywords, list) else [])
        _append_list("capabilities", capabilities if isinstance(capabilities, list) else [])
        _append_list("use_when", use_when if isinstance(use_when, list) else [])
        _append_list("avoid_when", avoid_when if isinstance(avoid_when, list) else [])
        return " | ".join(blocks).strip()

    def _recall_provider_skill_candidates_from_metadata(
        self,
        *,
        user_message: str,
        recent_history: list[dict[str, Any]],
        available_tools: list[dict[str, Any]],
        provider_hint_docs: list[dict[str, Any]],
        skill_hint_docs: list[dict[str, Any]],
        top_k_provider: int,
        top_k_skill: int,
    ) -> dict[str, Any]:
        top_k_provider = max(1, int(top_k_provider or 1))
        top_k_skill = max(1, int(top_k_skill or 1))
        history_text = " ".join(
            str(item.get("content", "") or "").strip()
            for item in recent_history[-4:]
            if isinstance(item, dict)
        )
        request_text = " ".join([str(user_message or "").strip(), history_text]).strip()
        request_tokens = self._tokenize_classifier_fallback_text(request_text)
        if not request_tokens:
            return {
                "provider_candidates": [],
                "skill_candidates": [],
                "preferred_provider_types": [],
                "preferred_capability_classes": [],
                "preferred_tool_names": [],
                "confidence": 0.0,
                "reason": "metadata_recall_empty_request",
            }

        request_text_lower = request_text.lower()
        tool_name_set = {
            str(tool.get("name", "") or "").strip()
            for tool in available_tools
            if isinstance(tool, dict) and str(tool.get("name", "") or "").strip()
        }

        def _score_doc(doc: dict[str, Any]) -> tuple[int, list[str]]:
            matched_tokens: list[str] = []
            token_bag = " ".join(
                [
                    str(doc.get("hint_text", "") or ""),
                    str(doc.get("provider_type", "") or ""),
                    " ".join(str(item) for item in (doc.get("tool_names", []) or [])),
                    " ".join(str(item) for item in (doc.get("capability_classes", []) or [])),
                    " ".join(str(item) for item in (doc.get("group_ids", []) or [])),
                ]
            ).strip()
            token_set = self._tokenize_classifier_fallback_text(token_bag)
            overlap = sorted(request_tokens.intersection(token_set))
            if overlap:
                matched_tokens.extend(overlap[:8])
            score = len(overlap) * 4

            provider_type = str(doc.get("provider_type", "") or "").strip().lower()
            if provider_type and provider_type in request_text_lower:
                score += 8
                matched_tokens.append(provider_type)

            for tool_name in doc.get("tool_names", []) or []:
                normalized_name = str(tool_name or "").strip()
                if not normalized_name:
                    continue
                if normalized_name.lower() in request_text_lower:
                    score += 6
                    matched_tokens.append(normalized_name.lower())

            try:
                priority = int(doc.get("priority", 100) or 100)
            except (TypeError, ValueError):
                priority = 100
            # Priority only breaks ties when request-doc relevance already exists.
            if score > 0 and priority <= 60:
                score += 1
            return score, self._dedupe_preserve_order(matched_tokens)

        provider_ranked: list[dict[str, Any]] = []
        for doc in provider_hint_docs:
            if not isinstance(doc, dict):
                continue
            score, matched_tokens = _score_doc(doc)
            if score <= 0:
                continue
            tool_names = [
                str(name).strip()
                for name in (doc.get("tool_names", []) or [])
                if str(name).strip() in tool_name_set
            ]
            provider_ranked.append(
                {
                    "hint_id": str(doc.get("hint_id", "") or "").strip(),
                    "provider_type": str(doc.get("provider_type", "") or "").strip().lower(),
                    "score": score,
                    "matched_tokens": matched_tokens,
                    "tool_names": tool_names,
                    "capability_classes": [
                        str(item).strip().lower()
                        for item in (doc.get("capability_classes", []) or [])
                        if str(item).strip()
                    ],
                }
            )
        provider_ranked.sort(key=lambda item: (-int(item.get("score", 0) or 0), str(item.get("hint_id", ""))))
        provider_top = provider_ranked[:top_k_provider]

        skill_ranked: list[dict[str, Any]] = []
        for doc in skill_hint_docs:
            if not isinstance(doc, dict):
                continue
            score, matched_tokens = _score_doc(doc)
            if score <= 0:
                continue
            tool_names = [
                str(name).strip()
                for name in (doc.get("tool_names", []) or [])
                if str(name).strip() in tool_name_set
            ]
            skill_ranked.append(
                {
                    "hint_id": str(doc.get("hint_id", "") or "").strip(),
                    "provider_type": str(doc.get("provider_type", "") or "").strip().lower(),
                    "score": score,
                    "matched_tokens": matched_tokens,
                    "tool_names": tool_names,
                    "capability_classes": [
                        str(item).strip().lower()
                        for item in (doc.get("capability_classes", []) or [])
                        if str(item).strip()
                    ],
                }
            )
        skill_ranked.sort(key=lambda item: (-int(item.get("score", 0) or 0), str(item.get("hint_id", ""))))
        skill_top = skill_ranked[:top_k_skill]

        preferred_provider_types = self._dedupe_preserve_order(
            [
                str(item.get("provider_type", "") or "").strip().lower()
                for item in provider_top + skill_top
                if str(item.get("provider_type", "") or "").strip()
            ]
        )
        preferred_capability_classes = self._dedupe_preserve_order(
            [
                str(capability).strip().lower()
                for item in provider_top + skill_top
                for capability in (item.get("capability_classes", []) or [])
                if str(capability).strip()
            ]
        )
        preferred_tool_names = self._dedupe_preserve_order(
            [
                str(name).strip()
                for item in provider_top + skill_top
                for name in (item.get("tool_names", []) or [])
                if str(name).strip() in tool_name_set
            ]
        )[:12]

        total_score = sum(int(item.get("score", 0) or 0) for item in provider_top + skill_top)
        confidence_denominator = max(24, len(request_tokens) * 8)
        confidence = min(1.0, float(total_score) / float(confidence_denominator))
        reason = "metadata_recall_matched" if (provider_top or skill_top) else "metadata_recall_no_match"
        return {
            "provider_candidates": provider_top,
            "skill_candidates": skill_top,
            "preferred_provider_types": preferred_provider_types,
            "preferred_capability_classes": preferred_capability_classes,
            "preferred_tool_names": preferred_tool_names,
            "confidence": confidence,
            "reason": reason,
        }

    def _apply_provider_hard_prefilter(
        self,
        *,
        decision: ToolGateDecision,
        match_result: CapabilityMatchResult,
        available_tools: list[dict[str, Any]],
        deps: Optional[SkillDeps] = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        trace: dict[str, Any] = {
            "enabled": False,
            "before_count": len(available_tools),
            "after_count": len(available_tools),
            "target_provider_types": [],
            "target_capability_classes": [],
            "matched_provider_tool_count": 0,
            "retained_builtin_tools": [],
            "reason": "prefilter_not_enabled",
        }
        if not decision.needs_external_system:
            return available_tools, trace
        if not available_tools:
            trace["reason"] = "prefilter_empty_toolset"
            return available_tools, trace

        metadata_candidates: dict[str, Any] = {}
        if deps is not None and isinstance(getattr(deps, "extra", None), dict):
            raw = deps.extra.get("tool_metadata_candidates")
            if isinstance(raw, dict):
                metadata_candidates = dict(raw)

        target_provider_types: list[str] = []
        target_capability_classes: list[str] = []
        target_tool_names: list[str] = []
        explicit_provider_types: list[str] = []
        explicit_provider_capability_classes: list[str] = []

        target_provider_types.extend(
            str(item).strip().lower()
            for item in (metadata_candidates.get("preferred_provider_types", []) or [])
            if str(item).strip()
        )
        target_capability_classes.extend(
            str(item).strip().lower()
            for item in (metadata_candidates.get("preferred_capability_classes", []) or [])
            if str(item).strip()
        )
        target_tool_names.extend(
            str(item).strip()
            for item in (metadata_candidates.get("preferred_tool_names", []) or [])
            if str(item).strip()
        )

        for capability in decision.suggested_tool_classes:
            normalized = str(capability or "").strip().lower()
            if not normalized:
                continue
            target_capability_classes.append(normalized)
            if normalized.startswith("provider:"):
                provider_key = normalized.split(":", 1)[1].strip()
                target_provider_types.append(provider_key)
                explicit_provider_types.append(provider_key)
                explicit_provider_capability_classes.append(normalized)

        for candidate in match_result.tool_candidates:
            candidate_name = str(getattr(candidate, "name", "") or "").strip()
            candidate_capability = str(getattr(candidate, "capability_class", "") or "").strip().lower()
            if candidate_name:
                target_tool_names.append(candidate_name)
            if candidate_capability:
                target_capability_classes.append(candidate_capability)
            provider_type = ""
            if candidate_capability.startswith("provider:"):
                provider_type = candidate_capability.split(":", 1)[1].strip()
            else:
                provider_type = str(getattr(candidate, "provider_type", "") or "").strip().lower()
            if provider_type:
                target_provider_types.append(provider_type)

        active_provider_capability = self._resolve_active_provider_capability_class(
            deps=deps,
            provider_skill_classes=self._collect_provider_skill_capability_classes(available_tools),
        )
        if active_provider_capability:
            target_capability_classes.append(active_provider_capability.lower())
            if active_provider_capability.startswith("provider:"):
                target_provider_types.append(active_provider_capability.split(":", 1)[1].strip().lower())

        if explicit_provider_types:
            explicit_provider_types = self._dedupe_preserve_order(explicit_provider_types)
            explicit_provider_capability_classes = self._dedupe_preserve_order(
                explicit_provider_capability_classes
            )
            target_provider_types = list(explicit_provider_types)
            target_capability_classes = [
                capability
                for capability in target_capability_classes
                if not capability.startswith("provider:")
            ]
            target_capability_classes.extend(explicit_provider_capability_classes)

        target_provider_types = self._dedupe_preserve_order(
            [item for item in target_provider_types if item]
        )
        target_capability_classes = self._dedupe_preserve_order(
            [item for item in target_capability_classes if item]
        )
        target_tool_names = self._dedupe_preserve_order(
            [item for item in target_tool_names if item]
        )

        filtered_tools: list[dict[str, Any]] = []
        retained_builtin_tools: list[str] = []
        matched_provider_tool_count = 0
        for tool in available_tools:
            tool_name = str(tool.get("name", "") or "").strip()
            capability = str(tool.get("capability_class", "") or "").strip().lower()
            provider_type = str(tool.get("provider_type", "") or "").strip().lower()
            is_provider_or_skill = self._is_provider_or_skill_tool(tool)

            if is_provider_or_skill:
                provider_match = (
                    provider_type in target_provider_types
                    if provider_type and target_provider_types
                    else False
                )
                capability_match = capability in target_capability_classes if capability else False
                name_match = tool_name in target_tool_names if tool_name else False
                implicit_provider_capability = f"provider:{provider_type}" if provider_type else ""
                implicit_match = (
                    implicit_provider_capability in target_capability_classes
                    if implicit_provider_capability
                    else False
                )
                if provider_match or capability_match or name_match or implicit_match:
                    filtered_tools.append(tool)
                    matched_provider_tool_count += 1
                continue

            if tool_name and tool_name in target_tool_names:
                filtered_tools.append(tool)
                retained_builtin_tools.append(tool_name)
                continue
            if capability and capability in target_capability_classes:
                filtered_tools.append(tool)
                retained_builtin_tools.append(tool_name or capability)

        if matched_provider_tool_count <= 0 or not filtered_tools:
            trace.update(
                {
                    "enabled": False,
                    "after_count": len(available_tools),
                    "target_provider_types": target_provider_types,
                    "target_capability_classes": target_capability_classes,
                    "matched_provider_tool_count": matched_provider_tool_count,
                    "retained_builtin_tools": retained_builtin_tools,
                    "reason": "prefilter_no_provider_match",
                }
            )
            return available_tools, trace

        trace.update(
            {
                "enabled": True,
                "after_count": len(filtered_tools),
                "target_provider_types": target_provider_types,
                "target_capability_classes": target_capability_classes,
                "matched_provider_tool_count": matched_provider_tool_count,
                "retained_builtin_tools": retained_builtin_tools,
                "reason": "prefilter_applied",
            }
        )
        return filtered_tools, trace

    @staticmethod
    def _is_provider_or_skill_tool(tool: dict[str, Any]) -> bool:
        capability = str(tool.get("capability_class", "") or "").strip().lower()
        provider_type = str(tool.get("provider_type", "") or "").strip().lower()
        category = str(tool.get("category", "") or "").strip().lower()
        if capability == "skill" or capability.startswith("provider:"):
            return True
        if provider_type:
            return True
        return category.startswith("provider") or "skill" in category

    @staticmethod
    def _dedupe_preserve_order(values: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for item in values:
            normalized = str(item or "").strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            ordered.append(normalized)
        return ordered

    @staticmethod
    def _should_attempt_hint_ranking(
        *,
        available_tools: list[dict[str, Any]],
        provider_hint_docs: list[dict[str, Any]],
        skill_hint_docs: list[dict[str, Any]],
        metadata_candidates: Optional[dict[str, Any]] = None,
        min_confidence: float = 0.3,
    ) -> bool:
        if len(available_tools) <= 1:
            return False
        if not (provider_hint_docs or skill_hint_docs):
            return False
        if not isinstance(metadata_candidates, dict):
            return False
        has_candidates = bool(
            metadata_candidates.get("provider_candidates")
            or metadata_candidates.get("skill_candidates")
        )
        if not has_candidates:
            return False
        confidence = float(metadata_candidates.get("confidence", 0.0) or 0.0)
        return confidence >= max(0.0, float(min_confidence))

    async def _rank_tools_with_hint_docs(
        self,
        *,
        agent: Any,
        deps: SkillDeps,
        user_message: str,
        recent_history: list[dict[str, Any]],
        available_tools: list[dict[str, Any]],
        provider_hint_docs: list[dict[str, Any]],
        skill_hint_docs: list[dict[str, Any]],
    ) -> tuple[Optional[dict[str, Any]], str]:
        if agent is None:
            return None, "ranking_agent_unavailable"
        if not self._should_attempt_hint_ranking(
            available_tools=available_tools,
            provider_hint_docs=provider_hint_docs,
            skill_hint_docs=skill_hint_docs,
        ):
            return None, "ranking_not_required"

        ranker_prompt = self._build_tool_hint_ranker_prompt(
            available_tools=available_tools,
            provider_hint_docs=provider_hint_docs,
            skill_hint_docs=skill_hint_docs,
        )
        ranker_message = self._build_tool_hint_ranker_message(
            user_message=user_message,
            recent_history=recent_history,
        )
        timeout_seconds = float(getattr(self, "TOOL_HINT_RANKER_TIMEOUT_SECONDS", 2.5) or 2.5)
        try:
            raw_output = await asyncio.wait_for(
                self._run_single_with_optional_override(
                    agent=agent,
                    user_message=ranker_message,
                    deps=deps,
                    system_prompt=ranker_prompt,
                ),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            return None, "hint_ranker_timeout"
        except Exception as exc:
            return None, f"hint_ranker_error:{exc.__class__.__name__}"

        parsed = self._extract_json_object(raw_output)
        if not parsed:
            return None, "hint_ranker_invalid_json"
        try:
            payload = json.loads(parsed)
        except Exception:
            return None, "hint_ranker_parse_failed"
        if not isinstance(payload, dict):
            return None, "hint_ranker_non_object"

        normalized = self._coerce_tool_hint_ranking_payload(
            payload=payload,
            available_tools=available_tools,
        )
        if normalized is None:
            return None, "hint_ranker_empty_result"
        return normalized, ""

    def _build_tool_hint_ranker_prompt(
        self,
        *,
        available_tools: list[dict[str, Any]],
        provider_hint_docs: list[dict[str, Any]],
        skill_hint_docs: list[dict[str, Any]],
    ) -> str:
        tool_lines: list[str] = []
        for tool in available_tools:
            name = str(tool.get("name", "") or "").strip()
            if not name:
                continue
            description = str(tool.get("description", "") or "").strip()
            provider_type = str(tool.get("provider_type", "") or "").strip()
            capability = str(tool.get("capability_class", "") or "").strip()
            group_ids = tool.get("group_ids", [])
            group_text = ", ".join(
                str(item).strip() for item in (group_ids or []) if str(item).strip()
            )
            tool_lines.append(
                f"- {name} | provider={provider_type or '-'} | capability={capability or '-'} | "
                f"groups={group_text or '-'} | desc={description}"
            )

        provider_lines: list[str] = []
        for doc in provider_hint_docs[:16]:
            provider_lines.append(
                f"- {doc.get('hint_id', 'provider:?')} | tools={', '.join(doc.get('tool_names', [])[:6])} | "
                f"{str(doc.get('hint_text', '') or '').strip()}"
            )
        if not provider_lines:
            provider_lines.append("- none")

        skill_lines: list[str] = []
        for doc in skill_hint_docs[:24]:
            skill_lines.append(
                f"- {doc.get('hint_id', 'skill:?')} | tools={', '.join(doc.get('tool_names', [])[:6])} | "
                f"{str(doc.get('hint_text', '') or '').strip()}"
            )
        if not skill_lines:
            skill_lines.append("- none")

        return (
            "You are AtlasClaw's internal tool-ranking classifier.\n"
            "Do not answer the user and do not call tools.\n"
            "Return one JSON object only.\n\n"
            "Task:\n"
            "Given the user request, allowed runtime tools, and provider/skill metadata hints,\n"
            "rank which provider/capability/tools should be preferred FIRST.\n"
            "This is a soft ranking hint, not an execution command.\n\n"
            "Constraints:\n"
            "- Only return provider types/capabilities/tool names that exist in the allowed tool list.\n"
            "- Keep output concise.\n"
            "- If uncertain, return empty lists with low confidence.\n\n"
            "Allowed tools:\n"
            f"{chr(10).join(tool_lines) if tool_lines else '- none'}\n\n"
            "Provider hint docs:\n"
            f"{chr(10).join(provider_lines)}\n\n"
            "Skill hint docs:\n"
            f"{chr(10).join(skill_lines)}\n\n"
            "Return JSON fields exactly:\n"
            "{\n"
            '  "preferred_provider_types": string[],\n'
            '  "preferred_capability_classes": string[],\n'
            '  "preferred_tool_names": string[],\n'
            '  "confidence": number,\n'
            '  "reason": string\n'
            "}\n"
        )

    @staticmethod
    def _build_tool_hint_ranker_message(
        *,
        user_message: str,
        recent_history: list[dict[str, Any]],
    ) -> str:
        history_lines: list[str] = []
        for item in recent_history[-4:]:
            role = str(item.get("role", "") or "").strip() or "unknown"
            content = str(item.get("content", "") or "").strip().replace("\n", " ")
            if len(content) > 180:
                content = content[:177] + "..."
            history_lines.append(f"- {role}: {content}")
        history_text = "\n".join(history_lines) if history_lines else "- none"
        return (
            "Rank preferred runtime tools for this turn.\n\n"
            f"User request:\n{user_message}\n\n"
            f"Recent history:\n{history_text}\n"
        )

    @staticmethod
    def _coerce_tool_hint_ranking_payload(
        *,
        payload: dict[str, Any],
        available_tools: list[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        allowed_tool_names = {
            str(item.get("name", "") or "").strip()
            for item in available_tools
            if isinstance(item, dict) and str(item.get("name", "") or "").strip()
        }
        allowed_provider_types = {
            str(item.get("provider_type", "") or "").strip().lower()
            for item in available_tools
            if isinstance(item, dict) and str(item.get("provider_type", "") or "").strip()
        }
        allowed_capabilities = {
            str(item.get("capability_class", "") or "").strip().lower()
            for item in available_tools
            if isinstance(item, dict) and str(item.get("capability_class", "") or "").strip()
        }

        def _normalize_list(raw: Any) -> list[str]:
            if isinstance(raw, str):
                return [part.strip() for part in re.split(r"[,;\n]", raw) if part.strip()]
            if isinstance(raw, list):
                return [str(item).strip() for item in raw if str(item).strip()]
            return []

        provider_types = [
            item.lower()
            for item in _normalize_list(payload.get("preferred_provider_types", []))
            if item.lower() in allowed_provider_types
        ]
        capability_classes = [
            item.lower()
            for item in _normalize_list(payload.get("preferred_capability_classes", []))
            if item.lower() in allowed_capabilities
        ]
        tool_names = [
            item
            for item in _normalize_list(payload.get("preferred_tool_names", []))
            if item in allowed_tool_names
        ]
        reason = str(payload.get("reason", "") or "").strip()
        confidence_raw = payload.get("confidence", 0.0)
        try:
            confidence = max(0.0, min(1.0, float(confidence_raw)))
        except Exception:
            confidence = 0.0

        if not provider_types and not capability_classes and not tool_names:
            return None
        return {
            "preferred_provider_types": provider_types,
            "preferred_capability_classes": capability_classes,
            "preferred_tool_names": tool_names,
            "confidence": confidence,
            "reason": reason,
        }

    @staticmethod
    def _reorder_tools_by_hint_ranking(
        *,
        available_tools: list[dict[str, Any]],
        ranking: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not available_tools:
            return [], {}
        preferred_tools = [
            str(item).strip()
            for item in ranking.get("preferred_tool_names", [])
            if str(item).strip()
        ]
        preferred_capabilities = {
            str(item).strip().lower()
            for item in ranking.get("preferred_capability_classes", [])
            if str(item).strip()
        }
        preferred_providers = {
            str(item).strip().lower()
            for item in ranking.get("preferred_provider_types", [])
            if str(item).strip()
        }
        tool_order = {name: len(preferred_tools) - idx for idx, name in enumerate(preferred_tools)}

        scored: list[tuple[int, str, dict[str, Any]]] = []
        for tool in available_tools:
            name = str(tool.get("name", "") or "").strip()
            if not name:
                continue
            capability = str(tool.get("capability_class", "") or "").strip().lower()
            provider_type = str(tool.get("provider_type", "") or "").strip().lower()
            try:
                priority = int(tool.get("priority", 100) or 100)
            except (TypeError, ValueError):
                priority = 100
            score = priority
            if name in tool_order:
                score += 300 + (tool_order[name] * 10)
            if capability and capability in preferred_capabilities:
                score += 220
            if provider_type and provider_type in preferred_providers:
                score += 160
            if capability.startswith("provider:"):
                provider_key = capability.split(":", 1)[1].strip().lower()
                if provider_key and provider_key in preferred_providers:
                    score += 140
            scored.append((score, name.lower(), tool))

        scored.sort(key=lambda item: (-item[0], item[1]))
        reordered = [item[2] for item in scored]
        top_tool_names = [
            str(item.get("name", "")).strip()
            for item in reordered[:3]
            if str(item.get("name", "")).strip()
        ]
        top_tool_hints = [
            f"{str(item.get('name', '')).strip()}: {str(item.get('description', '')).strip()}"
            for item in reordered[:3]
            if str(item.get("name", "")).strip()
        ]
        trace = {
            "preferred_provider_types": sorted(preferred_providers),
            "preferred_capability_classes": sorted(preferred_capabilities),
            "preferred_tool_names": preferred_tools,
            "confidence": float(ranking.get("confidence", 0.0) or 0.0),
            "reason": str(ranking.get("reason", "") or "").strip(),
            "top_tool_names": top_tool_names,
            "top_tool_hints": top_tool_hints,
        }
        return reordered, trace

    @staticmethod
    def _coerce_tool_gate_payload(payload: dict[str, Any]) -> dict[str, Any]:
        def _read_bool(key: str, default: bool = False) -> bool:
            value = payload.get(key, default)
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(value)
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"true", "1", "yes", "y"}:
                    return True
                if lowered in {"false", "0", "no", "n"}:
                    return False
            return default

        suggested = payload.get("suggested_tool_classes", [])
        if isinstance(suggested, str):
            suggested = [part.strip() for part in re.split(r"[,;\n]", suggested) if part.strip()]
        elif not isinstance(suggested, list):
            suggested = []
        suggested = [str(item).strip() for item in suggested if str(item).strip()]

        confidence = payload.get("confidence", 0.0)
        try:
            confidence_value = float(confidence)
        except Exception:
            confidence_value = 0.0
        confidence_value = max(0.0, min(1.0, confidence_value))

        policy_raw = str(payload.get("policy", ToolPolicyMode.ANSWER_DIRECT.value) or "").strip().lower()
        policy_aliases = {
            "answer": ToolPolicyMode.ANSWER_DIRECT.value,
            "direct": ToolPolicyMode.ANSWER_DIRECT.value,
            "answer_direct": ToolPolicyMode.ANSWER_DIRECT.value,
            "prefer": ToolPolicyMode.PREFER_TOOL.value,
            "prefer_tool": ToolPolicyMode.PREFER_TOOL.value,
            "tool_preferred": ToolPolicyMode.PREFER_TOOL.value,
            "must": ToolPolicyMode.MUST_USE_TOOL.value,
            "must_use": ToolPolicyMode.MUST_USE_TOOL.value,
            "must_use_tool": ToolPolicyMode.MUST_USE_TOOL.value,
            "tool_required": ToolPolicyMode.MUST_USE_TOOL.value,
        }
        policy_value = policy_aliases.get(policy_raw, ToolPolicyMode.ANSWER_DIRECT.value)

        needs_live_data = _read_bool("needs_live_data")
        needs_private_context = _read_bool("needs_private_context")
        needs_external_system = _read_bool("needs_external_system")
        needs_browser_interaction = _read_bool("needs_browser_interaction")
        needs_grounded_verification = _read_bool("needs_grounded_verification")
        needs_tool = _read_bool("needs_tool") or bool(
            suggested
            or needs_live_data
            or needs_private_context
            or needs_external_system
            or needs_browser_interaction
            or needs_grounded_verification
        )

        reason = str(payload.get("reason", "") or "").strip()
        if not reason:
            reason = "Model classifier returned a partial decision; normalized by runtime."

        return {
            "needs_tool": needs_tool,
            "needs_live_data": needs_live_data,
            "needs_private_context": needs_private_context,
            "needs_external_system": needs_external_system,
            "needs_browser_interaction": needs_browser_interaction,
            "needs_grounded_verification": needs_grounded_verification,
            "suggested_tool_classes": suggested,
            "confidence": confidence_value,
            "reason": reason,
            "policy": policy_value,
        }

    def _build_tool_gate_classifier_prompt(self, available_tools: list[dict[str, Any]]) -> str:
        capabilities: list[str] = []
        for tool in available_tools:
            name = str(tool.get("name", "")).strip()
            capability = str(tool.get("capability_class", "")).strip()
            description = str(tool.get("description", "")).strip()
            if capability:
                capabilities.append(f"- {name}: {capability} ({description})")
            else:
                capabilities.append(f"- {name}: {description}")

        capability_text = "\n".join(capabilities) if capabilities else "- no runtime tools available"
        return (
            "You are AtlasClaw's internal tool-necessity classifier.\n"
            "Your job is to decide whether the user request can be answered reliably without tools.\n"
            "Do not answer the user. Do not call tools. Return a single JSON object only.\n\n"
            "Policy rubric:\n"
            "- Use must_use_tool when reliable response requires fresh external facts, enterprise system actions, or verifiable evidence.\n"
            "- If the user asks to query/operate enterprise systems or provider-backed skills, set needs_external_system=true and prefer provider/skill classes over web classes.\n"
            "- Use web_search/web_fetch for public web real-time verification (news, prices, schedules, etc.) when no dedicated domain tool is available.\n"
            "- Do not route provider/skill requests to web_search when provider/skill capabilities are available.\n"
            "- Requests about current or near-future changing facts must prefer tool-backed verification over direct answers.\n"
            "- Use prefer_tool when tools would improve confidence but a general direct answer is still acceptable.\n"
            "- Use answer_direct only when the request can be answered reliably from stable knowledge.\n\n"
            "Available runtime capabilities:\n"
            f"{capability_text}\n\n"
            "Return JSON with exactly these fields:\n"
            "{\n"
            '  "needs_tool": boolean,\n'
            '  "needs_live_data": boolean,\n'
            '  "needs_private_context": boolean,\n'
            '  "needs_external_system": boolean,\n'
            '  "needs_browser_interaction": boolean,\n'
            '  "needs_grounded_verification": boolean,\n'
            '  "suggested_tool_classes": string[],\n'
            '  "confidence": number,\n'
            '  "reason": string,\n'
            '  "policy": "answer_direct" | "prefer_tool" | "must_use_tool"\n'
            "}\n"
        )

    def _build_tool_gate_classifier_message(
        self,
        *,
        user_message: str,
        recent_history: list[dict[str, Any]],
        metadata_candidates: Optional[dict[str, Any]] = None,
    ) -> str:
        now_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
        history_lines: list[str] = []
        for item in recent_history[-4:]:
            role = str(item.get("role", "")).strip() or "unknown"
            content = str(item.get("content", "")).strip().replace("\n", " ")
            if len(content) > 180:
                content = content[:177] + "..."
            history_lines.append(f"- {role}: {content}")
        history_text = "\n".join(history_lines) if history_lines else "- none"
        provider_candidates = []
        skill_candidates = []
        preferred_capabilities = []
        preferred_tools = []
        if isinstance(metadata_candidates, dict):
            confidence = float(metadata_candidates.get("confidence", 0.0) or 0.0)
            min_confidence = float(
                getattr(self, "TOOL_HINT_RANKER_MIN_METADATA_CONFIDENCE", 0.3) or 0.3
            )
            if confidence < min_confidence:
                metadata_candidates = {}
        if isinstance(metadata_candidates, dict):
            provider_candidates = [
                str(item.get("provider_type", "")).strip()
                for item in (metadata_candidates.get("provider_candidates", []) or [])[:3]
                if isinstance(item, dict) and str(item.get("provider_type", "")).strip()
            ]
            skill_candidates = [
                str(item.get("hint_id", "")).strip()
                for item in (metadata_candidates.get("skill_candidates", []) or [])[:4]
                if isinstance(item, dict) and str(item.get("hint_id", "")).strip()
            ]
            preferred_capabilities = [
                str(item).strip()
                for item in (metadata_candidates.get("preferred_capability_classes", []) or [])[:8]
                if str(item).strip()
            ]
            preferred_tools = [
                str(item).strip()
                for item in (metadata_candidates.get("preferred_tool_names", []) or [])[:8]
                if str(item).strip()
            ]
        metadata_hint_block = (
            "Metadata candidates (runtime pre-recall):\n"
            f"- provider_types: {', '.join(provider_candidates) if provider_candidates else 'none'}\n"
            f"- skill_hints: {', '.join(skill_candidates) if skill_candidates else 'none'}\n"
            f"- preferred_capabilities: {', '.join(preferred_capabilities) if preferred_capabilities else 'none'}\n"
            f"- preferred_tools: {', '.join(preferred_tools) if preferred_tools else 'none'}\n"
        )
        return (
            "Classify the following request for runtime policy.\n\n"
            f"Runtime UTC time:\n{now_utc}\n\n"
            f"User request:\n{user_message}\n\n"
            f"Recent history:\n{history_text}\n\n"
            f"{metadata_hint_block}"
        )

    async def _run_single_with_optional_override(
        self,
        *,
        agent: Any,
        user_message: str,
        deps: SkillDeps,
        system_prompt: Optional[str] = None,
    ) -> str:
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
                result = await agent.run(user_message, deps=deps)
        else:
            with override_cm:
                result = await agent.run(user_message, deps=deps)

        output = result.output if hasattr(result, "output") else result
        return str(output).strip()

    @staticmethod
    def _extract_json_object(raw_output: str) -> str:
        text = (raw_output or "").strip()
        if not text:
            return ""
        if text.startswith("```"):
            lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
            text = "\n".join(lines).strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return ""
        return text[start : end + 1]

    @staticmethod
    def _extract_tool_call_arguments(raw_args: Any) -> dict[str, Any]:
        if isinstance(raw_args, dict):
            return dict(raw_args)
        if isinstance(raw_args, str):
            try:
                parsed = json.loads(raw_args)
            except Exception:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

