from __future__ import annotations

import asyncio
from contextlib import nullcontext
from datetime import datetime, timezone
import inspect
import json
import logging
import re
from typing import Any, Optional

from app.atlasclaw.agent.tool_gate_models import ToolGateDecision, ToolPolicyMode
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

class RunnerToolGateModelMixin:
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

