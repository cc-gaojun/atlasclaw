from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Optional

from app.atlasclaw.core.deps import SkillDeps


class RunnerToolEvidenceMixin:
    def _collect_tool_call_summaries_from_messages(
        self,
        *,
        messages: list[dict[str, Any]],
        start_index: int = 0,
    ) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        safe_start = max(0, min(int(start_index), len(messages)))
        for message in messages[safe_start:]:
            if not isinstance(message, dict):
                continue
            if str(message.get("role", "")).strip().lower() != "assistant":
                continue
            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                name = str(call.get("name", "") or call.get("tool_name", "")).strip()
                if not name:
                    continue
                args_raw = call.get("args", call.get("arguments"))
                args: dict[str, Any] = {}
                if isinstance(args_raw, dict):
                    args = dict(args_raw)
                elif isinstance(args_raw, str):
                    payload = args_raw.strip()
                    if payload.startswith("{"):
                        try:
                            parsed = json.loads(payload)
                            if isinstance(parsed, dict):
                                args = parsed
                        except Exception:
                            args = {}
                signature = (name, json.dumps(args, ensure_ascii=False, sort_keys=True))
                if signature in seen:
                    continue
                seen.add(signature)
                summary: dict[str, Any] = {"name": name}
                if args:
                    summary["args"] = args
                summaries.append(summary)
        return summaries

    def _extract_tool_text_from_messages(
        self,
        *,
        messages: list[dict[str, Any]],
        start_index: int = 0,
        max_chars: int = 6000,
    ) -> str:
        safe_start = max(0, min(int(start_index), len(messages)))
        for message in messages[safe_start:]:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "")).strip().lower()
            if role not in {"tool", "toolresult", "tool_result"}:
                continue
            text = self._coerce_tool_payload_to_text(message.get("content"))
            if not text:
                continue
            normalized = text.strip()
            if not normalized:
                continue
            return normalized[:max_chars]
        return ""

    def _coerce_tool_payload_to_text(self, payload: Any) -> str:
        if payload is None:
            return ""
        if isinstance(payload, str):
            return payload
        if isinstance(payload, list):
            chunks: list[str] = []
            for item in payload:
                block = self._coerce_tool_payload_to_text(item)
                if block:
                    chunks.append(block)
            return "\n".join(chunks).strip()
        if isinstance(payload, dict):
            for key in ("output", "text", "summary", "message"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            if "content" in payload:
                return self._coerce_tool_payload_to_text(payload.get("content"))
            if "results" in payload:
                return self._coerce_tool_payload_to_text(payload.get("results"))
            if "data" in payload:
                return self._coerce_tool_payload_to_text(payload.get("data"))
            try:
                return json.dumps(payload, ensure_ascii=False)
            except Exception:
                return str(payload)
        try:
            return str(payload)
        except Exception:
            return ""

    async def _build_timeout_fallback_message_from_messages(
        self,
        *,
        runtime_agent: Any,
        deps: SkillDeps,
        user_message: str,
        messages: list[dict[str, Any]],
        start_index: int = 0,
    ) -> str:
        """Build a best-effort final answer from tool outputs already present in transcript."""
        tool_text = self._extract_tool_text_from_messages(
            messages=messages,
            start_index=start_index,
            max_chars=9000,
        ).strip()
        if not tool_text:
            return ""

        synthesize_system_prompt = (
            "You are a strict response renderer.\n"
            "You are given tool output captured by the runtime.\n"
            "Rules:\n"
            "1) Use only the provided tool output.\n"
            "2) Do not invent facts.\n"
            "3) Preserve numbers, dates, and units exactly.\n"
            "4) Reply in the same language as the user request.\n"
            "5) Keep the response concise and practical.\n"
            "6) Do not call tools."
        )
        synthesize_user_prompt = (
            f"User request:\n{user_message}\n\n"
            f"Tool output:\n{tool_text}\n\n"
            "Write the final response for the user."
        )
        try:
            synthesized = await asyncio.wait_for(
                self._run_single_with_optional_override(
                    agent=runtime_agent,
                    user_message=synthesize_user_prompt,
                    deps=deps,
                    system_prompt=synthesize_system_prompt,
                ),
                timeout=6.0,
            )
        except Exception:
            synthesized = ""
        final_text = str(synthesized or "").strip()
        if final_text:
            return final_text
        return self._compact_tool_fallback_text(tool_text)

    async def _build_post_tool_wrapped_message(
        self,
        *,
        runtime_agent: Any,
        deps: SkillDeps,
        user_message: str,
        tool_calls: list[dict[str, Any]],
    ) -> str:
        """Wrap tool evidence with a concise model-rendered final answer."""
        evidence_items = await self._collect_tool_evidence_items(tool_calls=tool_calls)
        if not evidence_items:
            return ""

        synthesize_system_prompt = (
            "You are a strict response renderer.\n"
            "You receive tool evidence already collected by the runtime.\n"
            "Rules:\n"
            "1) Use only the provided evidence.\n"
            "2) Do not invent facts.\n"
            "3) Preserve numbers, dates, locations, and units exactly.\n"
            "4) Respond in the same language as the user request.\n"
            "5) Keep the answer concise and include source links when available.\n"
            "6) Do not call tools."
        )
        synthesize_user_prompt = (
            f"User request:\n{user_message}\n\n"
            f"Tool evidence (JSON):\n{json.dumps(evidence_items, ensure_ascii=False)}\n\n"
            "Write the final answer for the user."
        )
        try:
            synthesized = await asyncio.wait_for(
                self._run_single_with_optional_override(
                    agent=runtime_agent,
                    user_message=synthesize_user_prompt,
                    deps=deps,
                    system_prompt=synthesize_system_prompt,
                ),
                timeout=6.0,
            )
        except Exception:
            synthesized = ""

        final_text = (synthesized or "").strip()
        if final_text:
            return final_text

        for item in evidence_items:
            result = item.get("result")
            if not isinstance(result, dict):
                continue
            fallback = self._extract_tool_text_result(result)
            if fallback:
                return self._compact_tool_fallback_text(fallback)
        return ""

    async def _collect_tool_evidence_items(
        self,
        *,
        tool_calls: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            name = str(tool_call.get("name", "") or "").strip()
            args = tool_call.get("args")
            if not name:
                continue
            evidence: dict[str, Any] = {"tool": name}
            if isinstance(args, dict) and args:
                evidence["arguments"] = dict(args)
                tool_result = await self._invoke_tool_evidence_adapter(
                    tool_name=name,
                    tool_args=args,
                )
                if isinstance(tool_result, dict) and not bool(tool_result.get("is_error")):
                    evidence["result"] = tool_result
            items.append(evidence)
        return items

    async def _invoke_tool_evidence_adapter(
        self,
        *,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        if tool_name == "openmeteo_weather":
            return await self._invoke_openmeteo_weather(tool_args=tool_args)
        return None

    async def _invoke_openmeteo_weather(
        self,
        *,
        tool_args: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        try:
            from app.atlasclaw.tools.web.openmeteo_weather_tool import openmeteo_weather_tool
        except Exception:
            return None

        allowed_keys = {
            "location",
            "target_date",
            "days",
            "country_code",
            "timezone",
            "temperature_unit",
            "wind_speed_unit",
            "precipitation_unit",
        }
        safe_args: dict[str, Any] = {}
        for key in allowed_keys:
            if key in tool_args:
                safe_args[key] = tool_args[key]

        location = str(safe_args.get("location", "") or "").strip()
        if not location:
            return None

        days_value = safe_args.get("days")
        if days_value is not None:
            try:
                safe_args["days"] = int(days_value)
            except (TypeError, ValueError):
                safe_args.pop("days", None)

        try:
            result = await openmeteo_weather_tool(None, **safe_args)
        except Exception:
            return None
        return result if isinstance(result, dict) else None

    @staticmethod
    def _extract_tool_text_result(tool_result: dict[str, Any]) -> str:
        content_blocks = tool_result.get("content")
        if not isinstance(content_blocks, list):
            return ""
        for block in content_blocks:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "text":
                continue
            text = str(block.get("text", "") or "").strip()
            if text:
                return text
        return ""

    @staticmethod
    def _compact_tool_fallback_text(text: str, max_chars: int = 1400) -> str:
        normalized = str(text or "").strip()
        if not normalized:
            return ""
        normalized = re.sub(
            r"##[A-Z_]+_META_START##.*?##[A-Z_]+_META_END##",
            "",
            normalized,
            flags=re.DOTALL,
        ).strip()
        if not normalized:
            return ""

        lines: list[str] = []
        total = 0
        for raw_line in normalized.splitlines():
            line = " ".join(raw_line.split()).strip()
            if not line:
                continue
            if line.startswith("{") and len(line) > 240:
                continue
            if line.startswith("[") and len(line) > 240:
                continue
            if total + len(line) + 1 > max_chars:
                break
            lines.append(line)
            total += len(line) + 1
            if len(lines) >= 18:
                break
        if not lines:
            clipped = normalized[:max_chars].strip()
            if len(normalized) > max_chars:
                clipped += " ..."
            return clipped

        compacted = "\n".join(lines).strip()
        if len(compacted) < len(normalized):
            compacted += "\n..."
        return compacted

    @staticmethod
    def _replace_last_assistant_message(
        *,
        messages: list[dict[str, Any]],
        content: str,
    ) -> list[dict[str, Any]]:
        updated = list(messages)
        for index in range(len(updated) - 1, -1, -1):
            item = updated[index]
            if str(item.get("role", "")).strip() != "assistant":
                continue
            replaced = dict(item)
            replaced["content"] = content
            updated[index] = replaced
            return updated
        updated.append({"role": "assistant", "content": content})
        return updated

    @staticmethod
    def _extract_latest_assistant_from_messages(
        *,
        messages: list[dict[str, Any]],
        start_index: int,
    ) -> str:
        if not isinstance(messages, list) or not messages:
            return ""
        safe_start = max(0, min(int(start_index), len(messages)))
        for item in reversed(messages[safe_start:]):
            if not isinstance(item, dict):
                continue
            if str(item.get("role", "")).strip() != "assistant":
                continue
            content = str(item.get("content", "") or "").strip()
            if content:
                return content
        return ""

    @staticmethod
    def _remove_last_assistant_from_run(
        *,
        messages: list[dict[str, Any]],
        start_index: int,
    ) -> list[dict[str, Any]]:
        updated = list(messages)
        safe_start = max(0, min(int(start_index), len(updated)))
        for index in range(len(updated) - 1, safe_start - 1, -1):
            item = updated[index]
            if not isinstance(item, dict):
                continue
            if str(item.get("role", "")).strip() != "assistant":
                continue
            return updated[:index] + updated[index + 1 :]
        return updated

