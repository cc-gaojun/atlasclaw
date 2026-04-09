# -*- coding: utf-8 -*-
"""Prompt-context helpers for AgentRunner."""

from __future__ import annotations

import inspect
from typing import Any, Optional


def build_system_prompt(
    prompt_builder,
    session: Any,
    deps,
    *,
    agent: Optional[Any] = None,
    context_window_tokens: Optional[int] = None,
) -> str:
    """Build the runtime system prompt for the current session."""
    kwargs = {
        "session": session,
        "skills": collect_skills_snapshot(deps),
        "tools": collect_tools_snapshot(agent=agent, deps=deps),
        "md_skills": collect_md_skills_snapshot(deps),
        "target_md_skill": collect_target_md_skill(deps),
        "tool_policy": collect_tool_policy(deps),
        "user_info": deps.user_info,
        "provider_contexts": collect_provider_contexts(deps),
        "context_window_tokens": context_window_tokens,
    }
    build_fn = prompt_builder.build
    try:
        signature = inspect.signature(build_fn)
    except (TypeError, ValueError):
        signature = None

    if signature is not None:
        accepted = set(signature.parameters.keys())
        kwargs = {key: value for key, value in kwargs.items() if key in accepted}

    return build_fn(**kwargs)


def collect_skills_snapshot(deps) -> list[dict]:
    """Read a structured skills snapshot from `deps.extra` if present."""
    extra = deps.extra if isinstance(deps.extra, dict) else {}
    for key in ("skills_snapshot", "skills"):
        value = extra.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def collect_md_skills_snapshot(deps) -> list[dict]:
    """Read a Markdown-skill snapshot from `deps.extra` if present."""
    extra = deps.extra if isinstance(deps.extra, dict) else {}
    for key in ("md_skills_snapshot", "md_skills"):
        value = extra.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def collect_target_md_skill(deps) -> Optional[dict]:
    """Read a targeted markdown-skill descriptor from `deps.extra` if present."""
    extra = deps.extra if isinstance(deps.extra, dict) else {}
    value = extra.get("target_md_skill")
    return value if isinstance(value, dict) else None


def collect_provider_contexts(deps) -> dict[str, dict]:
    """Collect provider LLM contexts from ServiceProviderRegistry."""
    extra = deps.extra if isinstance(deps.extra, dict) else {}
    registry = extra.get("_service_provider_registry")
    if registry is None:
        return {}

    get_contexts = getattr(registry, "get_all_provider_contexts", None)
    if get_contexts is None:
        return {}

    try:
        contexts = get_contexts()
        result = {}
        for provider_type, ctx in contexts.items():
            if hasattr(ctx, "__dict__"):
                result[provider_type] = {
                    "display_name": getattr(ctx, "display_name", ""),
                    "description": getattr(ctx, "description", ""),
                    "keywords": getattr(ctx, "keywords", []),
                    "capabilities": getattr(ctx, "capabilities", []),
                    "use_when": getattr(ctx, "use_when", []),
                    "avoid_when": getattr(ctx, "avoid_when", []),
                }
            elif isinstance(ctx, dict):
                result[provider_type] = ctx
        return result
    except Exception:
        return {}


def collect_tool_policy(deps) -> Optional[dict]:
    """Read a structured tool policy from `deps.extra` if present."""
    extra = deps.extra if isinstance(deps.extra, dict) else {}
    value = extra.get("tool_policy")
    return value if isinstance(value, dict) else None


def collect_tool_groups_snapshot(deps) -> dict[str, list[str]]:
    """Read normalized tool-group snapshot from `deps.extra` if present."""
    extra = deps.extra if isinstance(deps.extra, dict) else {}
    value = extra.get("tool_groups_snapshot")
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, list[str]] = {}
    for key, members in value.items():
        group_id = str(key or "").strip()
        if not group_id:
            continue
        if isinstance(members, list):
            tool_names = [str(item or "").strip() for item in members if str(item or "").strip()]
        else:
            tool_names = []
        normalized[group_id] = tool_names
    return normalized


def collect_tools_snapshot(*, agent: Any, deps=None) -> list[dict]:
    """Collect tool name and description pairs for prompt building."""
    extra = getattr(deps, "extra", {}) if deps is not None else {}
    normalized_extra_tools: list[dict] = []
    tools_snapshot_authoritative = False
    if isinstance(extra, dict):
        tools_snapshot_authoritative = bool(extra.get("tools_snapshot_authoritative"))
        extra_tools = extra.get("tools_snapshot")
        if isinstance(extra_tools, list):
            for item in extra_tools:
                if not isinstance(item, dict):
                    continue
                payload = _normalize_snapshot_tool(item)
                if payload:
                    normalized_extra_tools.append(payload)

    skills_snapshot = collect_skills_snapshot(deps) if deps is not None else []
    md_skills_snapshot = collect_md_skills_snapshot(deps) if deps is not None else []
    skill_meta_index = _build_skill_metadata_index(
        skills_snapshot,
        md_skills_snapshot,
    )
    tools: list[dict] = []
    seen_names: set[str] = set()

    def _append_tool_record(
        *,
        name: Any,
        description: Any,
        provider_type: Any = None,
        category: Any = None,
        source: Any = None,
        group_ids: Any = None,
        capability_class: Any = None,
        priority: Any = None,
    ) -> None:
        normalized_name = str(name or "").strip()
        if not normalized_name or normalized_name in seen_names:
            return
        normalized_description = str(description or "").strip()
        indexed_meta = skill_meta_index.get(normalized_name, {})
        normalized_provider_type = str(
            provider_type
            or indexed_meta.get("provider_type", "")
            or ""
        ).strip()
        normalized_category = str(
            category
            or indexed_meta.get("category", "")
            or ""
        ).strip()
        normalized_source = str(
            source
            or indexed_meta.get("source", "")
            or ""
        ).strip()
        resolved_group_ids = _normalize_group_ids(
            group_ids
            if group_ids is not None
            else indexed_meta.get("group_ids", []),
        )
        tool_record = {
            "name": normalized_name,
            "description": normalized_description,
        }
        if normalized_provider_type:
            tool_record["provider_type"] = normalized_provider_type
        if normalized_category:
            tool_record["category"] = normalized_category
        if normalized_source:
            tool_record["source"] = normalized_source
        if resolved_group_ids:
            tool_record["group_ids"] = resolved_group_ids
        resolved_priority = _normalize_priority(priority if priority is not None else indexed_meta.get("priority"))
        if resolved_priority is not None:
            tool_record["priority"] = resolved_priority

        explicit_capability_class = str(
            capability_class
            or indexed_meta.get("capability_class", "")
            or ""
        ).strip()
        inferred_capability_class = _infer_capability_class(
            name=normalized_name,
            description=normalized_description,
            provider_type=normalized_provider_type,
            category=normalized_category,
        )
        capability = explicit_capability_class or inferred_capability_class
        if capability:
            tool_record["capability_class"] = capability

        tools.append(tool_record)
        seen_names.add(normalized_name)

    for tool in normalized_extra_tools:
        _append_tool_record(
            name=tool.get("name"),
            description=tool.get("description", ""),
            provider_type=tool.get("provider_type"),
            category=tool.get("category"),
            source=tool.get("source"),
            group_ids=tool.get("group_ids"),
            capability_class=tool.get("capability_class"),
            priority=tool.get("priority"),
        )

    if tools_snapshot_authoritative and tools:
        return tools

    for tool in _iter_tool_entries(agent):
        if isinstance(tool, dict):
            name = tool.get("name")
            description = tool.get("description", "")
            provider_type = tool.get("provider_type")
            category = tool.get("category")
            source = tool.get("source")
            group_ids = tool.get("group_ids")
            capability_class = tool.get("capability_class")
            priority = tool.get("priority")
        else:
            name = getattr(tool, "name", None) or getattr(tool, "__name__", None)
            description = getattr(tool, "description", "") or getattr(tool, "__doc__", "") or ""
            provider_type = (
                getattr(tool, "provider_type", None)
                or getattr(getattr(tool, "metadata", None), "provider_type", None)
            )
            category = (
                getattr(tool, "category", None)
                or getattr(getattr(tool, "metadata", None), "category", None)
            )
            source = (
                getattr(tool, "source", None)
                or getattr(getattr(tool, "metadata", None), "source", None)
            )
            group_ids = (
                getattr(tool, "group_ids", None)
                or getattr(getattr(tool, "metadata", None), "group_ids", None)
            )
            capability_class = (
                getattr(tool, "capability_class", None)
                or getattr(getattr(tool, "metadata", None), "capability_class", None)
            )
            priority = (
                getattr(tool, "priority", None)
                or getattr(getattr(tool, "metadata", None), "priority", None)
            )
        _append_tool_record(
            name=name,
            description=description,
            provider_type=provider_type,
            category=category,
            source=source,
            group_ids=group_ids,
            capability_class=capability_class,
            priority=priority,
        )

    # Fallback/merge path: when pydantic-ai internal tool exposure is partial or missing,
    # recover from the runtime skills snapshot so capability matching remains stable.
    for item in skills_snapshot:
        if not isinstance(item, dict):
            continue
        _append_tool_record(
            name=item.get("name"),
            description=item.get("description", ""),
            provider_type=item.get("provider_type"),
            category=item.get("category"),
            source=item.get("source"),
            group_ids=item.get("group_ids", []),
            capability_class=item.get("capability_class"),
            priority=item.get("priority"),
        )

    return tools


def _iter_tool_entries(agent: Any):
    """Iterate tool entries from legacy and current pydantic_ai agent shapes."""
    raw_tools = getattr(agent, "tools", None)
    if isinstance(raw_tools, dict):
        for tool in raw_tools.values():
            yield tool
    elif isinstance(raw_tools, (list, tuple, set)):
        for tool in raw_tools:
            yield tool

    # pydantic_ai (>=1.5x) exposes tools via toolsets rather than agent.tools
    toolsets = getattr(agent, "toolsets", None)
    if isinstance(toolsets, (list, tuple)):
        for toolset in toolsets:
            toolset_tools = getattr(toolset, "tools", None)
            if isinstance(toolset_tools, dict):
                for tool in toolset_tools.values():
                    yield tool
            elif isinstance(toolset_tools, (list, tuple, set)):
                for tool in toolset_tools:
                    yield tool

    # Defensive fallback for internal function-toolset storage.
    function_toolset = getattr(agent, "_function_toolset", None)
    internal_tools = getattr(function_toolset, "tools", None)
    if isinstance(internal_tools, dict):
        for tool in internal_tools.values():
            yield tool


def _build_skill_metadata_index(
    skills_snapshot: list[dict],
    md_skills_snapshot: list[dict],
) -> dict[str, dict[str, str]]:
    """Build a per-tool metadata index from skill snapshots."""
    index: dict[str, dict[str, str]] = {}
    for item in skills_snapshot:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        index[name] = {
            "provider_type": str(item.get("provider_type", "")).strip(),
            "category": str(item.get("category", "")).strip(),
            "source": str(item.get("source", "")).strip(),
            "group_ids": _normalize_group_ids(item.get("group_ids", [])),
            "capability_class": str(item.get("capability_class", "")).strip(),
            "priority": _normalize_priority(item.get("priority")),
        }

    for entry in md_skills_snapshot:
        if not isinstance(entry, dict):
            continue
        metadata = entry.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        provider_type = str(
            metadata.get("provider_type", "")
            or entry.get("provider", "")
            or ""
        ).strip()
        category = str(metadata.get("category", "skill")).strip() or "skill"
        for tool_name in _extract_md_tool_names(entry):
            index[tool_name] = {
                "provider_type": provider_type,
                "category": category,
                "source": "provider" if provider_type else "md_skill",
                "group_ids": _normalize_group_ids(metadata.get("group_ids", [])),
                "capability_class": str(metadata.get("capability_class", "")).strip(),
                "priority": _normalize_priority(metadata.get("priority")),
            }

    return index


def _extract_md_tool_names(entry: dict) -> list[str]:
    metadata = entry.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    names: list[str] = []
    single_name = str(metadata.get("tool_name", "")).strip()
    if single_name:
        names.append(single_name)
    for key, value in metadata.items():
        key_str = str(key)
        if not key_str.startswith("tool_") or not key_str.endswith("_name"):
            continue
        tool_name = str(value or "").strip()
        if tool_name:
            names.append(tool_name)

    fallback_name = str(entry.get("name", "")).strip()
    if fallback_name and fallback_name not in names:
        names.append(fallback_name)
    return names


def _infer_capability_class(
    *,
    name: str,
    description: str,
    provider_type: str,
    category: str,
) -> str:
    lowered_name = (name or "").strip().lower()
    lowered_description = (description or "").strip().lower()
    lowered_category = (category or "").strip().lower()
    lowered_provider = (provider_type or "").strip().lower()

    if lowered_provider:
        return f"provider:{lowered_provider}"
    if lowered_name in {"web_search", "web_fetch"}:
        return lowered_name
    if lowered_name == "openmeteo_weather":
        return "weather"
    if lowered_name == "browser":
        return "browser"
    if "jira" in lowered_name or "jira" in lowered_description:
        return "provider:jira"
    if "provider:" in lowered_description or lowered_category.startswith("provider"):
        return "provider:generic"
    if "skill" in lowered_category or lowered_category == "md_skill":
        return "skill"
    if "skill" in lowered_description and lowered_name not in {"web_search", "web_fetch"}:
        return "skill"
    return ""


def _normalize_snapshot_tool(item: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        "name": str(item.get("name", "")).strip(),
        "description": str(item.get("description", "")).strip(),
    }
    if not normalized["name"]:
        return {}

    provider_type = str(item.get("provider_type", "")).strip()
    category = str(item.get("category", "")).strip()
    source = str(item.get("source", "")).strip()
    capability_class = str(item.get("capability_class", "")).strip()
    group_ids = _normalize_group_ids(item.get("group_ids", []))
    priority = _normalize_priority(item.get("priority"))

    if provider_type:
        normalized["provider_type"] = provider_type
    if category:
        normalized["category"] = category
    if source:
        normalized["source"] = source
    if capability_class:
        normalized["capability_class"] = capability_class
    if group_ids:
        normalized["group_ids"] = group_ids
    if priority is not None:
        normalized["priority"] = priority
    return normalized


def _normalize_group_ids(values: Any) -> list[str]:
    if not isinstance(values, list):
        values = [values] if values else []
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value or "").strip()
        if not normalized:
            continue
        if not normalized.startswith("group:"):
            normalized = f"group:{normalized}"
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _normalize_priority(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
