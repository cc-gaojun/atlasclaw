# -*- coding: utf-8 -*-
"""Reusable section renderers for PromptBuilder."""

from __future__ import annotations

import platform
from datetime import datetime
from pathlib import Path
from typing import Optional


def build_target_md_skill(target_md_skill: dict[str, str]) -> str:
    """Build a focused section for webhook-directed markdown skill execution."""
    qualified_name = target_md_skill.get("qualified_name", "")
    file_path = target_md_skill.get("file_path", "")
    provider = target_md_skill.get("provider", "")
    lines = ["## Target Markdown Skill", ""]
    if qualified_name:
        lines.append(f"Qualified name: {qualified_name}")
    if provider:
        lines.append(f"Provider: {provider}")
    if file_path:
        lines.append(f"File path: {file_path}")
    lines.append("You must execute only this markdown skill for the current run.")
    lines.append("Prefer any executable tool already registered for this skill.")
    return "\n".join(lines)


def build_user_context(user_info) -> str:
    """Build a user identity section for the current authenticated operator."""
    lines = ["## Current User", ""]
    if user_info.display_name:
        lines.append(f"Name: {user_info.display_name}")
    lines.append(f"User ID: {user_info.user_id}")
    if user_info.tenant_id and user_info.tenant_id != "default":
        lines.append(f"Tenant: {user_info.tenant_id}")
    if user_info.roles:
        lines.append(f"Roles: {', '.join(user_info.roles)}")
    return "\n".join(lines)


def build_identity(config) -> str:
    """Build the identity section."""
    return f"""## Identity

You are {config.agent_name}, {config.agent_description}.

Your core capabilities include:
- Handling complex multi-turn conversations with context continuity
- Invoking various business skills (cloud resource management, ITSM, ticket processing, etc.)
- Managing long-term memory with semantic retrieval
- Supporting multi-step workflows and task collaboration"""


def build_tooling(tools: list[dict]) -> str:
    """Build the tool listing section."""
    lines = ["## Tools", ""]
    lines.append("You can use the following tools to complete tasks:")
    lines.append("")
    for tool in tools:
        name = tool.get("name", "unknown")
        description = tool.get("description", "")
        lines.append(f"- **{name}**: {description}")
    return "\n".join(lines)


def build_tool_policy(tool_policy: Optional[dict]) -> str:
    """Build explicit tool-policy guidance for the current turn."""
    if not isinstance(tool_policy, dict):
        return ""

    mode = str(tool_policy.get("mode", "") or "").strip()
    reason = str(tool_policy.get("reason", "") or "").strip()
    required_tools = tool_policy.get("required_tools", [])
    execution_hint = str(tool_policy.get("execution_hint", "") or "").strip().lower()
    retry_count = int(tool_policy.get("retry_count", 0) or 0)
    retry_missing_tools = tool_policy.get("retry_missing_tools", [])
    top_tool_hints = tool_policy.get("top_tool_hints", [])
    if not mode:
        return ""

    lines = ["## Tool Policy", ""]
    lines.append(f"Policy mode: {mode}")
    if reason:
        lines.append(f"Reason: {reason}")
    if isinstance(required_tools, list) and required_tools:
        lines.append(f"Preferred tools: {', '.join(str(item) for item in required_tools)}")
    if isinstance(top_tool_hints, list) and top_tool_hints:
        lines.append("Top tool hints:")
        for hint in top_tool_hints[:3]:
            normalized_hint = str(hint or "").strip()
            if normalized_hint:
                lines.append(f"- {normalized_hint}")
    if retry_count > 0:
        lines.append(f"Policy retry: {retry_count}")
        if isinstance(retry_missing_tools, list) and retry_missing_tools:
            lines.append(
                "Previously missing evidence tools: "
                + ", ".join(str(item) for item in retry_missing_tools)
            )
    lines.extend(
        [
            "",
            "You must not claim any search, verification, lookup, or provider query happened unless tool execution evidence exists in this run.",
        ]
    )
    if mode == "must_use_tool":
        lines.append("A grounded tool-backed result is required before you provide a final answer.")
        lines.append("For this turn, execute at least one required tool before any substantive assistant response.")
        if execution_hint == "provider_tool_first":
            lines.append("This turn is provider/skill tool-first: issue a required provider/skill tool call immediately.")
            lines.append("Do not provide narrative analysis before the first required tool call.")
        lines.append("If required tool execution fails or returns no evidence, explicitly state verification failed and do not fabricate results.")
    elif mode == "prefer_tool":
        lines.append("Prefer the listed tools or scoped context before answering.")
    else:
        lines.append("You may answer directly when the request is stable and does not require verification.")
    return "\n".join(lines)


def build_safety() -> str:
    """Build the safety section."""
    return """## Safety

Please follow these safety guidelines:
- Avoid power-seeking behaviors or bypassing oversight
- Do not execute operations that may cause irreversible damage
- Sensitive information must be desensitized
- Respect user data privacy"""


def build_skills_listing(skills: list[dict]) -> str:
    """Build available built-in executable skill listing."""
    if not skills:
        return ""

    lines = ["## Built-in Tools (Use ONLY if no MD Skill matches)", "", "<available_skills>"]
    for skill in skills:
        name = skill.get("name", "unknown")
        description = skill.get("description", "")
        location = skill.get("location", "built-in")
        category = skill.get("category", "utility")
        lines.append(
            f"""  <skill>
    <name>{name}</name>
    <description>{description}</description>
    <category>{category}</category>
    <location>{location}</location>
  </skill>"""
        )
    lines.append("</available_skills>")
    lines.append("\nNOTE: These built-in tools are fallback options. ALWAYS check MD Skills section above first.")
    return "\n".join(lines)


def build_md_skills_index(
    config,
    md_skills: list[dict],
    provider_contexts: Optional[dict[str, dict]] = None,
) -> str:
    """Build a compact markdown-skills index for prompt-time discovery."""
    if not md_skills:
        return ""

    max_count = config.md_skills_max_count
    desc_max = config.md_skills_desc_max_chars
    budget = config.md_skills_max_index_chars
    home_prefix = str(Path.home())
    _ = provider_contexts or {}

    header_lines = [
        "## Skills",
        "",
        "Skills are listed as compact metadata only to save context tokens.",
        "When you need detailed instructions for a skill, call the `read` tool on the skill `file_path` (`SKILL.md`) before executing.",
        "Do not assume the full skill file is already loaded in context.",
        "",
        "Format: `name | description | file_path`",
        "",
    ]
    accumulated = "\n".join(header_lines)
    shown = 0
    total_count = len(md_skills)

    for skill in md_skills[:max_count]:
        name = str(skill.get("qualified_name") or skill.get("name") or "unknown").strip()
        desc = str(skill.get("description", "") or "").strip()
        file_path = str(skill.get("file_path", "") or "").strip()

        if len(desc) > desc_max:
            desc = desc[: desc_max - 3] + "..."
        if home_prefix and file_path.startswith(home_prefix):
            file_path = "~" + file_path[len(home_prefix) :]

        entry = f"- `{name}` | {desc} | `{file_path}`\n"
        if len(accumulated) + len(entry) > budget:
            break
        accumulated += entry
        shown += 1

    if shown < total_count:
        note = f"\n<!-- Showing {shown} of {total_count} skills due to budget/count limits -->"
        if len(accumulated) + len(note) <= budget:
            accumulated += note
        else:
            remaining = budget - len(accumulated)
            if remaining > 4:
                accumulated += note[: remaining - 3] + "..."
            elif remaining > 0:
                accumulated += note[:remaining]
    return accumulated


def build_self_update() -> str:
    """Build self-update section."""
    return """## Self-Update

To apply configuration changes, use the appropriate configuration commands."""


def build_workspace_info(config) -> str:
    """Build workspace section."""
    workspace = Path(config.workspace_path).expanduser()
    return f"""## Workspace

Working directory: `{workspace}`

You can read and write files in this directory."""


def build_documentation() -> str:
    """Build documentation pointers section."""
    return """## Documentation

Local documentation path: `docs/`

To understand AtlasClaw's behavior, commands, configuration, or architecture, please refer to the local documentation first."""


def build_sandbox(config) -> str:
    """Build sandbox section."""
    return f"""## Sandbox

Mode: {config.sandbox.mode}
Sandbox path: {config.sandbox.workspace_root}
Elevated execution: {"Available" if config.sandbox.elevated_exec else "Unavailable"}

In sandbox mode, some operations may be restricted."""


def build_datetime(config) -> str:
    """Build current datetime section."""
    now = datetime.now()
    tz = config.user_timezone or "System timezone"
    if config.time_format == "12":
        time_str = now.strftime("%Y-%m-%d %I:%M:%S %p")
    else:
        time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    return f"""## Current Time

Timezone: {tz}
Current time: {time_str}"""


def build_reply_tags() -> str:
    """Build reply-tags section (currently optional/no-op)."""
    return ""


def build_heartbeats(
    *,
    heartbeat_markdown: str = "",
    every_seconds: Optional[int] = None,
    active_hours: str = "",
    isolated_session: bool = False,
) -> str:
    """Build heartbeat guidance section when heartbeat context is available."""
    if not heartbeat_markdown.strip():
        return ""

    lines = ["## Heartbeat", ""]
    if every_seconds is not None:
        lines.append(f"Schedule: every {every_seconds} seconds")
    if active_hours:
        lines.append(f"Active hours: {active_hours}")
    lines.append(
        "Execution mode: isolated session"
        if isolated_session
        else "Execution mode: shared session"
    )
    lines.append("")
    lines.append(heartbeat_markdown.strip())
    return "\n".join(lines)


def build_runtime_info() -> str:
    """Build runtime environment info section."""
    return f"""## Runtime

Host: {platform.node()}
OS: {platform.system()} {platform.release()}
Python: {platform.python_version()}
Framework: AtlasClaw v0.1.0"""
