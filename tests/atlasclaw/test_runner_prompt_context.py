# -*- coding: utf-8 -*-
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

from pydantic_ai import Agent

from app.atlasclaw.agent.prompt_builder import PromptMode
from app.atlasclaw.agent.runner_prompt_context import collect_tools_snapshot
from app.atlasclaw.agent.runner_tool.runner_execution_prepare import (
    build_explicit_tool_execution_prompt,
    select_execution_prompt_mode,
    select_explicit_tool_execution_target,
)
from app.atlasclaw.agent.tool_gate_models import ToolIntentAction, ToolIntentPlan


def test_collect_tools_snapshot_prefers_deps_extra_snapshot() -> None:
    deps = SimpleNamespace(extra={"tools_snapshot": [{"name": "web_search", "description": "search web"}]})
    snapshot = collect_tools_snapshot(agent=object(), deps=deps)
    assert snapshot == [
        {"name": "web_search", "description": "search web", "capability_class": "web_search"}
    ]


def test_collect_tools_snapshot_keeps_authoritative_snapshot_without_remerge() -> None:
    agent = SimpleNamespace(
        tools=[
            {"name": "web_search", "description": "search web"},
            {"name": "web_fetch", "description": "fetch web page"},
        ]
    )
    deps = SimpleNamespace(
        extra={
            "tools_snapshot": [{"name": "web_search", "description": "search web"}],
            "tools_snapshot_authoritative": True,
        }
    )
    snapshot = collect_tools_snapshot(agent=agent, deps=deps)
    assert snapshot == [
        {"name": "web_search", "description": "search web", "capability_class": "web_search"}
    ]


def test_collect_tools_snapshot_merges_agent_tools_when_snapshot_not_authoritative() -> None:
    agent = SimpleNamespace(
        tools=[
            {"name": "web_search", "description": "search web"},
            {"name": "web_fetch", "description": "fetch web page"},
        ]
    )
    deps = SimpleNamespace(
        extra={
            "tools_snapshot": [{"name": "web_search", "description": "search web"}],
            "tools_snapshot_authoritative": False,
        }
    )
    snapshot = collect_tools_snapshot(agent=agent, deps=deps)
    names = [item["name"] for item in snapshot]
    assert names == ["web_search", "web_fetch"]


def test_collect_tools_snapshot_preserves_normalized_metadata_from_deps() -> None:
    deps = SimpleNamespace(
        extra={
            "tools_snapshot": [
                {
                    "name": "cmp_list_pending",
                    "description": "List pending CMP approvals",
                    "source": "provider",
                    "provider_type": "smartcmp",
                    "group_ids": ["group:cmp"],
                    "capability_class": "provider:smartcmp",
                    "priority": 150,
                    "parameters_schema": {
                        "type": "object",
                        "properties": {
                            "identifier": {"type": "string"},
                            "days": {"type": "integer"},
                        },
                        "required": ["identifier"],
                    },
                    "planner_visibility": "always",
                    "aliases": ["cmp", "smartcmp approvals"],
                    "keywords": ["approval", "pending"],
                    "use_when": ["User asks for pending approvals"],
                    "avoid_when": ["User asks for weather"],
                    "result_mode": "tool_only_ok",
                }
            ]
        }
    )
    snapshot = collect_tools_snapshot(agent=object(), deps=deps)
    assert snapshot == [
        {
            "name": "cmp_list_pending",
            "description": "List pending CMP approvals",
            "source": "provider",
            "provider_type": "smartcmp",
            "group_ids": ["group:cmp"],
            "capability_class": "provider:smartcmp",
            "priority": 150,
            "parameters_schema": {
                "type": "object",
                "properties": {
                    "identifier": {"type": "string"},
                    "days": {"type": "integer"},
                },
                "required": ["identifier"],
            },
            "planner_visibility": "always",
            "aliases": ["cmp", "smartcmp approvals"],
            "keywords": ["approval", "pending"],
            "use_when": ["User asks for pending approvals"],
            "avoid_when": ["User asks for weather"],
            "result_mode": "tool_only_ok",
        }
    ]


def test_collect_tools_snapshot_reads_pydantic_ai_toolsets() -> None:
    agent = Agent("test")

    @agent.tool_plain
    def web_search(query: str) -> str:
        """Search the web by query."""
        return query

    deps = SimpleNamespace(extra={"tools_snapshot": []})
    snapshot = collect_tools_snapshot(agent=agent, deps=deps)
    assert snapshot
    assert any(tool["name"] == "web_search" for tool in snapshot)


def test_collect_tools_snapshot_infers_provider_capability_from_skills_snapshot() -> None:
    agent = SimpleNamespace(
        tools=[
            {
                "name": "jira_search",
                "description": "Search Jira issues",
            }
        ]
    )
    deps = SimpleNamespace(
        extra={
            "tools_snapshot": [],
            "skills_snapshot": [
                {
                    "name": "jira_search",
                    "description": "Search Jira issues",
                    "category": "provider",
                    "provider_type": "jira",
                }
            ],
            "md_skills_snapshot": [],
        }
    )
    snapshot = collect_tools_snapshot(agent=agent, deps=deps)
    assert snapshot == [
        {
            "name": "jira_search",
            "description": "Search Jira issues",
            "provider_type": "jira",
            "category": "provider",
            "capability_class": "provider:jira",
        }
    ]


def test_collect_tools_snapshot_infers_md_skill_capability() -> None:
    agent = SimpleNamespace(
        tools=[
            {
                "name": "summarize_skill_run",
                "description": "Run summarize skill",
            }
        ]
    )
    deps = SimpleNamespace(
        extra={
            "tools_snapshot": [],
            "skills_snapshot": [],
            "md_skills_snapshot": [
                {
                    "name": "summarize",
                    "provider": "",
                    "metadata": {
                        "tool_name": "summarize_skill_run",
                        "category": "skill",
                    },
                }
            ],
        }
    )
    snapshot = collect_tools_snapshot(agent=agent, deps=deps)
    assert snapshot == [
        {
            "name": "summarize_skill_run",
            "description": "Run summarize skill",
            "category": "skill",
            "source": "md_skill",
            "skill_name": "summarize",
            "capability_class": "skill",
        }
    ]


def test_collect_tools_snapshot_falls_back_to_skills_snapshot_when_agent_has_no_tools() -> None:
    agent = SimpleNamespace(tools=[])
    deps = SimpleNamespace(
        extra={
            "tools_snapshot": [],
            "skills_snapshot": [
                {
                    "name": "web_search",
                    "description": "Web search",
                    "category": "builtin:web",
                },
                {
                    "name": "openmeteo_weather",
                    "description": "Weather lookup",
                    "category": "builtin:web",
                },
            ],
            "md_skills_snapshot": [],
        }
    )

    snapshot = collect_tools_snapshot(agent=agent, deps=deps)
    assert any(tool["name"] == "web_search" for tool in snapshot)
    assert any(tool["name"] == "openmeteo_weather" for tool in snapshot)
    assert any(
        tool["name"] == "web_search" and tool.get("capability_class") == "web_search"
        for tool in snapshot
    )


def test_collect_tools_snapshot_preserves_runtime_tool_metadata_fields() -> None:
    agent = SimpleNamespace(
        tools=[
            {
                "name": "openmeteo_weather",
                "description": "Get current and forecast weather via Open-Meteo APIs",
                "source": "builtin",
                "group_ids": ["group:web"],
                "capability_class": "weather",
                "planner_visibility": "contextual",
                "aliases": ["weather", "forecast"],
                "keywords": ["天气", "预报", "temperature"],
                "use_when": ["User asks for a forecast by place and date"],
                "avoid_when": ["User asks for enterprise approvals"],
                "result_mode": "tool_only_ok",
            }
        ]
    )
    deps = SimpleNamespace(extra={"tools_snapshot": []})

    snapshot = collect_tools_snapshot(agent=agent, deps=deps)

    assert snapshot == [
        {
            "name": "openmeteo_weather",
            "description": "Get current and forecast weather via Open-Meteo APIs",
            "source": "builtin",
            "group_ids": ["group:web"],
            "capability_class": "weather",
            "planner_visibility": "contextual",
            "aliases": ["weather", "forecast"],
            "keywords": ["天气", "预报", "temperature"],
            "use_when": ["User asks for a forecast by place and date"],
            "avoid_when": ["User asks for enterprise approvals"],
            "result_mode": "tool_only_ok",
        }
    ]


def test_collect_tools_snapshot_does_not_stringify_none_provider_metadata() -> None:
    deps = SimpleNamespace(
        extra={
            "tools_snapshot": [
                {
                    "name": "web_search",
                    "description": "Web search",
                    "provider_type": "",
                    "category": "builtin:web",
                    "source": "builtin",
                    "group_ids": ["group:web"],
                    "capability_class": "web_search",
                    "planner_visibility": "general",
                }
            ],
            "skills_snapshot": [
                {
                    "name": "web_search",
                    "description": "Web search",
                    "provider_type": None,
                    "category": "builtin:web",
                    "source": "builtin",
                    "group_ids": ["group:web"],
                    "capability_class": "web_search",
                    "planner_visibility": "general",
                }
            ],
            "md_skills_snapshot": [],
        }
    )

    snapshot = collect_tools_snapshot(agent=object(), deps=deps)

    assert snapshot == [
        {
            "name": "web_search",
            "description": "Web search",
            "category": "builtin:web",
            "source": "builtin",
            "group_ids": ["group:web"],
            "capability_class": "web_search",
            "planner_visibility": "general",
        }
    ]


def test_select_execution_prompt_mode_uses_minimal_for_small_explicit_toolset() -> None:
    mode = select_execution_prompt_mode(
        intent_action="use_tools",
        is_follow_up=False,
        projected_tool_count=1,
    )

    assert mode is PromptMode.MINIMAL


def test_select_execution_prompt_mode_keeps_full_for_follow_up_tool_turn() -> None:
    mode = select_execution_prompt_mode(
        intent_action="use_tools",
        is_follow_up=True,
        projected_tool_count=1,
    )

    assert mode is PromptMode.FULL


def test_select_explicit_tool_execution_target_returns_single_tool_only_candidate() -> None:
    target = select_explicit_tool_execution_target(
        intent_plan=ToolIntentPlan(action=ToolIntentAction.USE_TOOLS, target_tool_names=["openmeteo_weather"]),
        is_follow_up=False,
        projected_tools=[
            {
                "name": "openmeteo_weather",
                "description": "Get weather forecast",
                "result_mode": "tool_only_ok",
                "capability_class": "weather",
            },
            {
                "name": "select_provider_instance",
                "description": "Select provider instance",
                "capability_class": "session",
            },
        ],
    )

    assert target is not None
    assert target["name"] == "openmeteo_weather"


def test_select_explicit_tool_execution_target_skips_follow_up_and_non_terminal_tools() -> None:
    follow_up_target = select_explicit_tool_execution_target(
        intent_plan=ToolIntentPlan(action=ToolIntentAction.USE_TOOLS, target_tool_names=["openmeteo_weather"]),
        is_follow_up=True,
        projected_tools=[
            {
                "name": "openmeteo_weather",
                "description": "Get weather forecast",
                "result_mode": "tool_only_ok",
            }
        ],
    )
    llm_target = select_explicit_tool_execution_target(
        intent_plan=ToolIntentPlan(action=ToolIntentAction.USE_TOOLS, target_tool_names=["smartcmp_submit"]),
        is_follow_up=False,
        projected_tools=[
            {
                "name": "smartcmp_submit",
                "description": "Submit SmartCMP request",
                "result_mode": "llm",
            }
        ],
    )

    assert follow_up_target is None
    assert llm_target is None


def test_build_explicit_tool_execution_prompt_is_compact_and_includes_tool_schema() -> None:
    prompt = build_explicit_tool_execution_prompt(
        tool={
            "name": "openmeteo_weather",
            "description": "Get weather forecast for a city.",
            "capability_class": "weather",
            "result_mode": "tool_only_ok",
            "parameters_schema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City or place name"},
                    "target_date": {"type": "string", "description": "Target date in YYYY-MM-DD"},
                },
                "required": ["location"],
            },
        },
        now_local=datetime(2026, 4, 11, 9, 30, 0),
    )

    assert "Allowed tool:" in prompt
    assert "- name: openmeteo_weather" in prompt
    assert "- location (string, required): City or place name" in prompt
    assert "- target_date (string, optional): Target date in YYYY-MM-DD" in prompt
    assert "Do not answer from memory." in prompt
