# -*- coding: utf-8 -*-
"""Tool catalog group and profile behavior tests."""

from __future__ import annotations

from app.atlasclaw.tools.catalog import (
    GROUP_ATLASCLAW,
    GROUP_AUTOMATION,
    GROUP_FS,
    GROUP_RUNTIME,
    GROUP_TOOLS,
    GROUP_UI,
    GROUP_WEB,
    ToolCatalog,
    ToolProfile,
)


def test_group_map_contains_required_openclaw_aligned_groups() -> None:
    assert GROUP_RUNTIME in GROUP_TOOLS
    assert GROUP_FS in GROUP_TOOLS
    assert GROUP_WEB in GROUP_TOOLS
    assert GROUP_UI in GROUP_TOOLS
    assert GROUP_AUTOMATION in GROUP_TOOLS
    assert GROUP_ATLASCLAW in GROUP_TOOLS


def test_required_group_membership_is_defined() -> None:
    assert {"exec", "process"}.issubset(set(GROUP_TOOLS[GROUP_RUNTIME]))
    assert {"read", "write", "edit"}.issubset(set(GROUP_TOOLS[GROUP_FS]))
    assert {"web_search", "web_fetch"}.issubset(set(GROUP_TOOLS[GROUP_WEB]))
    assert {"browser"}.issubset(set(GROUP_TOOLS[GROUP_UI]))


def test_atlasclaw_group_is_union_of_core_builtin_tools() -> None:
    atlasclaw_group = set(GROUP_TOOLS[GROUP_ATLASCLAW])
    assert {"exec", "process", "read", "write", "edit"}.issubset(atlasclaw_group)
    assert {"web_search", "web_fetch", "session_status"}.issubset(atlasclaw_group)


def test_expand_groups_dedupes_tools_and_ignores_unknown_groups() -> None:
    expanded = ToolCatalog.expand_groups([GROUP_RUNTIME, GROUP_FS, GROUP_RUNTIME, "group:missing"])
    assert expanded.count("exec") == 1
    assert expanded.count("process") == 1
    assert "read" in expanded
    assert "write" in expanded


def test_full_profile_resolves_to_atlasclaw_tool_union() -> None:
    tools = ToolCatalog.get_tools_by_profile(ToolProfile.FULL)
    for required in ("exec", "process", "read", "write", "edit", "web_search", "web_fetch"):
        assert required in tools


def test_filter_tools_respects_allow_and_deny_group_rules() -> None:
    tools = ToolCatalog.get_tools_by_profile(ToolProfile.FULL)
    filtered = ToolCatalog.filter_tools(tools, allow=[GROUP_WEB, "session_status"], deny=["web_fetch"])
    assert "web_search" in filtered
    assert "web_fetch" not in filtered
    assert "session_status" in filtered
    assert "exec" not in filtered
