# -*- coding: utf-8 -*-

from __future__ import annotations

from pydantic_ai.messages import ModelRequest, SystemPromptPart, UserPromptPart

from app.atlasclaw.agent.compaction import CompactionConfig, CompactionPipeline
from app.atlasclaw.agent.history_memory import HistoryMemoryCoordinator


def test_history_memory_normalize_messages_splits_system_and_user_parts():
    coordinator = HistoryMemoryCoordinator(
        session_manager=object(),
        compaction=CompactionPipeline(CompactionConfig()),
    )
    message = ModelRequest(
        parts=[
            SystemPromptPart(content="system rules"),
            UserPromptPart(content="hello atlas"),
        ]
    )

    normalized = coordinator.normalize_messages([message])

    assert normalized == [
        {"role": "system", "content": "system rules"},
        {"role": "user", "content": "hello atlas"},
    ]
