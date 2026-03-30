# -*- coding: utf-8 -*-
from __future__ import annotations

import pytest

from app.atlasclaw.agent.runner import AgentRunner
from app.atlasclaw.auth.models import UserInfo
from app.atlasclaw.core.deps import SkillDeps
from app.atlasclaw.hooks.runtime import HookRuntime, HookRuntimeContext
from app.atlasclaw.hooks.runtime_builtin import RUNTIME_AUDIT_MODULE, register_builtin_hook_handlers
from app.atlasclaw.hooks.runtime_models import HookEventType
from app.atlasclaw.hooks.runtime_sinks import ContextSink, MemorySink
from app.atlasclaw.hooks.runtime_store import HookStateStore
from app.atlasclaw.session.manager import SessionManager
from app.atlasclaw.session.queue import SessionQueue
from app.atlasclaw.session.router import SessionManagerRouter


class _TextNode:
    def __init__(self, content: str):
        self.content = content


class _FakeAgentRun:
    def __init__(self, nodes: list[object], all_messages: list[dict]):
        self._nodes = nodes
        self._all_messages = all_messages
        self._index = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._nodes):
            raise StopAsyncIteration
        node = self._nodes[self._index]
        self._index += 1
        return node

    def all_messages(self):
        return self._all_messages


class _EchoAgent:
    def iter(self, user_message, deps, message_history):
        final_messages = list(message_history) + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": f"reply:{user_message}"},
        ]
        return _FakeAgentRun([_TextNode(f"reply:{user_message}")], final_messages)


class _FailingAgent:
    def iter(self, user_message, deps, message_history):
        raise RuntimeError("simulated runner failure")


def _build_runner(tmp_path, agent):
    session_manager = SessionManager(workspace_path=str(tmp_path), user_id="default")
    session_router = SessionManagerRouter.from_manager(session_manager)
    hook_state_store = HookStateStore(workspace_path=str(tmp_path))
    hook_runtime = HookRuntime(
        HookRuntimeContext(
            workspace_path=str(tmp_path),
            hook_state_store=hook_state_store,
            memory_sink=MemorySink(str(tmp_path)),
            context_sink=ContextSink(hook_state_store),
            session_manager_router=session_router,
        )
    )
    register_builtin_hook_handlers(hook_runtime)
    runner = AgentRunner(
        agent=agent,
        session_manager=session_manager,
        session_manager_router=session_router,
        session_queue=SessionQueue(),
        hook_runtime=hook_runtime,
    )
    deps = SkillDeps(
        user_info=UserInfo(user_id="alice", display_name="alice"),
        session_key="agent:main:user:alice:web:dm:alice:topic:test",
        session_manager=session_router.for_user("alice"),
        memory_manager=None,
        cookies={},
        extra={"run_id": "run-1"},
    )
    return runner, hook_state_store, deps


@pytest.mark.asyncio
async def test_runner_emits_hook_runtime_events(tmp_path):
    runner, hook_state_store, deps = _build_runner(tmp_path, _EchoAgent())

    events = [event async for event in runner.run(
        session_key=deps.session_key,
        user_message="hello",
        deps=deps,
    )]

    assert events
    stored = await hook_state_store.list_events(RUNTIME_AUDIT_MODULE, "alice")
    event_types = [item.event_type for item in stored]
    assert HookEventType.MESSAGE_RECEIVED in event_types
    assert HookEventType.RUN_STARTED in event_types
    assert HookEventType.LLM_REQUESTED in event_types
    assert HookEventType.LLM_COMPLETED in event_types
    assert HookEventType.RUN_CONTEXT_READY in event_types
    assert HookEventType.RUN_COMPLETED in event_types


@pytest.mark.asyncio
async def test_runner_failure_emits_failed_events_and_pending(tmp_path):
    runner, hook_state_store, deps = _build_runner(tmp_path, _FailingAgent())

    events = [event async for event in runner.run(
        session_key=deps.session_key,
        user_message="hello",
        deps=deps,
    )]

    assert any(event.type == "error" for event in events)
    stored = await hook_state_store.list_events(RUNTIME_AUDIT_MODULE, "alice")
    event_types = [item.event_type for item in stored]
    assert HookEventType.RUN_FAILED in event_types
    assert HookEventType.LLM_FAILED in event_types
    assert HookEventType.RUN_CONTEXT_READY in event_types
    assert HookEventType.RUN_COMPLETED not in event_types
    pending = await hook_state_store.list_pending(RUNTIME_AUDIT_MODULE, "alice")
    assert len(pending) >= 1
