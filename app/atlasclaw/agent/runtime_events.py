"""Runtime event and hook dispatch helpers for AgentRunner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.atlasclaw.agent.stream import StreamEvent
from app.atlasclaw.hooks.runtime import HookRuntime
from app.atlasclaw.hooks.runtime_models import HookEventType
from app.atlasclaw.session.context import SessionKey


@dataclass
class ToolDispatchResult:
    """Result returned by tool-call dispatch."""

    events: list[StreamEvent]
    tool_calls_count: int
    should_break: bool = False


class RuntimeEventDispatcher:
    """Dispatch runtime hooks and convert tool activity into stream events."""

    def __init__(
        self,
        hooks: Any = None,
        session_queue: Any = None,
        hook_runtime: HookRuntime | None = None,
    ) -> None:
        self.hooks = hooks
        self.queue = session_queue
        self.hook_runtime = hook_runtime

    async def trigger_llm_input(
        self,
        *,
        session_key: str,
        run_id: str = "",
        user_message: str,
        system_prompt: str,
        message_history: list[dict],
    ) -> None:
        payload = {
            "session_key": session_key,
            "user_message": user_message,
            "system_prompt": system_prompt,
            "message_history": message_history,
        }
        if self.hooks:
            await self.hooks.trigger("llm_input", payload)
        await self._emit_runtime_event(
            HookEventType.LLM_REQUESTED,
            session_key=session_key,
            run_id=run_id,
            payload=payload,
        )

    async def trigger_agent_end(
        self,
        *,
        session_key: str,
        run_id: str = "",
        tool_calls_count: int,
        compaction_applied: bool,
    ) -> None:
        payload = {
            "session_key": session_key,
            "tool_calls_count": tool_calls_count,
            "compaction_applied": compaction_applied,
        }
        if self.hooks:
            await self.hooks.trigger("agent_end", payload)
        await self._emit_runtime_event(
            HookEventType.RUN_COMPLETED,
            session_key=session_key,
            run_id=run_id,
            payload=payload,
        )

    async def trigger_run_started(
        self,
        *,
        session_key: str,
        run_id: str,
        user_message: str,
    ) -> None:
        await self._emit_runtime_event(
            HookEventType.RUN_STARTED,
            session_key=session_key,
            run_id=run_id,
            payload={"user_message": user_message},
        )

    async def trigger_message_received(
        self,
        *,
        session_key: str,
        run_id: str,
        user_message: str,
    ) -> None:
        await self._emit_runtime_event(
            HookEventType.MESSAGE_RECEIVED,
            session_key=session_key,
            run_id=run_id,
            payload={"message": user_message},
        )

    async def trigger_run_failed(
        self,
        *,
        session_key: str,
        run_id: str,
        error: str,
    ) -> None:
        await self._emit_runtime_event(
            HookEventType.RUN_FAILED,
            session_key=session_key,
            run_id=run_id,
            payload={"error": error},
        )

    async def trigger_llm_completed(
        self,
        *,
        session_key: str,
        run_id: str,
        assistant_message: str,
    ) -> None:
        await self._emit_runtime_event(
            HookEventType.LLM_COMPLETED,
            session_key=session_key,
            run_id=run_id,
            payload={"assistant_message": assistant_message},
        )

    async def trigger_llm_failed(
        self,
        *,
        session_key: str,
        run_id: str,
        error: str,
    ) -> None:
        await self._emit_runtime_event(
            HookEventType.LLM_FAILED,
            session_key=session_key,
            run_id=run_id,
            payload={"error": error},
        )

    async def trigger_run_context_ready(
        self,
        *,
        session_key: str,
        run_id: str,
        user_message: str,
        system_prompt: str,
        message_history: list[dict],
        assistant_message: str,
        tool_calls: list[dict[str, Any]],
        run_status: str,
        error: str = "",
        session_title: str = "",
    ) -> None:
        await self._emit_runtime_event(
            HookEventType.RUN_CONTEXT_READY,
            session_key=session_key,
            run_id=run_id,
            payload={
                "user_message": user_message,
                "system_prompt": system_prompt,
                "message_history": message_history,
                "assistant_message": assistant_message,
                "tool_calls": tool_calls,
                "run_status": run_status,
                "error": error,
                "session_title": session_title,
            },
        )

    def collect_tool_calls(self, node: Any) -> list[Any]:
        """Collect tool-call metadata from an agent node."""
        if hasattr(node, "tool_call_metadata") and node.tool_call_metadata:
            return list(node.tool_call_metadata)
        if hasattr(node, "tool_calls") and node.tool_calls:
            return list(node.tool_calls)
        if hasattr(node, "tool_name"):
            return [{"name": str(node.tool_name)}]
        return []

    async def dispatch_tool_calls(
        self,
        tool_calls_in_node: list[Any],
        *,
        tool_calls_count: int,
        max_tool_calls: int,
        deps: Any,
        session_key: str,
        run_id: str = "",
    ) -> ToolDispatchResult:
        """Dispatch tool start/end events and related hooks."""
        events: list[StreamEvent] = []

        for tc in tool_calls_in_node:
            tool_calls_count += 1
            if isinstance(tc, dict):
                tool_name = tc.get("name", tc.get("tool_name", "unknown_tool"))
            else:
                tool_name = getattr(tc, "tool_name", getattr(tc, "name", "unknown_tool"))
            tool_name = str(tool_name)

            if deps.is_aborted():
                events.append(StreamEvent.lifecycle_aborted())
                return ToolDispatchResult(events=events, tool_calls_count=tool_calls_count, should_break=True)

            if tool_calls_count > max_tool_calls:
                events.append(StreamEvent.error_event("max_tool_calls_exceeded"))
                return ToolDispatchResult(events=events, tool_calls_count=tool_calls_count, should_break=True)

            if self.hooks:
                await self.hooks.trigger("before_tool_call", {"tool": tool_name})
            await self._emit_runtime_event(
                HookEventType.TOOL_STARTED,
                session_key=session_key,
                run_id=run_id,
                payload={"tool_name": tool_name},
            )

            events.append(StreamEvent.tool_start(tool_name))
            events.append(StreamEvent.tool_end(tool_name))

            if self.hooks:
                await self.hooks.trigger("after_tool_call", {"tool": tool_name})
            await self._emit_runtime_event(
                HookEventType.TOOL_COMPLETED,
                session_key=session_key,
                run_id=run_id,
                payload={"tool_name": tool_name},
            )

            if self.queue:
                steer_messages = self.queue.get_steer_messages(session_key)
                if steer_messages:
                    combined = "\n".join(steer_messages)
                    events.append(StreamEvent.assistant_delta(f"\n[用户补充]: {combined}\n"))

        return ToolDispatchResult(events=events, tool_calls_count=tool_calls_count, should_break=False)

    async def _emit_runtime_event(
        self,
        event_type: HookEventType,
        *,
        session_key: str,
        run_id: str,
        payload: dict[str, Any],
    ) -> None:
        if self.hook_runtime is None:
            return
        parsed = SessionKey.from_string(session_key)
        await self.hook_runtime.emit(
            event_type=event_type,
            user_id=parsed.user_id,
            session_key=session_key,
            run_id=run_id,
            channel=parsed.channel,
            agent_id=parsed.agent_id,
            payload=payload,
        )

