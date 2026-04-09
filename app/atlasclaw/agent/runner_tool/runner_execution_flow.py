from __future__ import annotations

from typing import Any, AsyncIterator

from app.atlasclaw.agent.runner_tool.runner_execution_flow_error import RunnerExecutionFlowErrorMixin
from app.atlasclaw.agent.runner_tool.runner_execution_flow_post import RunnerExecutionFlowPostMixin
from app.atlasclaw.agent.runner_tool.runner_execution_flow_stream import RunnerExecutionFlowStreamMixin
from app.atlasclaw.agent.stream import StreamEvent


class RunnerExecutionFlowPhaseMixin(
    RunnerExecutionFlowStreamMixin,
    RunnerExecutionFlowPostMixin,
    RunnerExecutionFlowErrorMixin,
):
    async def _run_loop_phase(self, *, state: dict[str, Any], _log_step: Any) -> AsyncIterator[StreamEvent]:
        """Main model/tool streaming loop phase."""
        deps = state.get("deps")
        user_message = state.get("user_message")
        message_history = list(state.get("message_history") or [])

        deps.user_message = user_message
        state["run_output_start_index"] = len(message_history)

        try:
            model_message_history = self.history.to_model_message_history(message_history)
            tool_gate_decision = state.get("tool_gate_decision")
            provider_fast_path_turn = bool(getattr(tool_gate_decision, "needs_external_system", False))
            async with self._run_iter_with_optional_override(
                agent=state.get("runtime_agent"),
                user_message=user_message,
                deps=deps,
                message_history=model_message_history,
                system_prompt=state.get("system_prompt"),
            ) as agent_run:
                async for event in self._run_agent_node_stream(
                    agent_run=agent_run,
                    state=state,
                    _log_step=_log_step,
                    provider_fast_path_turn=provider_fast_path_turn,
                ):
                    yield event

                thinking_emitter = state.get("thinking_emitter")
                if thinking_emitter is not None:
                    async for event in thinking_emitter.close_if_active():
                        yield event

                async for event in self._process_agent_run_outcome(
                    agent_run=agent_run,
                    state=state,
                    _log_step=_log_step,
                    provider_fast_path_turn=provider_fast_path_turn,
                ):
                    yield event

        except Exception as error:
            async for event in self._handle_loop_phase_exception(error=error, state=state):
                yield event

