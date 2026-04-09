from __future__ import annotations

from app.atlasclaw.agent.runner_tool.runner_tool_gate_cache import RunnerToolGateCacheMixin
from app.atlasclaw.agent.runner_tool.runner_tool_gate_model import RunnerToolGateModelMixin
from app.atlasclaw.agent.runner_tool.runner_tool_gate_policy import RunnerToolGatePolicyMixin
from app.atlasclaw.agent.runner_tool.runner_tool_gate_routing import RunnerToolGateRoutingMixin


class RunnerToolGateMixin(
    RunnerToolGateModelMixin,
    RunnerToolGatePolicyMixin,
    RunnerToolGateRoutingMixin,
    RunnerToolGateCacheMixin,
):
    """Composite mixin for tool-gate routing, policy, model and cache behaviors."""

    pass

