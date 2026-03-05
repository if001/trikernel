from __future__ import annotations

from langgraph.prebuilt import InjectedState
from typing_extensions import Annotated

from ..runtime import ToolRuntime, get_runtime


def require_runtime(state: Annotated[dict, InjectedState]) -> ToolRuntime:
    runtime_id = state.get("runtime_id") if isinstance(state, dict) else None
    if not isinstance(runtime_id, str) or not runtime_id:
        raise ValueError("runtime_id is required in state")
    runtime = get_runtime(runtime_id)
    if runtime is None:
        raise ValueError("runtime is required in tool runtime registry")
    return runtime


def require_state_api(state: Annotated[dict, InjectedState]):
    return require_runtime(state).state_api


def require_tool_llm(state: Annotated[dict, InjectedState]):
    return require_runtime(state).tool_api.tool_llm_api()
