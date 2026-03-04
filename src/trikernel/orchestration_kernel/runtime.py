from __future__ import annotations

from typing import Any, Mapping

from langmem.utils import RunnableConfig

from ..state_kernel.protocols import StateKernelAPI
from ..tool_kernel.kernel import ToolKernel
from ..tool_kernel.runtime import ToolRuntime, register_runtime


def build_runnable_config(
    *,
    conversation_id: str,
    state_api: StateKernelAPI,
    tool_api: ToolKernel,
    recursion_limit: int | None = None,
) -> RunnableConfig:
    config: RunnableConfig = {
        "configurable": {
            "thread_id": conversation_id,
            "langgraph_user_id": conversation_id,
            "runtime_id": conversation_id,
            "state_api": state_api,
            "tool_api": tool_api,
        }
    }
    if recursion_limit is not None:
        config["recursion_limit"] = recursion_limit
    register_runtime(_runtime_from_config(config))
    return config


def _runtime_from_config(config: Mapping[str, Any]) -> ToolRuntime:
    configurable = config.get("configurable", {})
    runtime_id = str(
        configurable.get("runtime_id")
        or configurable.get("thread_id")
        or configurable.get("langgraph_user_id")
        or "default"
    )
    state_api = configurable.get("state_api")
    tool_api = configurable.get("tool_api")
    if state_api is None or tool_api is None:
        raise ValueError("state_api/tool_api are required in config")
    return ToolRuntime(
        runtime_id=runtime_id,
        state_api=state_api,
        tool_api=tool_api,
    )
