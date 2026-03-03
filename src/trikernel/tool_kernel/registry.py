from __future__ import annotations

from typing import Iterable

from langchain_core.tools import BaseTool

from .kernel import ToolKernel
from langgraph.store.base import BaseStore
from .tools.file_tools import build_file_tools
from .tools.memory_tools import build_memory_tools
from .tools.state_tools import build_state_tools
from .tools.system_tools import build_system_tools
from .tools.writing_tools import build_writing_tools


def register_default_tools(kernel: ToolKernel) -> None:
    tools: Iterable[BaseTool] = (
        build_state_tools()
        + build_system_tools()
        + build_writing_tools()
        + build_file_tools()
        # + build_memory_tools(store)
    )
    for tool in tools:
        kernel.tool_register(tool)


def register_deep_agent_tools(kernel: ToolKernel, *, store: BaseStore) -> None:
    tools: Iterable[BaseTool] = build_state_tools()
    for tool in tools:
        kernel.tool_register(tool)
