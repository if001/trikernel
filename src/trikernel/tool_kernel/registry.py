from __future__ import annotations

from pathlib import Path

from .dsl import build_tools_from_dsl
from .kernel import ToolKernel
from .state_tools import state_tool_functions
from .system_tools import system_tool_functions
from .web_tools import web_tool_functions


def register_default_tools(kernel: ToolKernel) -> None:
    dsl_dir = Path(__file__).resolve().parent / "dsl"
    state_dsl = dsl_dir / "state_tools.yaml"
    web_dsl = dsl_dir / "web_tools.yaml"
    system_dsl = dsl_dir / "system_tools.yaml"
    function_map = state_tool_functions()
    web_tool_map = web_tool_functions()
    system_tool_map = system_tool_functions()
    tools = build_tools_from_dsl(state_dsl, function_map)
    tools += build_tools_from_dsl(web_dsl, web_tool_map)
    tools += build_tools_from_dsl(system_dsl, system_tool_map)
    for tool in tools:
        kernel.tool_register_structured(tool)
