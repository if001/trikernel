from __future__ import annotations

from pathlib import Path

from .dsl import build_tools_from_dsl
from .kernel import ToolKernel
from .tools.state_tools import state_tool_functions
from .tools.system_tools import system_tool_functions
from .tools.writing_tools import writing_tool_functions
from .tools.user_profile_tools import user_profile_tool_functions
from .tools.file_tools import file_tool_functions


def register_default_tools(kernel: ToolKernel) -> None:
    dsl_dir = Path(__file__).resolve().parent / "dsl"
    state_dsl = dsl_dir / "state_tools.yaml"
    system_dsl = dsl_dir / "system_tools.yaml"
    writing_dsl = dsl_dir / "writing_tools.yaml"
    profile_dsl = dsl_dir / "user_profile_tools.yaml"
    file_dsl = dsl_dir / "file_tools.yaml"
    function_map = state_tool_functions()
    system_tool_map = system_tool_functions()
    writing_tool_map = writing_tool_functions()
    profile_tool_map = user_profile_tool_functions()
    file_tool_map = file_tool_functions()
    tools = build_tools_from_dsl(state_dsl, function_map)
    tools += build_tools_from_dsl(system_dsl, system_tool_map)
    tools += build_tools_from_dsl(writing_dsl, writing_tool_map)
    tools += build_tools_from_dsl(profile_dsl, profile_tool_map)
    tools += build_tools_from_dsl(file_dsl, file_tool_map)
    for tool in tools:
        kernel.tool_register_structured(tool)
