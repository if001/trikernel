from __future__ import annotations

from pathlib import Path

from trikernel.tool_kernel.dsl import build_tools_from_dsl
from trikernel.tool_kernel.kernel import ToolKernel
from trikernel.tool_kernel.registry import register_default_tools

from tools.web_tools import web_list, web_page, web_page_ref, web_query


def build_web_tools():
    dsl_path = Path(__file__).resolve().parent / "tools" / "web_tools.yaml"
    function_map = {
        "web.query": web_query,
        "web.list": web_list,
        "web.page": web_page,
        "web.page_ref": web_page_ref,
    }
    tools = build_tools_from_dsl(dsl_path, function_map)
    return tools


def main() -> None:
    kernel = ToolKernel()
    register_default_tools(kernel)

    tools = build_web_tools()
    for tool in tools:
        kernel.tool_register_structured(tool)

    print("Registered web tools:", [tool.tool_name for tool in kernel.tool_list()])


if __name__ == "__main__":
    main()
