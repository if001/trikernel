from __future__ import annotations

from trikernel.tool_kernel.kernel import ToolKernel
from trikernel.tool_kernel.registry import register_default_tools

from tools.web_tools import build_web_tools


def main() -> None:
    kernel = ToolKernel()
    register_default_tools(kernel)

    for tool, handler in build_web_tools():
        kernel.tool_register(tool, handler)

    print("Registered web tools:", [tool.name for tool in kernel.tool_list()])


if __name__ == "__main__":
    main()
