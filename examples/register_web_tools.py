from __future__ import annotations

import asyncio

from trikernel.state_kernel.memory_store import build_memory_store
from trikernel.tool_kernel.kernel import ToolKernel
from trikernel.tool_kernel.registry import register_default_tools

from tools.web_tools import build_web_tools


async def main() -> None:
    kernel = ToolKernel()
    async with build_memory_store() as store:
        register_default_tools(kernel, store=store)
        for tool in build_web_tools():
            kernel.tool_register(tool)
        print("Registered web tools:", [tool.name for tool in kernel.tool_list()])


if __name__ == "__main__":
    asyncio.run(main())
