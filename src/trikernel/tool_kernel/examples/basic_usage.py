from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from trikernel.tool_kernel import ToolContext, ToolKernel


class AddArgs(BaseModel):
    x: int
    y: int


def add(x: int, y: int, *, context: ToolContext) -> int:
    _ = context
    return x + y


if __name__ == "__main__":
    kernel = ToolKernel()
    tool = StructuredTool.from_function(
        func=lambda x, y: x + y,
        name="add",
        description="Add two numbers",
        args_schema=AddArgs,
    )
    kernel.tool_register(tool, handler=add)
    context = ToolContext(runner_id="example", task_id=None, state_api=None, now="")
    result = kernel.tool_invoke("add", {"x": 2, "y": 3}, tool_context=context)
    print(result)
