from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from trikernel.tool_kernel.kernel import ToolKernel
from trikernel.tool_kernel.models import ToolContext


class AddArgs(BaseModel):
    x: int
    y: int


def add(x: int, y: int, *, context: ToolContext) -> int:
    _ = context
    return x + y


def test_tool_invoke_with_context():
    kernel = ToolKernel()
    tool = StructuredTool.from_function(
        func=lambda x, y: x + y,
        name="demo.add",
        description="Add numbers",
        args_schema=AddArgs,
    )
    kernel.tool_register(tool, handler=add)
    context = ToolContext(runner_id="test", task_id=None, state_api=None, now="now")
    result = kernel.tool_invoke("demo.add", {"x": 1, "y": 2}, tool_context=context)
    assert result == 3

