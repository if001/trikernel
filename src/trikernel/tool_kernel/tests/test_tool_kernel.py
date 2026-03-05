from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from trikernel.tool_kernel.kernel import ToolKernel
class AddArgs(BaseModel):
    x: int
    y: int


def add(payload: AddArgs) -> int:
    return payload.x + payload.y


def test_tool_register_and_list():
    kernel = ToolKernel()
    tool = StructuredTool.from_function(
        func=add,
        name="demo.add",
        description="Add numbers",
    )
    kernel.tool_register(tool)
    tools = kernel.tool_structured_list()
    assert tools
    assert tools[0].name == "demo.add"
