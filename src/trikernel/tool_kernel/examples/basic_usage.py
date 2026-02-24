from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from trikernel.tool_kernel import ToolKernel


class AddArgs(BaseModel):
    x: int
    y: int


def add(payload: AddArgs) -> int:
    return payload.x + payload.y


if __name__ == "__main__":
    kernel = ToolKernel()
    tool = StructuredTool.from_function(
        func=add,
        name="add",
        description="Add two numbers",
    )
    kernel.tool_register(tool)
    print([t.name for t in kernel.tool_list()])
