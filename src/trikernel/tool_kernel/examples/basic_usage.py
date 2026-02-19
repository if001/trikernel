from trikernel.tool_kernel.kernel import ToolKernel
from langchain_core.tools import StructuredTool


def add(x: int, y: int) -> int:
    return x + y


if __name__ == "__main__":
    kernel = ToolKernel()
    tool = StructuredTool.from_function(add, name="add", description="Add two numbers")
    kernel.tool_register_structured(tool)
    result = kernel.tool_invoke("add", {"x": 2, "y": 3}, tool_context=None)
    print(result)
