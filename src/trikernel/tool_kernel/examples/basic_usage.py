from trikernel.tool_kernel.kernel import ToolKernel
from trikernel.tool_kernel.structured_tool import StructuredTool


def add(x: int, y: int) -> int:
    return x + y


if __name__ == "__main__":
    kernel = ToolKernel()
    tool = StructuredTool.from_function(add, description="Add two numbers")
    kernel.tool_register_structured(tool)
    result = kernel.tool_invoke("add", {"x": 2, "y": 3}, tool_context=None)
    print(result)
