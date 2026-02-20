from trikernel.tool_kernel import ToolContext, ToolDefinition, ToolKernel


def add(x: int, y: int) -> int:
    return x + y


if __name__ == "__main__":
    kernel = ToolKernel()
    definition = ToolDefinition(
        tool_name="add",
        description="Add two numbers",
        input_schema={
            "type": "object",
            "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
            "required": ["x", "y"],
        },
        output_schema={"type": "object", "properties": {}},
        effects=[],
    )
    kernel.tool_register(definition, add)
    context = ToolContext(runner_id="example", task_id=None, state_api=None, now="")
    result = kernel.tool_invoke("add", {"x": 2, "y": 3}, tool_context=context)
    print(result)
