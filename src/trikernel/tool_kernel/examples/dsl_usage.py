from pathlib import Path

from trikernel.state_kernel.kernel import StateKernel
from trikernel.tool_kernel.dsl import build_tools_from_dsl
from trikernel.tool_kernel.models import ToolContext
from trikernel.tool_kernel.tools.state_tools import state_tool_functions
from trikernel.tool_kernel.kernel import ToolKernel


if __name__ == "__main__":
    state = StateKernel()
    kernel = ToolKernel()
    dsl_dir = Path(__file__).resolve().parents[1] / "dsl"
    state_dsl = dsl_dir / "state_tools.yaml"
    function_map = state_tool_functions()
    tools = build_tools_from_dsl(state_dsl, function_map)
    for tool in tools:
        kernel.tool_register(tool.definition, tool.handler)

    context = ToolContext(runner_id="example", task_id=None, state_api=state, now="now")
    task_id = kernel.tool_invoke(
        "task.create",
        {"task_type": "user_request", "payload": {"user_message": "from dsl"}},
        tool_context=context,
    )
    print(f"created task {task_id}")
