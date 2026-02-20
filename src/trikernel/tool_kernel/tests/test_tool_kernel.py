from trikernel.state_kernel.kernel import StateKernel
from trikernel.tool_kernel.kernel import ToolKernel
from trikernel.tool_kernel.models import ToolContext, ToolDefinition
from pathlib import Path

from trikernel.tool_kernel.dsl import build_tools_from_dsl
from trikernel.tool_kernel.state_tools import state_tool_functions
from trikernel.tool_kernel.web_tools import web_list, web_page, web_query


def add(x: int, y: int) -> int:
    return x + y


def test_tool_invoke():
    kernel = ToolKernel()
    kernel.tool_register(
        ToolDefinition(
            tool_name="add",
            description="Add",
            input_schema={
                "type": "object",
                "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                "required": ["x", "y"],
            },
            output_schema={"type": "object", "properties": {}},
            effects=[],
        ),
        add,
    )
    result = kernel.tool_invoke("add", {"x": 1, "y": 2}, tool_context=None)
    assert result == 3


def test_state_tools_task_create(tmp_path):
    state = StateKernel(data_dir=tmp_path)
    kernel = ToolKernel()
    for tool in build_tools_from_dsl(
        Path(__file__).resolve().parents[1] / "dsl" / "state_tools.yaml",
        state_tool_functions(),
    ):
        kernel.tool_register_structured(tool)

    context = ToolContext(runner_id="test", task_id=None, state_api=state, now="now")
    task_id = kernel.tool_invoke(
        "task.create",
        {"task_type": "user_request", "payload": {"user_message": "hi"}},
        tool_context=context,
    )
    assert state.task_get(task_id) is not None


def test_build_tools_from_dsl(tmp_path):
    state = StateKernel(data_dir=tmp_path)
    kernel = ToolKernel()
    dsl_dir = Path(__file__).resolve().parents[1] / "dsl"
    state_dsl = dsl_dir / "state_tools.yaml"
    web_dsl = dsl_dir / "web_tools.yaml"
    function_map = state_tool_functions()
    function_map.update(
        {
            "web.query": web_query,
            "web.list": web_list,
            "web.page": web_page,
        }
    )
    tools = build_tools_from_dsl(state_dsl, function_map)
    tools += build_tools_from_dsl(web_dsl, function_map)
    for tool in tools:
        kernel.tool_register_structured(tool)

    context = ToolContext(runner_id="test", task_id=None, state_api=state, now="now")
    task_id = kernel.tool_invoke(
        "task.create",
        {"task_type": "user_request", "payload": {"user_message": "dsl"}},
        tool_context=context,
    )
    assert state.task_get(task_id) is not None
