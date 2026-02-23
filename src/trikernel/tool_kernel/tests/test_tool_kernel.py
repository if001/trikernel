from langchain_core.utils.function_calling import convert_to_openai_tool
from trikernel.state_kernel.kernel import StateKernel
from trikernel.tool_kernel.kernel import ToolKernel
from trikernel.tool_kernel.models import ToolContext, ToolDefinition
from pathlib import Path

from trikernel.tool_kernel.dsl import build_tools_from_dsl
from trikernel.tool_kernel.tools.state_tools import state_tool_functions
from trikernel.tool_kernel.langchain_tools import build_structured_tool
import json
import pytest


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
        kernel.tool_register(tool.definition, tool.handler)

    context = ToolContext(runner_id="test", task_id=None, state_api=state, now="now")
    task_id = kernel.tool_invoke(
        "task.create_user_request",
        {"payload": {"user_message": "hi"}},
        tool_context=context,
    )
    assert state.task_get(task_id) is not None


def test_task_create_missing_required_raises(tmp_path):
    state = StateKernel(data_dir=tmp_path)
    kernel = ToolKernel()
    for tool in build_tools_from_dsl(
        Path(__file__).resolve().parents[1] / "dsl" / "state_tools.yaml",
        state_tool_functions(),
    ):
        kernel.tool_register(tool.definition, tool.handler)

    context = ToolContext(runner_id="test", task_id=None, state_api=state, now="now")
    with pytest.raises(ValueError):
        kernel.tool_invoke(
            "task.create_user_request",
            {"payload": {}},
            tool_context=context,
        )


def test_build_tools_from_dsl(tmp_path):
    state = StateKernel(data_dir=tmp_path)
    kernel = ToolKernel()
    dsl_dir = Path(__file__).resolve().parents[1] / "dsl"
    state_dsl = dsl_dir / "state_tools.yaml"
    custom_dsl = tmp_path / "custom_tools.json"
    custom_dsl.write_text(
        json.dumps(
            {
                "tools": [
                    {
                        "tool_name": "demo.echo",
                        "description": "Echo input text",
                        "input_schema": {
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                            "required": ["text"],
                        },
                        "output_schema": {"type": "object", "properties": {}},
                    }
                ]
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    function_map = state_tool_functions()
    function_map["demo.echo"] = lambda text: {"text": text}
    tools = build_tools_from_dsl(state_dsl, function_map)
    tools += build_tools_from_dsl(custom_dsl, function_map)
    for tool in tools:
        kernel.tool_register(tool.definition, tool.handler)

    context = ToolContext(runner_id="test", task_id=None, state_api=state, now="now")
    task_id = kernel.tool_invoke(
        "task.create_user_request",
        {"payload": {"user_message": "dsl"}},
        tool_context=context,
    )
    assert state.task_get(task_id) is not None


def test_structured_tool_adapter_preserves_args_schema():
    def demo_tool(x: int, y: int = 0) -> int:
        return x + y

    definition = ToolDefinition(
        tool_name="demo.add",
        description="Add two numbers",
        input_schema={
            "type": "object",
            "properties": {
                "x": {"type": "integer"},
                "y": {"type": "integer"},
            },
            "required": ["x"],
        },
        output_schema={"type": "object", "properties": {}},
        effects=[],
    )
    structured = build_structured_tool(definition, demo_tool)
    langchain_tool = structured.as_langchain()
    schema = langchain_tool.args_schema.model_json_schema()
    required = schema.get("required") or []
    properties = schema.get("properties") or {}
    assert "x" in required
    assert "x" in properties
    assert "y" in properties


def test_dsl_arg_description_propagates_to_structured_tool(tmp_path):
    dsl_path = tmp_path / "desc_tool.yaml"
    dsl_path.write_text(
        json.dumps(
            {
                "tools": [
                    {
                        "tool_name": "demo.echo",
                        "description": "Echo input text",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "text": {
                                    "type": "string",
                                    "description": "Text to echo",
                                }
                            },
                            "required": ["text"],
                        },
                        "output_schema": {"type": "object", "properties": {}},
                    }
                ]
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    tools = build_tools_from_dsl(dsl_path, {"demo.echo": lambda text: text})
    definition = tools[0].definition
    assert (
        definition.input_schema["properties"]["text"]["description"] == "Text to echo"
    )

    structured = build_structured_tool(definition, tools[0].handler)
    schema = structured.as_langchain().args_schema.model_json_schema()
    assert schema["properties"]["text"]["description"] == "Text to echo"


def test_task_create_work_definition_and_structured_tool():
    dsl_path = Path(__file__).resolve().parents[1] / "dsl" / "state_tools.yaml"
    tools = build_tools_from_dsl(dsl_path, state_tool_functions())
    registration = next(
        tool for tool in tools if tool.definition.tool_name == "task.create_work"
    )
    definition = registration.definition
    assert definition.input_schema["properties"]["payload"]["required"] == ["message"]

    structured = build_structured_tool(definition, registration.handler)
    schema = structured.as_langchain().args_schema.model_json_schema()
    required = schema.get("required") or []
    assert "payload" in required
    assert "payload" in (schema.get("properties") or {})


def test_task_create_work_payload_descriptions_on_structured_tool():
    dsl_path = Path(__file__).resolve().parents[1] / "dsl" / "state_tools.yaml"
    tools = build_tools_from_dsl(dsl_path, state_tool_functions())
    registration = next(
        tool for tool in tools if tool.definition.tool_name == "task.create_work"
    )
    definition = registration.definition
    payload_props = definition.input_schema["properties"]["payload"]["properties"]
    assert (
        payload_props["message"]["description"]
        == "Work instruction message for the worker."
    )

    structured = build_structured_tool(definition, registration.handler)
    schema = structured.as_langchain().args_schema.model_json_schema()
    payload_schema = schema["properties"]["payload"]
    if "$ref" in payload_schema:
        ref_key = payload_schema["$ref"].split("/")[-1]
        payload_schema = schema.get("$defs", {}).get(ref_key, {})
    print("payload_schema", payload_schema)
    di = convert_to_openai_tool(structured.as_langchain())
    print("di", di)

    payload_fields = payload_schema["properties"]

    assert (
        payload_fields["message"]["description"]
        == "Work instruction message for the worker."
    )
