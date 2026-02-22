from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

from .models import ToolDefinition
from .structured_tool import TrikernelStructuredTool, adapt_langchain_tool


def build_structured_tool(
    definition: ToolDefinition, handler: Any
) -> TrikernelStructuredTool:
    args_schema = _build_args_schema(definition.tool_name, definition.input_schema)
    tool = StructuredTool.from_function(
        func=handler,
        name=definition.tool_name,
        description=definition.description,
        args_schema=args_schema,
    )
    return adapt_langchain_tool(tool)


def tool_definition_from_structured(tool: TrikernelStructuredTool) -> ToolDefinition:
    input_schema = _schema_from_tool(tool.as_langchain())
    return ToolDefinition(
        tool_name=tool.name,
        description=tool.description or "",
        input_schema=input_schema,
        output_schema={"type": "object", "properties": {}},
        effects=[],
    )


def _build_args_schema(
    tool_name: str, input_schema: Dict[str, Any]
) -> Optional[Type[BaseModel]]:
    if input_schema is None:
        return None
    properties = input_schema.get("properties") or {}
    required = set(input_schema.get("required") or [])
    fields: Dict[str, Tuple[Any, Any]] = {}
    for prop, spec in properties.items():
        field_type = _json_schema_type_to_python(spec)
        description = spec.get("description")
        if prop in required:
            fields[prop] = (field_type, Field(..., description=description))
        else:
            fields[prop] = (Optional[field_type], Field(default=None, description=description))
    model_name = f"{_safe_class_name(tool_name)}Args"
    return create_model(model_name, **fields)  # type: ignore[arg-type]


def _schema_from_tool(tool: StructuredTool) -> Dict[str, Any]:
    args_schema = getattr(tool, "args_schema", None)
    if not args_schema:
        return {"type": "object", "properties": {}}
    if hasattr(args_schema, "model_json_schema"):
        return args_schema.model_json_schema()
    if hasattr(args_schema, "schema"):
        return args_schema.schema()
    return {"type": "object", "properties": {}}


def _json_schema_type_to_python(spec: Dict[str, Any]) -> Any:
    raw_type = spec.get("type")
    if isinstance(raw_type, list):
        non_null = [t for t in raw_type if t != "null"]
        raw_type = non_null[0] if non_null else None
    if raw_type == "string":
        return str
    if raw_type == "integer":
        return int
    if raw_type == "number":
        return float
    if raw_type == "boolean":
        return bool
    if raw_type == "array":
        return List[Any]
    if raw_type == "object":
        return Dict[str, Any]
    return Any


def _safe_class_name(name: str) -> str:
    cleaned = "".join(ch for ch in name.title() if ch.isalnum())
    return cleaned or "Tool"
