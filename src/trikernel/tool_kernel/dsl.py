from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .langchain_tools import build_structured_tool
from .models import ToolDefinition
from .structured_tool import TrikernelStructuredTool


def load_tool_definitions(path: Path) -> List[ToolDefinition]:
    raw = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError("PyYAML is required to load YAML DSL files") from exc
        data = yaml.safe_load(raw)
    else:
        data = json.loads(raw)

    tools = data if isinstance(data, list) else data.get("tools", [])
    definitions = []
    for tool in tools:
        definitions.append(
            ToolDefinition(
                tool_name=tool["tool_name"],
                description=tool.get("description", ""),
                input_schema=tool.get(
                    "input_schema", {"type": "object", "properties": {}}
                ),
                output_schema=tool.get(
                    "output_schema", {"type": "object", "properties": {}}
                ),
                effects=tool.get("effects", []),
            )
        )
    return definitions


def build_tools_from_dsl(
    path: Path, function_map: Dict[str, Any]
) -> List[TrikernelStructuredTool]:
    definitions = load_tool_definitions(path)
    tools = []
    for definition in definitions:
        handler = function_map[definition.tool_name]
        tools.append(build_structured_tool(definition, handler))
    return tools
