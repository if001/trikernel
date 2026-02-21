from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from .models import ToolDefinition
@dataclass(frozen=True)
class ToolRegistration:
    definition: ToolDefinition
    handler: Any


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
) -> List[ToolRegistration]:
    definitions = load_tool_definitions(path)
    tools = []
    for definition in definitions:
        handler = function_map[definition.tool_name]
        tools.append(ToolRegistration(definition=definition, handler=handler))
    return tools
