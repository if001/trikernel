from __future__ import annotations

from typing import Any, Dict, List, Protocol

from .models import ToolContext, ToolDefinition
from .structured_tool import TrikernelStructuredTool


class ToolAPI(Protocol):
    def tool_register(self, tool_definition: ToolDefinition, handler: Any) -> None: ...

    def tool_register_structured(
        self, tool_definition: ToolDefinition, tool: TrikernelStructuredTool
    ) -> None: ...

    def tool_describe(self, tool_name: str) -> ToolDefinition: ...

    def tool_search(self, query: str) -> List[str]: ...

    def tool_invoke(
        self, tool_name: str, args: Dict[str, Any], tool_context: ToolContext
    ) -> Any: ...

    def tool_list(self) -> List[ToolDefinition]: ...

    def tool_descriptions(self) -> List[Dict[str, Any]]: ...

    def tool_structured_list(self) -> List[TrikernelStructuredTool]: ...


class ToolLLMAPI(Protocol):
    def generate(self, prompt: str, tools: List[Any] | None = None) -> str: ...
