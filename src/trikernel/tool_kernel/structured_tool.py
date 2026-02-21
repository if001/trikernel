from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from langchain_core.tools import ArgsSchema, StructuredTool as LangchainStructuredTool


class TrikernelStructuredTool(Protocol):
    name: str
    description: str
    args_schema: ArgsSchema | dict[str, Any]

    def invoke(self, args: dict[str, Any]) -> Any: ...

    def as_langchain(self) -> LangchainStructuredTool: ...


@dataclass(frozen=True)
class LangchainStructuredToolAdapter:
    tool: LangchainStructuredTool

    @property
    def name(self) -> str:
        return self.tool.name

    @property
    def description(self) -> str:
        return self.tool.description or ""

    @property
    def args_schema(self) -> ArgsSchema | dict[str, Any]:
        return self.tool.args_schema or {}

    def invoke(self, args: dict[str, Any]) -> Any:
        return self.tool.invoke(args)

    def as_langchain(self) -> LangchainStructuredTool:
        return self.tool


def adapt_langchain_tool(
    tool: LangchainStructuredTool,
) -> TrikernelStructuredTool:
    return LangchainStructuredToolAdapter(tool)
