from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from langchain_core.tools import StructuredTool as LangchainStructuredTool


class TrikernelStructuredTool(Protocol):
    name: str
    description: str

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

    def invoke(self, args: dict[str, Any]) -> Any:
        return self.tool.invoke(args)

    def as_langchain(self) -> LangchainStructuredTool:
        return self.tool


def adapt_langchain_tool(
    tool: LangchainStructuredTool,
) -> TrikernelStructuredTool:
    return LangchainStructuredToolAdapter(tool)
