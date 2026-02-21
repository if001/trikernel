from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

from .langchain_tools import build_structured_tool
from .models import ToolContext, ToolDefinition
from .protocols import ToolAPI
from .structured_tool import TrikernelStructuredTool, adapt_langchain_tool
from .validation import validate_input
from ..utils.search import HybridSearchIndex


@dataclass
class ToolEntry:
    definition: ToolDefinition
    handler: Optional[Any]
    structured_tool: Optional[TrikernelStructuredTool]


class ToolKernel(ToolAPI):
    def __init__(self, data_dir: Optional[Path] = None) -> None:
        if data_dir is None:
            data_dir = Path(".state")
        self._tools: Dict[str, ToolEntry] = {}
        self._search_index = _init_tool_search(data_dir)

    def tool_register(
        self,
        tool_definition: ToolDefinition,
        handler: Any,
    ) -> None:
        self._tools[tool_definition.tool_name] = ToolEntry(
            definition=tool_definition,
            handler=handler,
            structured_tool=None,
        )
        self._index_tool(tool_definition, force=False)

    def tool_register_structured(
        self, tool_definition: ToolDefinition, tool: TrikernelStructuredTool
    ) -> None:
        structured_tool = tool
        if not hasattr(structured_tool, "as_langchain"):
            structured_tool = adapt_langchain_tool(structured_tool)  # type: ignore[arg-type]
        handler = _extract_handler(structured_tool)
        self._tools[tool_definition.tool_name] = ToolEntry(
            definition=tool_definition,
            handler=handler,
            structured_tool=structured_tool,
        )
        self._index_tool(tool_definition, force=False)

    def tool_describe(self, tool_name: str) -> ToolDefinition:
        return self._tools[tool_name].definition

    def tool_search(self, query: str) -> List[str]:
        if not query:
            return list(self._tools.keys())
        if not self._tools:
            return []
        docs = self._search_index.search(query, k=min(10, len(self._tools)))
        if docs:
            return [
                tool_name
                for tool_name in (
                    doc.metadata.get("tool_name", "") for doc in docs if doc.metadata
                )
                if tool_name
            ]
        query_lower = query.lower()
        return [
            name
            for name in self._tools.keys()
            if query_lower in name.lower()
            or query_lower in self._tools[name].definition.description.lower()
        ]

    def tool_invoke(
        self, tool_name: str, args: Dict[str, Any], tool_context: ToolContext
    ) -> Any:
        entry = self._tools[tool_name]
        validate_input(entry.definition.input_schema, args)
        if entry.handler:
            return _invoke_handler(entry.handler, args, tool_context)
        if entry.structured_tool:
            return entry.structured_tool.invoke(args)
        raise ValueError(f"Tool '{tool_name}' has no handler.")

    def tool_list(self) -> List[ToolDefinition]:
        return [tool.definition for tool in self._tools.values()]

    def tool_descriptions(self) -> List[Dict[str, Any]]:
        return [
            {
                "tool_name": tool.tool_name,
                "description": tool.description,
                "input_schema": tool.input_schema,
                "output_schema": tool.output_schema,
                "effects": tool.effects,
            }
            for tool in self.tool_list()
        ]

    def tool_structured_list(self) -> List[TrikernelStructuredTool]:
        tools: List[TrikernelStructuredTool] = []
        for entry in self._tools.values():
            structured = entry.structured_tool
            if structured is None and entry.handler:
                structured = build_structured_tool(entry.definition, entry.handler)
                entry.structured_tool = structured
            if structured:
                tools.append(structured)
        return tools

    def _index_tool(self, definition: ToolDefinition, *, force: bool = False) -> None:
        metadata = {"tool_name": definition.tool_name}
        description = definition.description or definition.tool_name
        metadata["id"] = definition.tool_name
        doc = Document(page_content=description, metadata=metadata)
        self._search_index.upsert_document(doc, definition.tool_name, force=force)


def _init_tool_search(data_dir: Path) -> HybridSearchIndex:
    load_dotenv()
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    embed_model = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    embeddings = OllamaEmbeddings(model=embed_model, base_url=base_url)
    persist_dir = data_dir / "search_tools"
    return HybridSearchIndex(persist_dir, "tools", embeddings)


def _extract_handler(tool: TrikernelStructuredTool) -> Optional[Any]:
    tool_impl = tool.as_langchain()
    handler = getattr(tool_impl, "func", None)
    if handler:
        return handler
    handler = getattr(tool_impl, "coroutine", None)
    if handler:
        return handler
    return getattr(tool_impl, "_run", None)


def _invoke_handler(handler: Any, args: Dict[str, Any], context: ToolContext) -> Any:
    signature = inspect.signature(handler)
    params = signature.parameters
    if "context" in params:
        return handler(**args, context=context)
    if "tool_context" in params:
        return handler(**args, tool_context=context)
    return handler(**args)
