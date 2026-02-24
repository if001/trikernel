from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.tools import BaseTool
from langchain_ollama import OllamaEmbeddings

from ..utils.search import HybridSearchIndex


@dataclass
class ToolEntry:
    tool: BaseTool


class ToolKernel:
    def __init__(self, data_dir: Optional[Path] = None, re_index: bool = False) -> None:
        if data_dir is None:
            data_dir = Path(".state")
        self._tools: Dict[str, ToolEntry] = {}
        self._search_index = _init_tool_search(data_dir)
        self._re_index = re_index

    def tool_register(
        self,
        tool: BaseTool,
    ) -> None:
        self._tools[tool.name] = ToolEntry(
            tool=tool,
        )
        self._index_tool(tool, force=self._re_index)

    def tool_describe(self, tool_name: str) -> BaseTool:
        return self._tools[tool_name].tool

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
            or query_lower in (self._tools[name].tool.description or "").lower()
        ]

    def tool_list(self) -> List[BaseTool]:
        return [tool.tool for tool in self._tools.values()]

    def tool_descriptions(self) -> List[Dict[str, Any]]:
        return [
            {
                "tool_name": tool.name,
                "description": tool.description or "",
            }
            for tool in self.tool_list()
        ]

    def tool_structured_list(self) -> List[BaseTool]:
        return [entry.tool for entry in self._tools.values()]

    def _index_tool(self, tool: BaseTool, *, force: bool = False) -> None:
        metadata = {"tool_name": tool.name, "id": tool.name}
        description = tool.description or tool.name
        doc = Document(page_content=description, metadata=metadata)
        self._search_index.upsert_document(doc, tool.name, force=force)


def _init_tool_search(data_dir: Path) -> HybridSearchIndex:
    load_dotenv()
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    embed_model = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    embeddings = OllamaEmbeddings(model=embed_model, base_url=base_url)
    persist_dir = data_dir / "search_tools"
    return HybridSearchIndex(persist_dir, "tools", embeddings)

