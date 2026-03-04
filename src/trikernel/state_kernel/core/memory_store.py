from __future__ import annotations

import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Optional

from langchain_ollama import OllamaEmbeddings
from langgraph.store.base import BaseStore, IndexConfig
from langgraph.store.sqlite.aio import AsyncSqliteStore

from ...utils.env import load_env
@dataclass(frozen=True)
class MemoryStoreConfig:
    sqlite_path: Path


def load_memory_store_config(data_dir: Optional[Path] = None) -> MemoryStoreConfig:
    if data_dir is not None:
        sqlite_path = Path(data_dir) / "memory_store.sqlite"
    else:
        load_env()
        base_dir = Path(".state")
        sqlite_path = Path(
            os.environ.get(
                "TRIKERNEL_MEMORY_STORE_PATH",
                str(base_dir / "memory_store.sqlite"),
            )
        )
    return MemoryStoreConfig(sqlite_path=sqlite_path)


def _sqlite_conn_string(path: Path) -> str:
    return str(path)


class MemoryStoreBuilder:
    def load_config(self, data_dir: Optional[Path] = None) -> MemoryStoreConfig:
        return load_memory_store_config(data_dir)

    @asynccontextmanager
    async def build(
        self,
        data_dir: Optional[Path] = None,
    ) -> AsyncIterator[BaseStore]:
        config = load_memory_store_config(data_dir)
        config.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        store_cm = AsyncSqliteStore.from_conn_string(
            _sqlite_conn_string(config.sqlite_path),
            index=_build_index_config(),
        )
        async with store_cm as store:
            maybe = store.setup()
            if hasattr(maybe, "__await__"):
                await maybe
            yield store


_builder = MemoryStoreBuilder()


@asynccontextmanager
async def build_memory_store(
    data_dir: Optional[Path] = None,
) -> AsyncIterator[BaseStore]:
    async with _builder.build(data_dir) as store:
        yield store


def _build_index_config() -> IndexConfig:
    load_env()
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    embed_model = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text:v1.5")
    embeddings = OllamaEmbeddings(model=embed_model, base_url=base_url)
    return {"dims": 768, "embed": embeddings}
