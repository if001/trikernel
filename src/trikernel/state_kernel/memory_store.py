from __future__ import annotations

import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Optional

from dotenv import load_dotenv
from langgraph.store.base import BaseStore
from langgraph.store.sqlite.aio import AsyncSqliteStore


@dataclass(frozen=True)
class MemoryStoreConfig:
    sqlite_path: Path


def load_memory_store_config(data_dir: Optional[Path] = None) -> MemoryStoreConfig:
    load_dotenv()
    base_dir = data_dir or Path(".state")
    sqlite_path = Path(
        os.environ.get(
            "TRIKERNEL_MEMORY_STORE_PATH",
            str(base_dir / "memory_store.sqlite"),
        )
    )
    return MemoryStoreConfig(sqlite_path=sqlite_path)


def _sqlite_conn_string(path: Path) -> str:
    return f"sqlite+aiosqlite:///{path}"


@asynccontextmanager
async def build_memory_store(
    data_dir: Optional[Path] = None,
) -> AsyncIterator[BaseStore]:
    config = load_memory_store_config(data_dir)
    store_cm = AsyncSqliteStore.from_conn_string(_sqlite_conn_string(config.sqlite_path))
    async with store_cm as store:
        maybe = store.setup()
        if hasattr(maybe, "__await__"):
            await maybe
        yield store
