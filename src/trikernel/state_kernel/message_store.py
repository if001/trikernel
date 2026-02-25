from __future__ import annotations

import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..utils.env import load_env
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from .protocols import MessageStoreAPI


@dataclass(frozen=True)
class MessageStoreConfig:
    sqlite_path: Path


def load_message_store_config(data_dir: Optional[Path] = None) -> MessageStoreConfig:
    if data_dir is not None:
        sqlite_path = Path(data_dir) / "checkpoints.sqlite"
    else:
        load_env()
        base_dir = Path(".state")
        sqlite_path = Path(
            os.environ.get(
                "TRIKERNEL_CHECKPOINT_PATH",
                str(base_dir / "checkpoints.sqlite"),
            )
        )
    return MessageStoreConfig(sqlite_path=sqlite_path)


def _sqlite_conn_string(path: Path) -> str:
    return str(path)


class LangGraphMessageStore(MessageStoreAPI):
    def __init__(
        self,
        config: MessageStoreConfig,
        *,
        checkpointer: BaseCheckpointSaver,
    ) -> None:
        self._config = config
        self._config.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpointer = checkpointer


@asynccontextmanager
async def build_message_store(
    data_dir: Optional[Path] = None,
):
    config = load_message_store_config(data_dir)
    config.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    conn_string = _sqlite_conn_string(config.sqlite_path)
    checkpointer_cm = AsyncSqliteSaver.from_conn_string(conn_string)
    async with checkpointer_cm as checkpointer:
        await _maybe_setup(checkpointer)
        yield LangGraphMessageStore(
            config,
            checkpointer=checkpointer,
        )


async def _maybe_setup(checkpointer) -> None:
    maybe = checkpointer.setup()
    if hasattr(maybe, "__await__"):
        await maybe
