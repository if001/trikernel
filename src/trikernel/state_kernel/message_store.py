from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver

from .protocols import MessageStoreAPI


@dataclass(frozen=True)
class MessageStoreConfig:
    sqlite_path: Path


def load_message_store_config(data_dir: Optional[Path] = None) -> MessageStoreConfig:
    load_dotenv()
    base_dir = data_dir or Path(".state")
    sqlite_path = Path(
        os.environ.get(
            "TRIKERNEL_CHECKPOINT_PATH",
            str(base_dir / "checkpoints.sqlite"),
        )
    )
    return MessageStoreConfig(sqlite_path=sqlite_path)


class LangGraphMessageStore(MessageStoreAPI):
    def __init__(self, config: MessageStoreConfig) -> None:
        self._config = config
        self._config.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._checkpointer_cm = None
        try:
            conn = sqlite3.connect(
                str(self._config.sqlite_path), check_same_thread=False
            )
            self._checkpointer = SqliteSaver(conn)
        except TypeError:
            if not hasattr(SqliteSaver, "from_conn_string"):
                raise
            self._checkpointer_cm = SqliteSaver.from_conn_string(
                str(self._config.sqlite_path)
            )
            self._checkpointer = self._checkpointer_cm.__enter__()
        self.checkpointer = self._checkpointer


def load_message_store(data_dir: Optional[Path] = None) -> LangGraphMessageStore:
    config = load_message_store_config(data_dir)
    return LangGraphMessageStore(config)
