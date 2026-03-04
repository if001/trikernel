from __future__ import annotations

from typing import Protocol

from langgraph.checkpoint.base import BaseCheckpointSaver

Checkpointer = BaseCheckpointSaver


class MessageStoreProtocol(Protocol):
    checkpointer: Checkpointer

