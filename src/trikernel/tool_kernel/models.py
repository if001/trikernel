from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..state_kernel.protocols import MessageStoreAPI


@dataclass
class ToolContext:
    runner_id: str
    task_id: str | None
    state_api: Any
    now: str
    llm_api: Optional[Any] = None
    message_store: Optional["MessageStoreAPI"] = None


@dataclass
class ToolCall:
    tool_name: str
    args: Dict[str, Any]
