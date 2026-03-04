from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Optional, Protocol

from trikernel.tool_kernel.protocols import ToolLLMBase

from ..state_kernel.protocols import StateKernelAPI


class ToolAPIProtocol(Protocol):
    def tool_structured_list(self): ...
    def tool_search(self, query: str): ...
    def tool_descriptions(self): ...
    def tool_llm_api(self) -> ToolLLMBase: ...


@dataclass(frozen=True)
class ToolRuntime:
    runtime_id: str
    state_api: StateKernelAPI
    tool_api: ToolAPIProtocol


_state_lock = Lock()
_runtimes: dict[str, ToolRuntime] = {}


def register_runtime(runtime: ToolRuntime) -> None:
    with _state_lock:
        _runtimes[runtime.runtime_id] = runtime


def get_runtime(runtime_id: str) -> Optional[ToolRuntime]:
    with _state_lock:
        return _runtimes.get(runtime_id)
