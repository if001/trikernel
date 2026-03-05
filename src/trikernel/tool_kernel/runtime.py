from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Optional

from ..state_kernel.protocols import StateKernelAPI
from .kernel import ToolKernel


@dataclass(frozen=True)
class ToolRuntime:
    runtime_id: str
    state_api: StateKernelAPI
    tool_api: ToolKernel


_state_lock = Lock()
_runtimes: dict[str, ToolRuntime] = {}


def register_runtime(runtime: ToolRuntime) -> None:
    with _state_lock:
        _runtimes[runtime.runtime_id] = runtime


def get_runtime(runtime_id: str) -> Optional[ToolRuntime]:
    with _state_lock:
        return _runtimes.get(runtime_id)
