from __future__ import annotations

from threading import Lock
from typing import Optional

from ..state_kernel.protocols import StateKernelAPI
from .protocols import ToolLLMBase


_state_lock = Lock()
_state_apis: dict[str, StateKernelAPI] = {}
_llm_apis: dict[str, ToolLLMBase] = {}


def register_runtime(runtime_id: str, state_api: StateKernelAPI, llm_api: ToolLLMBase) -> None:
    with _state_lock:
        _state_apis[runtime_id] = state_api
        _llm_apis[runtime_id] = llm_api


def get_state_api(runtime_id: str) -> Optional[StateKernelAPI]:
    with _state_lock:
        return _state_apis.get(runtime_id)


def get_llm_api(runtime_id: str) -> Optional[ToolLLMBase]:
    with _state_lock:
        return _llm_apis.get(runtime_id)
