from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from langchain_core.messages import BaseMessage

from ..state_kernel.models import Turn


@dataclass(frozen=True)
class LLMInput:
    message: Optional[str] = None
    messages: Optional[Sequence[BaseMessage]] = None
    history: Optional[Sequence[Turn | Dict[str, Any] | BaseMessage]] = None


def build_llm_payload(
    message: Optional[str] = None,
    messages: Optional[Sequence[BaseMessage]] = None,
    history: Optional[Sequence[Turn | Dict[str, Any] | BaseMessage]] = None,
) -> Dict[str, Any]:
    llm_input: Dict[str, Any] = {}
    if message is not None:
        llm_input["message"] = message
    if messages is not None:
        llm_input["messages"] = list(messages)
    if history is not None:
        llm_input["history"] = [
            item.to_dict() if hasattr(item, "to_dict") else item for item in history
        ]
    return {"llm_input": llm_input}


def extract_llm_input(payload: Dict[str, Any]) -> Dict[str, Any]:
    return payload.get("llm_input") or payload


def extract_user_message(payload: Dict[str, Any]) -> str:
    if "user_message" in payload:
        return str(payload.get("user_message") or "")
    if "message" in payload:
        return str(payload.get("message") or "")
    if "prompt" in payload:
        return str(payload.get("prompt") or "")
    return ""
