from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from langchain_core.messages import BaseMessage


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
