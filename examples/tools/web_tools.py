from __future__ import annotations

import os
import json
import urllib.error
import urllib.request
from typing import Any, Dict, List
from dataclasses import dataclass
from dotenv import load_dotenv

from trikernel.tool_kernel.config import load_ollama_config
from trikernel.tool_kernel.models import ToolContext


@dataclass(frozen=True)
class WebClientConfig:
    base_url: str


def load_web_client_config() -> WebClientConfig:
    load_dotenv()
    base_url = os.environ.get("SIMPLE_CLIENT_BASE_URL", "http://localhost:8000")
    return WebClientConfig(base_url=base_url)


def web_query(
    user_message: str,
    conversation_id: str,
    limit: int,
    *,
    context: ToolContext,
) -> str:
    state_api = _require_state_api(context)
    history = state_api.turn_list_recent(conversation_id, limit)
    messages = _build_query_messages(user_message, history)
    config = load_ollama_config()
    payload = {
        "model": config.small_model,
        "messages": messages,
        "stream": False,
    }
    response = _post_json(f"{config.base_url}/api/chat", payload)
    message = response.get("message", {})
    content = message.get("content", "")
    return content.strip()


def web_list(q: str, k: int, *, context: ToolContext) -> Dict[str, Any]:
    _require_state_api(context)
    config = load_web_client_config()
    payload = {"q": q, "k": k}
    return _post_json(f"{config.base_url}/list", payload)


def web_page(urls: str, *, context: ToolContext) -> Dict[str, Any]:
    _require_state_api(context)
    config = load_web_client_config()
    payload = {"urls": urls}
    return _post_json(f"{config.base_url}/page", payload)


def _build_query_messages(
    user_message: str, history: List[Any]
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": "Create a concise web search query based on the user message and recent history.",
        }
    ]
    for turn in history:
        if getattr(turn, "user_message", None):
            messages.append({"role": "user", "content": turn.user_message})
        if getattr(turn, "assistant_message", None):
            messages.append({"role": "assistant", "content": turn.assistant_message})
    messages.append({"role": "user", "content": user_message})
    return messages


def _post_json(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8") if exc.fp else ""
        raise RuntimeError(f"HTTP {exc.code}: {error_body}") from exc
    return json.loads(body)


def _require_state_api(context: ToolContext) -> Any:
    if context is None or context.state_api is None:
        raise ValueError("state_api is required in ToolContext")
    return context.state_api


def web_tool_functions() -> Dict[str, Any]:
    return {
        "web.query": web_query,
        "web.list": web_list,
        "web.page": web_page,
    }
