from __future__ import annotations

import os
import json
import urllib.error
import urllib.request
from typing import Any, Dict, List
from dataclasses import dataclass
from dotenv import load_dotenv

from trikernel.tool_kernel.config import load_ollama_config
from langchain_core.tools import BaseTool
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel

from trikernel.tool_kernel.models import ToolContext
from trikernel.tool_kernel.tools.structured_tools import build_structured_tool
from typing_extensions import Annotated


@dataclass(frozen=True)
class WebClientConfig:
    base_url: str


def load_web_client_config() -> WebClientConfig:
    load_dotenv()
    base_url = os.environ.get("SIMPLE_CLIENT_BASE_URL", "http://localhost:8000")
    return WebClientConfig(base_url=base_url)


def web_query(
    user_message: str,
    state: Annotated[dict, InjectedState],
    *,
    context: ToolContext,
) -> str:
    _ = context
    history = _history_from_state(state)
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
    _ = context
    config = load_web_client_config()
    payload = {"q": q, "k": k}
    return _post_json(f"{config.base_url}/list", payload)


def web_page(urls: str, *, context: ToolContext) -> Dict[str, Any]:
    _ = context
    config = load_web_client_config()
    payload = {"urls": urls}
    return _post_json(f"{config.base_url}/page", payload)


def web_page_ref(urls: str, *, context: ToolContext) -> Dict[str, Any]:
    state_api = _require_state_api(context)
    response = web_page(urls, context=context)
    artifact_id = state_api.artifact_write(
        "application/json",
        json.dumps(response, ensure_ascii=False),
        {"source": "web.page", "urls": urls},
    )
    return {"artifact_id": artifact_id}


def _build_query_messages(
    user_message: str, history: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": "Create a concise web search query based on the user message and recent context.",
        }
    ]
    messages.extend(history)
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


def _history_from_state(state: dict, limit: int = 6) -> List[Dict[str, str]]:
    messages = []
    for msg in state.get("messages", [])[-limit:]:
        role = getattr(msg, "type", None)
        content = getattr(msg, "content", None)
        if isinstance(role, str) and isinstance(content, str):
            if role == "human":
                messages.append({"role": "user", "content": content})
            elif role == "ai":
                messages.append({"role": "assistant", "content": content})
    return messages


def _require_state_api(context: ToolContext) -> Any:
    if context is None or context.state_api is None:
        raise ValueError("state_api is required in ToolContext")
    return context.state_api


class WebQueryArgs(BaseModel):
    user_message: str


class WebListArgs(BaseModel):
    q: str
    k: int


class WebPageArgs(BaseModel):
    urls: str


def build_web_tools() -> List[tuple[BaseTool, Any]]:
    return [
        (
            build_structured_tool(
                web_query,
                name="web.query",
                description="Generate a web search query from user message and history.",
                args_schema=WebQueryArgs,
            ),
            web_query,
        ),
        (
            build_structured_tool(
                web_list,
                name="web.list",
                description="Fetch a list of web search results.",
                args_schema=WebListArgs,
            ),
            web_list,
        ),
        (
            build_structured_tool(
                web_page,
                name="web.page",
                description="Fetch web page content by URLs.",
                args_schema=WebPageArgs,
            ),
            web_page,
        ),
        (
            build_structured_tool(
                web_page_ref,
                name="web.page_ref",
                description="Fetch web pages and store content as an artifact.",
                args_schema=WebPageArgs,
            ),
            web_page_ref,
        ),
    ]
