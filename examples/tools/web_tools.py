from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field
from typing_extensions import Annotated

from trikernel.tool_kernel.config import load_ollama_config
from trikernel.tool_kernel.runtime import get_state_api


@dataclass(frozen=True)
class WebClientConfig:
    base_url: str


class WebQueryArgs(BaseModel):
    user_message: str = Field(
        ..., description="User message to summarize into a query."
    )


class WebListArgs(BaseModel):
    q: str = Field(..., description="Query string.")
    k: int = Field(..., description="Number of results.")


class WebPageArgs(BaseModel):
    urls: str = Field(..., description="Comma-separated URLs.")


def load_web_client_config() -> WebClientConfig:
    load_dotenv()
    base_url = os.environ.get("SIMPLE_CLIENT_BASE_URL", "http://localhost:8000")
    return WebClientConfig(base_url=base_url)


def web_query(
    user_message: str,
    state: Annotated[dict, InjectedState] = {},
) -> str:
    history = _history_from_state(state)
    messages = _build_query_messages(user_message, history)
    config = load_ollama_config()
    payload_dict = {
        "model": config.small_model,
        "messages": messages,
        "stream": False,
    }
    response = _post_json(f"{config.base_url}/api/chat", payload_dict)
    message = response.get("message", {})
    content = message.get("content", "")
    return content.strip()


def web_list(
    q: str,
    k: int = 5,
    state: Annotated[dict, InjectedState] = {},
) -> Dict[str, Any]:
    _ = state
    config = load_web_client_config()
    payload_dict = {"q": q, "k": k}
    return _post_json(f"{config.base_url}/list", payload_dict)


def web_page(
    payload: WebPageArgs,
    state: Annotated[dict, InjectedState],
) -> Dict[str, Any]:
    _ = state
    config = load_web_client_config()
    payload_dict = {"urls": payload.urls}
    return _post_json(f"{config.base_url}/page", payload_dict)


def web_page_ref(
    payload: WebPageArgs,
    state: Annotated[dict, InjectedState],
) -> Dict[str, Any]:
    state_api = _require_state_api(state)
    response = web_page(payload, state=state)
    artifact_id = state_api.artifact_write(
        "application/json",
        json.dumps(response, ensure_ascii=False),
        {"source": "web.page", "urls": payload.urls},
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


def _require_state_api(state: dict) -> Any:
    state_api = state.get("state_api") if isinstance(state, dict) else None
    if state_api is None and isinstance(state, dict):
        runtime_id = state.get("runtime_id")
        if isinstance(runtime_id, str):
            state_api = get_state_api(runtime_id)
    if state_api is None:
        raise ValueError("state_api is required in state")
    return state_api


def build_web_tools() -> List[BaseTool]:
    return [
        StructuredTool.from_function(
            web_query,
            name="web.query",
            description="Generate a web search query from user message and history.",
            args_schema=WebQueryArgs,
        ),
        StructuredTool.from_function(
            web_list,
            name="web.list",
            description="Fetch a list of web search results.",
            args_schema=WebListArgs,
        ),
        # StructuredTool.from_function(
        #     web_page,
        #     name="web.page",
        #     description="Fetch web page content by URLs.",
        # ),
        StructuredTool.from_function(
            web_page_ref,
            name="web.page_ref",
            description="Fetch web pages and store content as an artifact.",
        ),
    ]
