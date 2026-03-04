from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain.tools import InjectedToolArg
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.prebuilt import InjectedState
from pydantic import Field
from typing_extensions import Annotated

from trikernel.tool_kernel.runtime import ToolRuntime, get_runtime
from langchain.tools import ToolRuntime as LCToolRuntime


@dataclass(frozen=True)
class WebClientConfig:
    base_url: str


def load_web_client_config() -> WebClientConfig:
    load_dotenv()
    base_url = os.environ.get("SIMPLE_CLIENT_BASE_URL", "http://localhost:8000")
    return WebClientConfig(base_url=base_url)


def web_query(
    user_message: str = Field(
        ..., description="User message to summarize into a query."
    ),
    state: Annotated[dict, InjectedState] = {},
) -> str:
    history = _history_from_state(state)
    messages = _build_query_messages(user_message, history)
    prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    runtime = _require_runtime(state)
    response_text = runtime.tool_api.tool_llm_api().generate(prompt, [])
    return response_text.strip()


def web_list(
    q: str = Field(..., description="Query string."),
    k: Optional[int] = Field(..., description="Number of results."),
    state: Annotated[dict, InjectedState] = {},
) -> Dict[str, Any]:
    _ = state
    config = load_web_client_config()
    payload_dict = {"q": q, "k": k}
    return _post_json(f"{config.base_url}/list", payload_dict)


def web_list_ref(
    q: str = Field(..., description="Query string."),
    k: Optional[int] = Field(..., description="Number of results."),
    state: Annotated[dict, InjectedState] = {},
) -> Dict[str, Any]:
    state_api = _require_runtime(state).state_api
    config = load_web_client_config()
    payload_dict = {"q": q, "k": k}
    response = _post_json(f"{config.base_url}/list", payload_dict)
    artifact_id = state_api.artifact_write(
        "application/json",
        json.dumps(response, ensure_ascii=False),
        {"source": "web.list"},
    )
    path = state_api.get_artifact_path(artifact_id)
    return {"artifact_id": artifact_id, "content_path": path}


def web_page(
    url: str = Field(..., description="Comma-separated URLs."),
    state: Annotated[dict, InjectedState] = {},
) -> Dict[str, Any]:
    _ = state
    config = load_web_client_config()
    payload_dict = {"urls": url}
    return _post_json(f"{config.base_url}/page", payload_dict)


def web_page_ref(
    url: str = Field(..., description="Comma-separated URLs."),
    state: Annotated[dict, InjectedState] = {},
) -> Dict[str, Any]:
    state_api = _require_runtime(state).state_api
    response = web_page(url, state=state)

    artifact_id = state_api.artifact_write(
        "application/json",
        json.dumps(response, ensure_ascii=False),
        {"source": "web.page", "url": url},
    )
    path = state_api.get_artifact_path(artifact_id)
    return {"artifact_id": artifact_id, "content_path": path}


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
        with urllib.request.urlopen(request, timeout=180) as response:
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


def _require_runtime(state: dict) -> ToolRuntime:
    runtime_id = state.get("runtime_id") if isinstance(state, dict) else None
    if not isinstance(runtime_id, str) or not runtime_id:
        raise ValueError("runtime_id is required in state")
    runtime = get_runtime(runtime_id)
    if runtime is None:
        raise ValueError("runtime is required in tool runtime registry")
    return runtime


def build_web_tools() -> List[BaseTool]:
    return [
        StructuredTool.from_function(
            web_query,
            name="web.query",
            description="Generate a focused web search query from the user message and current context. Use before web.list.",
        ),
        # StructuredTool.from_function(
        #     web_list,
        #     name="web.list",
        #     description="Fetch top-k web search results (snippets/urls). Use to choose candidate pages for reading.",
        # ),
        StructuredTool.from_function(
            web_list_ref,
            name="web.list",
            description="Fetch top-k web search results (snippets/urls) and store the extracted text as artifacts. Use to choose candidate pages for reading.",
        ),
        # StructuredTool.from_function(
        #     web_page,
        #     name="web.page",
        #     description="Fetch web page content by URLs.",
        # ),
        StructuredTool.from_function(
            web_page_ref,
            name="web.page",
            description=(
                "Fetch one or more web pages and store the extracted text as artifacts."
                "Use artifact.search to retrieve relevant pages later by meaning, then artifact.read / artifact.extract to consume them."
            ),
        ),
    ]


def _slugify(s: str, max_len: int = 60) -> str:
    s = re.sub(r"[^a-zA-Z0-9_-]+", "-", s).strip("-").lower()
    return s[:max_len] if s else "page"


def web_list_and_store(
    q: str,
    k: Optional[int],
    tool_runtime: Annotated[LCToolRuntime, InjectedToolArg],
) -> Dict[str, Any]:
    """
    args:
    q: search query(required). このツールを使う前に、web.queryでqueryを作成し、このツールを呼び出してください。
    k: 何件の検索結果を取得するか. default 5.
    """
    import datetime as dt
    from urllib.parse import urlparse

    k = k if k else 5

    config = load_web_client_config()
    payload_dict = {"q": q, "k": k}
    result = _post_json(f"{config.base_url}/list", payload_dict)

    result = _post_json(f"{config.base_url}/page", payload_dict)
    if tool_runtime.store is None:
        raise RuntimeError(
            "tool_runtime.store is None. "
            "StoreBackend を使うなら create_deep_agent(..., store=...) と "
            "CompositeBackend で /memories/ を StoreBackend にルーティングしてください。"
        )
    base = _slugify(q)
    today = dt.date.today().strftime("%Y%m%d")
    vpath = f"/memories/page_overview/{today}_{base}.md"
    tool_runtime.store.put(
        namespace=("filesystem",),
        key=vpath,
        value=result,
    )

    return {"saved_path": vpath, "bytes": len(result)}


def fetch_webpage_and_store(
    url: str,
    tool_runtime: Annotated[LCToolRuntime, InjectedToolArg],
) -> dict:
    """
    Fetch a web page and store it into the deepagents virtual filesystem under /memories/.
    Returns the saved virtual path.
    """
    import datetime as dt
    from urllib.parse import urlparse

    config = load_web_client_config()
    payload_dict = {"urls": url}
    result = _post_json(f"{config.base_url}/page", payload_dict)
    if tool_runtime.store is None:
        raise RuntimeError(
            "tool_runtime.store is None. "
            "StoreBackend を使うなら create_deep_agent(..., store=...) と "
            "CompositeBackend で /memories/ を StoreBackend にルーティングしてください。"
        )
    parsed = urlparse(url)
    base = _slugify(parsed.netloc + parsed.path)
    today = dt.date.today().strftime("%Y%m%d")
    vpath = f"/memories/pages/{today}_{base}.md"
    tool_runtime.store.put(
        namespace=("filesystem",),
        key=vpath,
        value=result,
    )

    return {"saved_path": vpath, "bytes": len(result)}


def build_web_tools_for_deep_agent() -> List[BaseTool]:
    return [
        StructuredTool.from_function(
            web_query,
            name="web.query",
            description="Generate a focused web search query from the user message and current context. Use before web.list.",
        ),
        StructuredTool.from_function(
            web_list_and_store,
            name="web.list",
            description="Fetch top-k web search results (snippets/urls). Use to choose candidate pages for reading.",
        ),
        StructuredTool.from_function(
            fetch_webpage_and_store,
            name="web.page_and_store",
            description="Fetch a web page and store it under /memories/pages/ as a markdown file. Returns saved_path.",
        ),
    ]
