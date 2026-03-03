from __future__ import annotations

from typing import Annotated, List, Optional, Sequence, Set, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langgraph.graph.message import add_messages
from langmem.short_term import RunningSummary

from trikernel.state_kernel.protocols import StateKernelAPI

from ..logging import get_logger
from ..models import RunnerContext, SimpleStepContext, ToolStepContext
from ...state_kernel.models import Task


class BaseState(TypedDict):
    task_id: str
    runtime_id: str
    messages: Annotated[List[BaseMessage], add_messages]
    state_api: Optional[StateKernelAPI]  ## runtime_idから取得される


class ToolLoopState(BaseState):
    tool_set: Set[str]
    stop: bool
    memory_context_text: str


class SimpleToolLoopState(ToolLoopState):
    step_context: SimpleStepContext


class DeepToolLoopState(BaseState):
    tool_set: Set[str]
    stop: bool
    memory_context_text: str
    tool_step_context: ToolStepContext
    phase: str
    phase_goal: str
    running_summary: RunningSummary | None


def budget_limit(task: Task, default_limit: int) -> int:
    payload = task.payload or {}
    budget = payload.get("budget") or {}
    remaining = budget.get("remaining_steps")
    try:
        if remaining is not None:
            steps = max(0, int(remaining))
            return max(default_limit, steps * 3 + 3)
    except (TypeError, ValueError):
        return default_limit
    return default_limit


def filter_tools(tools: Sequence[BaseTool], tool_set: Set[str]) -> List[BaseTool]:
    if not tool_set:
        return []
    return [tool for tool in tools if tool.name in tool_set]


def history_messages(messages: Sequence[BaseMessage]) -> Sequence[BaseMessage]:
    selected: List[BaseMessage] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            continue
        if isinstance(message, ToolMessage):
            continue
        if isinstance(message, AIMessage) and message.tool_calls:
            continue
        selected.append(message)
    return selected


def budget_exceeded_text() -> str:
    return (
        "上限に達したためtool使用をストップしました。"
        "ここまでのtoolの結果を利用し、調査が足りていない旨を含めて回答してください。"
    )


def handle_tool_error(exc: Exception) -> str:
    return f"TOOL_ERROR: {exc}"


logger = get_logger(__name__)


def tools_text(runner_context: RunnerContext) -> str:
    tools_text = "tool_list:\n"
    for v in runner_context.tool_api.tool_descriptions():
        tools_text += f"{v['tool_name']}: {v['description']}\n"
    return tools_text


def build_memory_context(
    runner_context: RunnerContext,
    conversation_id: str,
    query: str,
    *,
    log_missing: bool = False,
    log_details: bool = False,
) -> str:
    memory_kernel = runner_context.state_api.memory_kernel(conversation_id)
    if memory_kernel is None:
        if log_missing:
            logger.warning("memory_kernel is None")
        return ""
    profile_text = memory_kernel.get_profile_context(limit=1)
    # semantic_text = memory_kernel.get_semantic_context(query, limit=1)
    semantic_text = ""
    episodic_text = memory_kernel.get_episodic_context(query, limit=1)
    if log_details:
        logger.info("profile_text: %s", profile_text)
        logger.info("semantic_text: %s", semantic_text)
        logger.info("episodic_text: %s", episodic_text)

    return "\n".join(
        part for part in (profile_text, semantic_text, episodic_text) if part
    )


def last_ai_message(messages: Sequence[BaseMessage]) -> AIMessage | None:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
    return None


def message_content_text(message: AIMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return str(content)


def load_checkpoint_messages(checkpointer, config: dict) -> Sequence[BaseMessage]:
    try:
        checkpoint_tuple = checkpointer.get_tuple(config)
    except Exception:
        logger.error("failed to load checkpoint messages", exc_info=True)
        return []
    if not checkpoint_tuple:
        return []
    checkpoint = checkpoint_tuple.checkpoint
    channel_values = checkpoint.get("channel_values", {})
    messages = channel_values.get("messages")
    if isinstance(messages, list):
        return messages
    return []


def recent_user_messages(
    messages: Sequence[BaseMessage], last_n: int
) -> Sequence[BaseMessage]:
    if last_n <= 0:
        return []
    selected: List[BaseMessage] = []
    seen_ai = False

    _last = messages[-1]
    if isinstance(_last, HumanMessage):
        messages = messages[:-1]
    for message in reversed(messages):
        if isinstance(message, SystemMessage):
            continue
        if isinstance(message, ToolMessage):
            continue
        if isinstance(message, AIMessage):
            if message.tool_calls:
                continue
            if not seen_ai:
                selected.append(message)
                seen_ai = True
            continue
        if isinstance(message, HumanMessage):
            selected.append(message)
            if seen_ai:
                last_n -= 1
                if last_n <= 0:
                    break
                seen_ai = False
    return list(reversed(selected))
