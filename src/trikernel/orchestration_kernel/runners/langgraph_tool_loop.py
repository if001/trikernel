from __future__ import annotations

from typing import Annotated, List, Optional, Sequence, Set, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, trim_messages
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from ..llm.config import load_ollama_config
from ..logging import get_logger
from ..models import Budget, RunResult, RunnerContext, SimpleStepContext
from ..payloads import extract_llm_input
from ...state_kernel.models import utc_now
from .common import add_budget_exceeded_message
from .prompts import (
    build_discover_tools_simple_prompt,
    build_tool_loop_prompt_simple,
    build_tool_loop_prompt_simple_for_notification,
    build_tool_loop_prompt_simple_for_worker,
)
from ...tool_kernel.context import tool_context_scope
from ...tool_kernel.models import ToolContext
from ...state_kernel.models import Task

logger = get_logger(__name__)


class LangGraphToolLoopRunner:
    def __init__(
        self,
        model: Optional[ChatOllama] = None,
        recursion_limit: int = 10,
    ) -> None:
        self._model = model
        self._recursion_limit = recursion_limit

    def run(self, task: Task, runner_context: RunnerContext) -> RunResult:
        try:
            user_message = _extract_user_message(task)
            if not user_message:
                return RunResult(
                    user_output=None,
                    task_state="failed",
                    artifact_refs=[],
                    error={"code": "MISSING_MESSAGE", "message": "message is required"},
                    stream_chunks=[],
                )
            tools = runner_context.tool_api.tool_structured_list()
            tools_text = _tools_text(runner_context)
            step_context = _initial_step_context(task)
            graph = _build_graph(
                self._model or _default_model(),
                tools,
                user_message,
                tools_text,
                step_context,
                task.task_type,
                runner_context,
            )
            limit = _budget_limit(task, self._recursion_limit)
            context = _build_tool_context(runner_context, task)
            with tool_context_scope(context):
                result = graph.invoke(
                    {
                        "messages": [HumanMessage(content=user_message)],
                        "tool_set": set(),
                        "step_context": step_context,
                    },
                    config={
                        "recursion_limit": limit,
                        "configurable": {"thread_id": runner_context.conversation_id},
                    },
                )
            last_message = _last_ai_message(result.get("messages", []))
            output = last_message.content if last_message else ""
            return RunResult(
                user_output=output,
                task_state="done",
                artifact_refs=[],
                error=None,
                stream_chunks=[],
            )
        except Exception as exc:
            logger.error("langgraph runner failed: %s", task.task_id, exc_info=True)
            return RunResult(
                user_output=None,
                task_state="failed",
                artifact_refs=[],
                error={"code": "LANGGRAPH_RUNNER_ERROR", "message": str(exc)},
                stream_chunks=[],
            )


def _default_model() -> ChatOllama:
    config = load_ollama_config()
    model = config.model or "llama3"
    return ChatOllama(model=model, base_url=config.base_url)


def _extract_user_message(task: Task) -> str:
    payload = task.payload or {}
    llm_input = extract_llm_input(payload)
    if llm_input.get("message") is not None:
        return str(llm_input.get("message"))
    if llm_input.get("prompt") is not None:
        return str(llm_input.get("prompt"))
    if llm_input.get("user_message") is not None:
        return str(llm_input.get("user_message"))
    return ""


def _tools_text(runner_context: RunnerContext) -> str:
    tools_text = "tool_list:\n"
    for v in runner_context.tool_api.tool_descriptions():
        tools_text += f"{v['tool_name']}: {v['description']}\n"
    return tools_text


def _build_tool_context(runner_context: RunnerContext, task: Task) -> ToolContext:
    return ToolContext(
        runner_id=runner_context.runner_id,
        task_id=task.task_id,
        state_api=runner_context.state_api,
        now=utc_now(),
        llm_api=runner_context.tool_llm_api,
        message_store=runner_context.message_store,
    )


def _initial_step_context(task: Task) -> SimpleStepContext:
    payload = task.payload or {}
    budget_payload = payload.get("budget") or {}
    budget = Budget(
        remaining_steps=int(budget_payload.get("remaining_steps", 10)),
        spent_steps=int(budget_payload.get("spent_steps", 0)),
    )
    role = "main" if task.task_type == "user_request" else "worker"
    return SimpleStepContext(
        role=role,
        task_type=task.task_type,
        tool_summary="",
        budget=budget,
    )


class ToolLoopState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    tool_set: Set[str]
    step_context: SimpleStepContext


def _build_graph(
    model: ChatOllama,
    tools: Sequence[BaseTool],
    user_message: str,
    tools_text: str,
    step_context: SimpleStepContext,
    task_type: str,
    runner_context: RunnerContext,
):
    graph = StateGraph(ToolLoopState)

    def discover(state):
        messages = _trim_state_messages(state["messages"])
        prompt = build_discover_tools_simple_prompt(
            user_input=user_message,
            tools_text=tools_text,
            step_context_text=state["step_context"].to_str(),
        )
        response = model.invoke(list(messages) + [HumanMessage(content=prompt)])
        query = response.content or ""
        selected = set(runner_context.tool_api.tool_search(str(query)))
        logger.info(f"tool_set: {selected}")
        return {"tool_set": selected}

    def agent(state):
        if state["step_context"].budget.remaining_steps <= 0:
            budget_messages: List[BaseMessage] = []
            add_budget_exceeded_message(budget_messages)
            return {"messages": budget_messages}

        if task_type == "user_request":
            prompt = build_tool_loop_prompt_simple(
                user_message=user_message,
                step_context_text=state["step_context"].to_str(),
            )
        elif task_type == "notification":
            prompt = build_tool_loop_prompt_simple_for_notification(
                message=user_message,
                step_context_text=state["step_context"].to_str(),
            )
        else:
            prompt = build_tool_loop_prompt_simple_for_worker(
                message=user_message,
                step_context_text=state["step_context"].to_str(),
            )

        allowed = _filter_tools(tools, state["tool_set"])
        messages = _trim_state_messages(state["messages"])
        response = model.bind_tools(allowed).invoke(
            list(messages) + [HumanMessage(content=prompt)]
        )
        state["step_context"].budget.spent_steps += 1
        state["step_context"].budget.remaining_steps -= 1
        return {
            "messages": [response],
            "step_context": state["step_context"],
        }

    tool_node = ToolNode(list(tools))

    graph.add_node("discover", discover)
    graph.add_node("agent", agent)
    graph.add_node("tools", tool_node)
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "discover")
    graph.add_edge("discover", "agent")
    graph.set_entry_point("discover")
    return graph.compile(checkpointer=runner_context.message_store.checkpointer)


def _budget_limit(task: Task, default_limit: int) -> int:
    payload = task.payload or {}
    budget = payload.get("budget") or {}
    remaining = budget.get("remaining_steps")
    try:
        if remaining is not None:
            return max(1, int(remaining))
    except (TypeError, ValueError):
        return default_limit
    return default_limit


def _last_ai_message(messages: Sequence[BaseMessage]) -> Optional[AIMessage]:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
    return None


def _token_counter(messages: Sequence[BaseMessage]) -> int:
    total = 0
    for message in messages:
        total += len(str(message.content))
    return total


def _filter_tools(tools: Sequence[BaseTool], tool_set: Set[str]) -> List[BaseTool]:
    if not tool_set:
        return list(tools)
    return [tool for tool in tools if tool.name in tool_set]


def _trim_state_messages(messages: Sequence[BaseMessage]) -> Sequence[BaseMessage]:
    return trim_messages(
        list(messages),
        max_tokens=2000,
        strategy="last",
        token_counter=_token_counter,
    )
