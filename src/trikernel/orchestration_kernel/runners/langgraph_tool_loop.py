from __future__ import annotations

from typing import Annotated, List, Optional, Sequence, Set, TypedDict, cast, Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from ..llm.config import load_ollama_config
from ..logging import get_logger
from ..models import Budget, RunResult, RunnerContext, SimpleStepContext
from ..payloads import extract_llm_input
from .prompts import (
    build_discover_tools_simple_prompt,
    build_tool_loop_followup_prompt,
    build_tool_loop_followup_prompt_for_notification,
    build_tool_loop_followup_prompt_for_worker,
    build_tool_loop_prompt_simple,
    build_tool_loop_prompt_simple_for_notification,
    build_tool_loop_prompt_simple_for_worker,
)
from ...state_kernel.models import Task
from ...tool_kernel.runtime import register_runtime

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
                task.task_type,
                runner_context,
                runner_context.store,
            )
            limit = _budget_limit(task, self._recursion_limit)
            register_runtime(
                runner_context.conversation_id,
                runner_context.state_api,
                runner_context.tool_llm_api,
            )
            config = {
                "recursion_limit": limit,
                "configurable": {
                    "thread_id": runner_context.conversation_id,
                    "langgraph_user_id": runner_context.conversation_id,
                },
            }
            try:
                result = graph.invoke(
                    {
                        "messages": [HumanMessage(content=user_message)],
                        "tool_set": set(),
                        "step_context": step_context,
                        "stop": False,
                        "runtime_id": runner_context.conversation_id,
                        "task_id": task.task_id,
                    },
                    config=config,
                    # debug=True,
                )
                last_message = _last_ai_message(result.get("messages", []))
                output = _message_content_text(last_message) if last_message else ""
                return RunResult(
                    user_output=output,
                    task_state="done",
                    artifact_refs=[],
                    error=None,
                    stream_chunks=[],
                )
            except GraphRecursionError:
                logger.warning("langgraph recursion limit hit: %s", task.task_id)
                messages = _load_checkpoint_messages(
                    runner_context.message_store.checkpointer,
                    config,
                )
                if not messages:
                    return RunResult(
                        user_output=_budget_exceeded_text(),
                        task_state="done",
                        artifact_refs=[],
                        error=None,
                        stream_chunks=[],
                    )
                messages = list(messages)
                memory_context_text = _build_memory_context(
                    runner_context.state_api,
                    runner_context.conversation_id,
                    user_message,
                )
                messages.append(AIMessage(content=_budget_exceeded_text()))
                followup_prompt = "ここまでのツール結果を使い、途中であることを明記して回答してください。"
                if memory_context_text:
                    followup_prompt = (
                        f"{followup_prompt}\n\nMemory context:\n{memory_context_text}\n"
                    )
                response = (self._model or _default_model()).invoke(
                    list(_trim_state_messages(messages))
                    + [HumanMessage(content=followup_prompt)]
                )
                messages.append(response)
                output = _message_content_text(response)
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
        budget=budget,
    )


class ToolLoopState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    tool_set: Set[str]
    step_context: SimpleStepContext
    stop: bool
    runtime_id: str
    task_id: str


def _build_graph(
    model: ChatOllama,
    tools: Sequence[BaseTool],
    user_message: str,
    tools_text: str,
    task_type: str,
    runner_context: RunnerContext,
    store,
):
    graph = StateGraph(ToolLoopState)

    def discover(state):
        messages = _trim_state_messages(state["messages"])
        memory_context_text = _build_memory_context(
            runner_context.state_api,
            runner_context.conversation_id,
            user_message,
        )
        system, prompt = build_discover_tools_simple_prompt(
            user_input=user_message,
            tools_text=tools_text,
            step_context_text=state["step_context"].to_str(),
            memory_context_text=memory_context_text,
        )

        response = model.invoke(
            [SystemMessage(content=system)]
            + list(messages)
            + [HumanMessage(content=prompt)]
        )
        query = response.content or ""
        logger.info(f"debug: tool select query: {query}")
        selected = set(runner_context.tool_api.tool_search(str(query)))
        return {"tool_set": selected}

    def agent(state):
        if state["step_context"].budget.remaining_steps <= 0:
            return {
                "messages": [AIMessage(content=_budget_exceeded_text())],
                "stop": True,
            }

        memory_context_text = _build_memory_context(
            runner_context.state_api,
            runner_context.conversation_id,
            user_message,
        )
        if task_type == "user_request":
            system, prompt = build_tool_loop_prompt_simple(
                user_message=user_message,
                step_context_text=state["step_context"].to_str(),
                memory_context_text=memory_context_text,
            )
        elif task_type == "notification":
            system, prompt = build_tool_loop_prompt_simple_for_notification(
                message=user_message,
                step_context_text=state["step_context"].to_str(),
                memory_context_text=memory_context_text,
            )
        else:
            system, prompt = build_tool_loop_prompt_simple_for_worker(
                message=user_message,
                step_context_text=state["step_context"].to_str(),
                memory_context_text=memory_context_text,
            )

        allowed = _filter_tools(tools, state["tool_set"])
        _tool_name = [v.name for v in allowed]
        logger.info(f"allowed: {_tool_name}")
        messages = _trim_state_messages(state["messages"])
        response = model.bind_tools(allowed).invoke(
            [SystemMessage(content=system)]
            + list(messages)
            + [HumanMessage(content=prompt)]
        )
        state["step_context"].budget.spent_steps += 1
        state["step_context"].budget.remaining_steps -= 1
        logger.info(f"response {response}")
        return {
            "messages": [response],
            "step_context": state["step_context"],
        }

    tool_node = ToolNode(list(tools), handle_tool_errors=_handle_tool_error)

    def followup(state: ToolLoopState):
        memory_context_text = _build_memory_context(
            runner_context.state_api,
            runner_context.conversation_id,
            user_message,
        )
        messages = _trim_state_messages(state["messages"])
        if task_type == "user_request":
            system, prompt = build_tool_loop_followup_prompt(
                user_message=user_message,
                step_context_text=state["step_context"].to_str(),
                memory_context_text=memory_context_text,
            )
        elif task_type == "notification":
            system, prompt = build_tool_loop_followup_prompt_for_notification(
                message=user_message,
                step_context_text=state["step_context"].to_str(),
                memory_context_text=memory_context_text,
            )
        else:
            system, prompt = build_tool_loop_followup_prompt_for_worker(
                message=user_message,
                step_context_text=state["step_context"].to_str(),
            )
        response = model.invoke(
            [SystemMessage(content=system)]
            + list(messages)
            + [HumanMessage(content=prompt)]
        )
        return {"messages": [response]}

    graph.add_node("discover", discover)
    graph.add_node("agent", agent)
    graph.add_node("tools", tool_node)
    graph.add_node("followup", followup)

    def route(state: ToolLoopState):
        if state.get("stop"):
            return END
        decision = tools_condition(cast(dict[str, Any], state))
        if decision == END:
            return "followup"
        return "tools"

    graph.set_entry_point("discover")
    graph.add_edge("discover", "agent")
    graph.add_conditional_edges(
        "agent", route, {"tools": "tools", "followup": "followup", END: END}
    )
    graph.add_edge("tools", "discover")
    graph.add_edge("followup", END)

    if store is None:
        return graph.compile(checkpointer=runner_context.message_store.checkpointer)
    return graph.compile(
        checkpointer=runner_context.message_store.checkpointer,
        store=store,
    )


def _budget_limit(task: Task, default_limit: int) -> int:
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
        return []
    return [tool for tool in tools if tool.name in tool_set]


def _trim_state_messages(messages: Sequence[BaseMessage]) -> Sequence[BaseMessage]:
    return trim_messages(
        list(messages),
        max_tokens=2000,
        strategy="last",
        token_counter=_token_counter,
    )


def _budget_exceeded_text() -> str:
    return (
        "上限に達したためtool使用をストップしました。"
        "ここまでのtoolの結果を利用し、調査が足りていない旨を含めて回答してください。"
    )


def _handle_tool_error(exc: Exception) -> str:
    logger.error("tool execution error: %s", exc, exc_info=True)
    return f"TOOL_ERROR: {exc}"


def _message_content_text(message: AIMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return str(content)


def _load_checkpoint_messages(checkpointer, config: dict) -> Sequence[BaseMessage]:
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


def _build_memory_context(
    state_api,
    conversation_id: str,
    query: str,
) -> str:
    memory_kernel = state_api.memory_kernel(conversation_id)
    if memory_kernel is None:
        return ""
    profile_text = memory_kernel.get_profile_context(limit=3)
    semantic_text = memory_kernel.get_semantic_context(query, limit=3)
    episodic_text = memory_kernel.get_episodic_context(query, limit=3)
    return "\n".join(
        part for part in (profile_text, semantic_text, episodic_text) if part
    )
