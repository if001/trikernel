from __future__ import annotations

from typing import List, Optional, Sequence, Set, cast, Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from trikernel.orchestration_kernel.runners.protcol import RunnerAPI

from ..llm.config import load_ollama_config
from ..logging import get_logger
from ..models import Budget, RunResult, RunnerContext, SimpleStepContext
from ..payloads import extract_llm_input, extract_user_message
from ._shared import (
    SimpleToolLoopState,
    build_memory_context,
    budget_exceeded_text,
    budget_limit,
    filter_tools,
    handle_tool_error,
    last_ai_message,
    load_checkpoint_messages,
    message_content_text,
    recent_user_messages,
    tools_text,
)
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


class SimpleGraphToolLoopRunner(RunnerAPI):
    def __init__(
        self,
        model: Optional[ChatOllama] = None,
        recursion_limit: int = 10,
    ) -> None:
        self._model = model
        self._recursion_limit = recursion_limit

    def run(self, task: Task, runner_context: RunnerContext) -> RunResult:
        try:
            user_message = extract_user_message(extract_llm_input(task.payload or {}))
            if not user_message:
                return RunResult(
                    user_output=None,
                    task_state="failed",
                    artifact_refs=[],
                    error={"code": "MISSING_MESSAGE", "message": "message is required"},
                    stream_chunks=[],
                )
            tools = runner_context.tool_api.tool_structured_list()
            tool_list_text = tools_text(runner_context)
            step_context = _initial_step_context(task)
            graph = _build_graph(
                self._model or _default_model(),
                tools,
                user_message,
                tool_list_text,
                task.task_type,
                runner_context,
                runner_context.store,
            )
            limit = budget_limit(task, self._recursion_limit)
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
                        "memory_context_text": "",
                    },
                    config=config,
                    # debug=True,
                )
                last_message = last_ai_message(result.get("messages", []))
                output = message_content_text(last_message) if last_message else ""
                return RunResult(
                    user_output=output,
                    task_state="done",
                    artifact_refs=[],
                    error=None,
                    stream_chunks=[],
                )
            except GraphRecursionError:
                logger.warning("langgraph recursion limit hit: %s", task.task_id)
                messages = load_checkpoint_messages(
                    runner_context.message_store.checkpointer,
                    config,
                )
                if not messages:
                    return RunResult(
                        user_output=budget_exceeded_text(),
                        task_state="done",
                        artifact_refs=[],
                        error=None,
                        stream_chunks=[],
                    )
                messages = list(messages)
                memory_context_text = build_memory_context(
                    runner_context,
                    runner_context.conversation_id,
                    user_message,
                    log_details=True,
                )
                messages.append(AIMessage(content=budget_exceeded_text()))
                followup_prompt = "ここまでのツール結果を使い、途中であることを明記して回答してください。"
                if memory_context_text:
                    followup_prompt = (
                        f"{followup_prompt}\n\nMemory context:\n{memory_context_text}\n"
                    )
                response = (self._model or _default_model()).invoke(
                    list(recent_user_messages(messages, last_n=3))
                    + [HumanMessage(content=followup_prompt)]
                )
                messages.append(response)
                output = message_content_text(response)
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


def _build_graph(
    model: ChatOllama,
    tools: Sequence[BaseTool],
    user_message: str,
    tools_text: str,
    task_type: str,
    runner_context: RunnerContext,
    store,
):
    graph = StateGraph(SimpleToolLoopState)

    def discover(state: SimpleToolLoopState):
        messages = recent_user_messages(state["messages"], last_n=3)
        memory_context_text = state.get("memory_context_text", "")
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
        selected = set(runner_context.tool_api.tool_search(str(query)))
        return {"tool_set": selected}

    def agent(state):
        if state["step_context"].budget.remaining_steps <= 0:
            return {
                "messages": [AIMessage(content=budget_exceeded_text())],
                "stop": True,
            }

        memory_context_text = state.get("memory_context_text", "")
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

        allowed = filter_tools(tools, state["tool_set"])
        _tool_name = [v.name for v in allowed]
        logger.info(f"allowed tools: {_tool_name}")
        messages = recent_user_messages(state["messages"], last_n=3)
        response = model.bind_tools(allowed).invoke(
            [SystemMessage(content=system)]
            + list(messages)
            + [HumanMessage(content=prompt)]
        )
        _in = (
            [SystemMessage(content=system)]
            + list(messages)
            + [HumanMessage(content=prompt)]
        )
        logger.info(f"debug tool prompt::: {_in}")
        state["step_context"].budget.spent_steps += 1
        state["step_context"].budget.remaining_steps -= 1
        if response.content == "" and not response.tool_calls:
            in_token_cnt, out_token_cnt, total_token = -1, -1, -1
            if response.usage_metadata:
                in_token_cnt = response.usage_metadata["input_tokens"]
                out_token_cnt = response.usage_metadata["output_tokens"]
                total_token = response.usage_metadata["total_tokens"]
            logger.error(
                f"may be token over... in: {in_token_cnt}, out: {out_token_cnt}, total: {total_token}"
            )
        return {
            "messages": [response],
            "step_context": state["step_context"],
        }

    tool_node = ToolNode(list(tools), handle_tool_errors=handle_tool_error)

    def build_memory(state: SimpleToolLoopState):
        memory_context_text = build_memory_context(
            runner_context,
            runner_context.conversation_id,
            user_message,
            log_details=True,
        )
        return {"memory_context_text": memory_context_text}

    def followup(state: SimpleToolLoopState):
        memory_context_text = state.get("memory_context_text", "")
        messages = recent_user_messages(state["messages"], last_n=3)
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
        _in = (
            [SystemMessage(content=system)]
            + list(messages)
            + [HumanMessage(content=prompt)]
        )
        logger.info(f"debug folow up prompt::: {_in}")
        return {"messages": [response]}

    graph.add_node("build_memory", build_memory)
    graph.add_node("discover", discover)
    graph.add_node("agent", agent)
    graph.add_node("tools", tool_node)
    graph.add_node("followup", followup)

    def route(state: SimpleToolLoopState):
        if state.get("stop"):
            return END
        decision = tools_condition(cast(dict[str, Any], state))
        if decision == END:
            return "followup"
        return "tools"

    graph.set_entry_point("build_memory")
    graph.add_edge("build_memory", "discover")
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
