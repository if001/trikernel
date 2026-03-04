from __future__ import annotations

from typing import Sequence, cast, Any

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langchain.chat_models import BaseChatModel
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langmem.short_term import summarize_messages
from langmem.utils import AnyMessage

from trikernel.orchestration_kernel.runners.protcol import RunnerAPI

from ..logging import get_logger
from ..runtime import build_runnable_config
from ._shared import (
    build_memory_context,
    last_ai_message,
    load_checkpoint_messages,
    message_content_text,
    recent_user_messages,
    tools_text,
)
from ..models import Budget, RunResult, SimpleStepContext
from ..payloads import extract_llm_input, extract_user_message
from ._shared import (
    SimpleToolLoopState,
    budget_exceeded_text,
    budget_limit,
    filter_tools,
    handle_tool_error,
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
from ...state_kernel.protocols import StateKernelAPI
from ...state_kernel.core.message_store_interface import MessageStoreProtocol
from ...tool_kernel.kernel import ToolKernel
from langgraph.store.base import BaseStore

logger = get_logger(__name__)


class SimpleGraphToolLoopRunner(RunnerAPI):
    def __init__(
        self,
        *,
        state_api: StateKernelAPI,
        tool_api: ToolKernel,
        message_store: MessageStoreProtocol,
        store: BaseStore,
        llm_api: BaseChatModel,
        large_llm_api: BaseChatModel,
        recursion_limit: int = 50,
    ) -> None:
        self._state_api = state_api
        self._tool_api = tool_api
        self._message_store = message_store
        self._store = store
        self._llm_api = llm_api
        self._large_llm_api = large_llm_api
        self._recursion_limit = recursion_limit

    def run(
        self,
        task: Task,
        *,
        conversation_id: str,
        stream: bool = False,
    ) -> RunResult:
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
            tools = self._tool_api.tool_structured_list()
            tool_list_text = tools_text(self._tool_api)
            step_context = _initial_step_context(task)
            graph = _build_graph(
                self._llm_api,
                self._large_llm_api,
                tools,
                user_message,
                tool_list_text,
                task.task_type,
                self._state_api,
                self._tool_api,
                self._message_store,
                self._store,
            )
            limit = budget_limit(task, self._recursion_limit)
            config = build_runnable_config(
                conversation_id=conversation_id,
                state_api=self._state_api,
                tool_api=self._tool_api,
                recursion_limit=limit,
            )
            try:
                init_state: SimpleToolLoopState = {
                    "messages": [HumanMessage(content=user_message)],
                    "tool_set": set(),
                    "step_context": step_context,
                    "stop": False,
                    "runtime_id": conversation_id,
                    "task_id": task.task_id,
                    "memory_context_text": "",
                    "running_summary": None,
                    "tool_results": [],
                }
                result = graph.invoke(
                    init_state,
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
                    self._message_store.checkpointer,
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
                    self._state_api,
                    conversation_id,
                    user_message,
                    log_details=True,
                )
                messages.append(AIMessage(content=budget_exceeded_text()))
                followup_prompt = "ここまでのツール結果を使い、途中であることを明記して回答してください。"
                if memory_context_text:
                    followup_prompt = (
                        f"{followup_prompt}\n\nMemory context:\n{memory_context_text}\n"
                    )
                response = self._llm_api.invoke(
                    list(recent_user_messages(messages, last_n=2))
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
    model: BaseChatModel,
    large_model: BaseChatModel,
    tools: Sequence[BaseTool],
    user_message: str,
    tools_text: str,
    task_type: str,
    state_api: StateKernelAPI,
    tool_api: ToolKernel,
    message_store: MessageStoreProtocol,
    store: BaseStore,
):
    graph = StateGraph(SimpleToolLoopState)

    def discover(state: SimpleToolLoopState):
        _last = state["messages"][-1]
        logger.info(f"last: {_last}")
        if isinstance(_last, ToolMessage):
            state["tool_results"].append(str(_last.name) + ":" + str(_last.content))
        if isinstance(_last, HumanMessage):
            logger.warning("last is HumanMessage...")

        messages = recent_user_messages(state["messages"], last_n=2)
        memory_context_text = state.get("memory_context_text", "")
        summary = state["running_summary"]
        system, prompt = build_discover_tools_simple_prompt(
            user_input=user_message,
            tools_text=tools_text,
            step_context_text=state["step_context"].to_str(),
            memory_context_text=memory_context_text,
            summary=summary.summary if summary else None,
        )

        response = model.invoke(
            [SystemMessage(content=system)]
            + list(messages)
            + [HumanMessage(content=prompt)]
        )
        query = response.content or ""
        selected = set(tool_api.tool_search(str(query)))
        return {"tool_set": selected}

    def agent(state: SimpleToolLoopState):
        if state["step_context"].budget.remaining_steps <= 0:
            return {
                "messages": [AIMessage(content=budget_exceeded_text())],
                "stop": True,
            }
        summary = state["running_summary"]
        memory_context_text = state.get("memory_context_text", "")
        tool_results = state["tool_results"]
        _last = state["messages"][-1]
        if isinstance(_last, HumanMessage):
            logger.warning("last is HumanMessage...")

        if task_type == "user_request":
            system, prompt = build_tool_loop_prompt_simple(
                user_message=user_message,
                step_context_text=state["step_context"].to_str(),
                memory_context_text=memory_context_text,
                summary=summary.summary if summary else None,
                tool_results=tool_results,
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
        messages = recent_user_messages(state["messages"], last_n=2)
        response = large_model.bind_tools(allowed).invoke(
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
            state_api,
            state.get("runtime_id", ""),
            user_message,
            log_details=True,
        )
        return {"memory_context_text": memory_context_text}

    def followup(state: SimpleToolLoopState):
        memory_context_text = state.get("memory_context_text", "")
        messages = recent_user_messages(state["messages"], last_n=2)
        summary = state["running_summary"]
        if task_type == "user_request":
            system, prompt = build_tool_loop_followup_prompt(
                user_message=user_message,
                notes=[],
                phase_goal=None,
                memory_context_text=memory_context_text,
                summary=summary.summary if summary else None,
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

    def summarization_node(state: SimpleToolLoopState):
        messages: list[AnyMessage] = cast(list[AnyMessage], state["messages"])
        summarization_result = summarize_messages(
            messages,
            running_summary=state.get("running_summary"),
            model=model,
            max_tokens=512,
            max_tokens_before_summary=512,
            max_summary_tokens=256,
        )
        return {"running_summary": summarization_result.running_summary}

    def init_node(state: SimpleToolLoopState):
        m = state["messages"]
        # logger.info(f"init_state: {m}")
        return {}

    graph.add_node("init", init_node)
    graph.add_node("summarization", summarization_node)
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

    graph.set_entry_point("init")
    graph.add_edge("init", "build_memory")
    graph.add_edge("build_memory", "summarization")
    graph.add_edge("summarization", "discover")
    graph.add_edge("discover", "agent")
    graph.add_conditional_edges(
        "agent", route, {"tools": "tools", "followup": "followup", END: END}
    )
    graph.add_edge("tools", "discover")
    graph.add_edge("followup", END)

    if store is None:
        return graph.compile(checkpointer=message_store.checkpointer)
    return graph.compile(
        checkpointer=message_store.checkpointer,
        store=store,
    )
