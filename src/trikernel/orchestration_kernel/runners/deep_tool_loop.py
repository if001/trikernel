from __future__ import annotations

import json
import time
from typing import List, Sequence, cast, Literal
from langchain.chat_models import BaseChatModel
from langmem.utils import AnyMessage, RunnableConfig
from pydantic import BaseModel, Field

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.tools import BaseTool
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langmem.short_term import summarize_messages

from trikernel.orchestration_kernel.runners.protcol import RunnerAPI
from ..logging import get_logger
from ..models import RunResult
from .models import Budget, ToolStepContext
from ..payloads import extract_llm_input, extract_user_message
from ..runtime import build_runnable_config
from .models import DeepToolLoopState
from ._shared import (
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
    build_discover_tools_deep_prompt,
    build_observe_prompt,
    build_plan_prompt,
    build_tool_loop_followup_prompt,
    build_tool_loop_followup_prompt_for_notification,
    build_tool_loop_followup_prompt_for_worker,
    build_tool_loop_prompt_deep,
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


class DeepToolLoopRunner(RunnerAPI):
    def __init__(
        self,
        *,
        state_api: StateKernelAPI,
        tool_api: ToolKernel,
        message_store: MessageStoreProtocol,
        store: BaseStore,
        llm_api: BaseChatModel,
        large_llm_api: BaseChatModel,
        recursion_limit: int = 100,
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
            step_context = _initial_step_context(task, self._recursion_limit)
            graph = _build_graph(
                self._large_llm_api,
                self._llm_api,
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
            config: RunnableConfig = build_runnable_config(
                conversation_id=conversation_id,
                state_api=self._state_api,
                tool_api=self._tool_api,
                recursion_limit=limit,
            )
            try:
                result = graph.invoke(
                    {
                        "messages": [HumanMessage(content=user_message)],
                        "running_summary": None,
                        "tool_set": set(),
                        "tool_step_context": step_context,
                        "stop": False,
                        "runtime_id": conversation_id,
                        "task_id": task.task_id,
                        "memory_context_text": "",
                        "phase": "GET",
                        "phase_goal": "",
                    },
                    config,
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
                checkpoint_config = {
                    "configurable": config.get("configurable", {}),
                }
                messages = load_checkpoint_messages(
                    self._message_store.checkpointer,
                    checkpoint_config,
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
                    log_missing=True,
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


def _initial_step_context(task: Task, limit: int) -> ToolStepContext:
    payload = task.payload or {}
    budget_payload = payload.get("budget") or {}
    budget = Budget(
        remaining_steps=int(budget_payload.get("remaining_steps", limit)),
        spent_steps=int(budget_payload.get("spent_steps", 0)),
    )
    return ToolStepContext(
        budget=budget,
    )


def _build_graph(
    large_model: BaseChatModel,
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    user_message: str,
    tools_text: str,
    task_type: str,
    state_api: StateKernelAPI,
    tool_api: ToolKernel,
    message_store: MessageStoreProtocol,
    store: BaseStore,
):
    graph = StateGraph(DeepToolLoopState)

    def build_memory(state: DeepToolLoopState):
        memory_context_text = build_memory_context(
            state_api,
            state.get("runtime_id", ""),
            user_message,
            log_missing=True,
        )
        return {"memory_context_text": memory_context_text}

    def plan(state: DeepToolLoopState):
        phase, goal = _plan_with_llm(
            model,
            state,
            user_message,
        )
        logger.info(f"phase: {phase}")
        logger.info(f"phase_goal: {goal}")
        # time.sleep(3)
        if phase == "FINISH":
            return {"phase": phase, "phase_goal": goal, "tool_set": set()}
        return {"phase": phase, "phase_goal": goal}

    def discover(state: DeepToolLoopState):
        messages = recent_user_messages(state["messages"], last_n=0)
        summary = state["running_summary"]
        phase_goal = state["phase_goal"]
        memory_context_text = state.get("memory_context_text", "")
        if state.get("phase") == "FINISH":
            return {"tool_set": set()}
        system, prompt = build_discover_tools_deep_prompt(
            user_input=user_message,
            tools_text=tools_text,
            step_context_text=state["tool_step_context"].to_str(),
            memory_context_text=memory_context_text,
            phase_goal=phase_goal,
            summary=summary.summary if summary else None,
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
        logger.info(f"discover, {_in}")
        query = response.content or ""
        selected = set(tool_api.tool_search(str(query)))
        logger.info(f"selected: {selected}")
        return {"tool_set": selected}

    def act(state: DeepToolLoopState):
        if state["tool_step_context"].budget.remaining_steps <= 0:
            return {
                "messages": [AIMessage(content=budget_exceeded_text())],
                "stop": True,
            }
        summary = state["running_summary"]
        memory_context_text = state.get("memory_context_text", "")
        if task_type == "user_request":
            system, prompt = build_tool_loop_prompt_deep(
                user_message=user_message,
                step_context_text=state["tool_step_context"].to_str(),
                memory_context_text=memory_context_text,
                summary=summary.summary if summary else None,
            )
        elif task_type == "notification":
            system, prompt = build_tool_loop_prompt_simple_for_notification(
                message=user_message,
                step_context_text=state["tool_step_context"].to_str(),
                memory_context_text=memory_context_text,
            )
        else:
            system, prompt = build_tool_loop_prompt_simple_for_worker(
                message=user_message,
                step_context_text=state["tool_step_context"].to_str(),
                memory_context_text=memory_context_text,
            )

        messages = recent_user_messages(state["messages"], last_n=2)
        if state.get("phase") == "FINISH":
            response = large_model.invoke(
                [SystemMessage(content=system)]
                + list(messages)
                + [HumanMessage(content=prompt)]
            )
        else:
            # logger.info(f"debug tools: {tools}")
            allowed = filter_tools(tools, state["tool_set"])
            _allowed_name = [v.name for v in allowed]
            logger.info(f"allowed_name: {_allowed_name}")
            logger.info(f"allowed: {allowed}")
            ## tool_choiceはollamaではサポートされてないらしい
            response = large_model.bind_tools(allowed, tool_choice="any").invoke(
                [SystemMessage(content=system)]
                + list(messages)
                + [HumanMessage(content=prompt)]
            )
            _in = (
                [SystemMessage(content=system)]
                + list(messages)
                + [HumanMessage(content=prompt)]
            )
            logger.info(f"act prompt: {_in}")

        state["tool_step_context"].budget.spent_steps += 1
        state["tool_step_context"].budget.remaining_steps -= 1
        if response.content == "" and not response.tool_calls:
            in_token_cnt, out_token_cnt, total_token = -1, -1, -1
            if response.usage_metadata:
                in_token_cnt = response.usage_metadata["input_tokens"]
                out_token_cnt = response.usage_metadata["output_tokens"]
                total_token = response.usage_metadata["total_tokens"]
            logger.error(
                f"may be token over... in: {in_token_cnt}, out: {out_token_cnt}, total: {total_token}"
            )
        logger.info(f"act response: {response}")
        time.sleep(5)
        return {
            "messages": [response],
            "tool_step_context": state["tool_step_context"],
        }

    tool_node = ToolNode(list(tools), handle_tool_errors=handle_tool_error)

    def observe(state: DeepToolLoopState):
        _s = state["messages"]
        logger.info(f"observe {_s}")

        observation = _observe_with_llm(large_model, state)
        state["tool_step_context"].last_observation = observation.last_observation
        state["tool_step_context"].error_summary = observation.error_summary
        state["tool_step_context"].need_clarification = observation.need_clarification
        state["tool_step_context"].notes = observation.notes
        # time.sleep(3)
        return {"tool_step_context": state["tool_step_context"]}

    def followup(state: DeepToolLoopState):
        memory_context_text = state.get("memory_context_text", "")
        messages = recent_user_messages(state["messages"], last_n=2)

        if task_type == "user_request":
            system, prompt = build_tool_loop_followup_prompt(
                user_message=user_message,
                notes=state["tool_step_context"].notes,
                phase_goal=state["phase_goal"],
                memory_context_text=memory_context_text,
            )
        elif task_type == "notification":
            system, prompt = build_tool_loop_followup_prompt_for_notification(
                message=user_message,
                step_context_text=state["tool_step_context"].to_str(),
                memory_context_text=memory_context_text,
            )
        else:
            system, prompt = build_tool_loop_followup_prompt_for_worker(
                message=user_message,
                step_context_text=state["tool_step_context"].to_str(),
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

        state["tool_set"] = set()  ## clean up
        # logger.info(f"final state: {state}")
        return {"messages": [response], "tool_set": set()}

    def summarization_node(state: DeepToolLoopState):
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

    def init_node(state: DeepToolLoopState):
        m = state["messages"]
        # logger.info(f"init_state: {m}")
        return {}

    graph.add_node("init", init_node)
    graph.add_node("summarization", summarization_node)
    graph.add_node("build_memory", build_memory)
    graph.add_node("plan", plan)
    graph.add_node("discover", discover)
    graph.add_node("act", act)
    graph.add_node("tools", tool_node)
    graph.add_node("observe", observe)
    graph.add_node("followup", followup)

    def route_after_act(state: DeepToolLoopState):
        if state.get("stop"):
            return "followup"
        decision = tools_condition(cast(dict[str, object], state))
        if decision == END:
            return "followup"
        return "tools"

    def route_after_observe(state: DeepToolLoopState):
        if state.get("stop"):
            return "followup"
        return "plan"

    graph.set_entry_point("init")
    graph.add_edge("init", "build_memory")
    graph.add_edge("build_memory", "summarization")
    graph.add_edge("summarization", "plan")
    graph.add_edge("plan", "discover")
    graph.add_edge("discover", "act")
    graph.add_conditional_edges(
        "act", route_after_act, {"tools": "tools", "followup": "followup"}
    )
    graph.add_edge("tools", "observe")
    graph.add_conditional_edges(
        "observe", route_after_observe, {"plan": "plan", "followup": "followup"}
    )
    graph.add_edge("followup", END)

    if store is None:
        return graph.compile(checkpointer=message_store.checkpointer)
    return graph.compile(
        checkpointer=message_store.checkpointer,
        store=store,
    )


class _ObservationResult:
    def __init__(
        self,
        last_observation: str,
        error_summary: str,
        need_clarification: List[str],
        notes: List[str],
    ) -> None:
        self.last_observation = last_observation
        self.error_summary = error_summary
        self.need_clarification = need_clarification
        self.notes = notes


class _PlanDecision(BaseModel):
    phase: Literal["GET", "WORK", "FINISH"] = Field(..., description="Next phase.")
    phase_goal: str = Field(..., description="Goal for the next iteration.")


class _ObserveDecision(BaseModel):
    last_observation: str = Field(
        default="",
        description="Short summary of the tool results.",
    )
    error_summary: str = Field(
        default="",
        description="Short sentence about failures if any.",
    )
    need_clarification: List[str] = Field(
        default_factory=list,
        description="Questions to ask the user.",
    )
    notes: List[str] = Field(
        default_factory=list,
        description="Concise bullet points about progress and next possible actions.",
    )


def _plan_with_llm(
    model: BaseChatModel,
    state: DeepToolLoopState,
    user_message: str,
) -> tuple[str, str]:
    _m = state["messages"][-2:]
    logger.info(f"last message {_m}")
    _ctx: ToolStepContext = state["tool_step_context"]

    memory_context_text = state.get("memory_context_text", "")
    summary = state["running_summary"]
    system, prompt = build_plan_prompt(
        user_message,
        memory_context_text,
        phase=state["phase"],
        phase_goal=state["phase_goal"],
        last_observation=_ctx.last_observation,
        notes=_ctx.notes,
        need_clarification=_ctx.need_clarification,
        remaining_steps=_ctx.budget.remaining_steps,
        spent_steps=_ctx.budget.spent_steps,
        summary=summary.summary if summary else None,
    )
    logger.info(f"plan system: {system}")
    logger.info(f"plan prompt: {prompt}")
    try:
        structured = model.with_structured_output(_PlanDecision)
        result = structured.invoke(
            [SystemMessage(content=system), HumanMessage(content=prompt)]
        )
        result = cast(_PlanDecision, result)
        logger.info(f"result plan: {result.phase},{result.phase_goal}")
        return result.phase, result.phase_goal
    except Exception:
        response = model.invoke(
            [SystemMessage(content=system), HumanMessage(content=prompt)]
        )
        parsed = _parse_json_block(response.content)
        phase = str(parsed.get("phase", "GET")).upper()
        if phase not in {"GET", "WORK", "FINISH"}:
            phase = "GET"
        goal = str(parsed.get("phase_goal", "") or parsed.get("next_goal", ""))
        if not goal:
            goal = "Proceed with the next step."
        return phase, goal


def _observe_with_llm(
    model: BaseChatModel, state: DeepToolLoopState
) -> _ObservationResult:
    _ctx: ToolStepContext = state["tool_step_context"]
    tool_result = _last_tool_result(state["messages"])
    summary = state["running_summary"]
    system, prompt = build_observe_prompt(
        tool_result=tool_result,
        phase=state["phase"],
        phase_goal=state["phase_goal"],
        last_observation=_ctx.last_observation,
        notes=_ctx.notes,
        need_clarification=_ctx.need_clarification,
        error_summary=_ctx.error_summary,
        summary=summary.summary if summary else None,
    )
    logger.info(f"observe system: {system}")
    logger.info(f"observe prompt: {prompt}")
    try:
        structured = model.with_structured_output(_ObserveDecision)
        result = structured.invoke(
            [SystemMessage(content=system), HumanMessage(content=prompt)]
        )
        result = cast(_ObserveDecision, result)
        return _ObservationResult(
            last_observation=result.last_observation,
            error_summary=result.error_summary,
            need_clarification=[str(x) for x in result.need_clarification if x],
            notes=[str(x) for x in result.notes if x],
        )
    except Exception:
        response = model.invoke(
            [SystemMessage(content=system), HumanMessage(content=prompt)]
        )
        parsed = _parse_json_block(response.content)
        last_observation = str(parsed.get("last_observation") or "")
        error_summary = str(parsed.get("error_summary") or "")
        need_clarification = parsed.get("need_clarification") or []
        if isinstance(need_clarification, str):
            need_clarification = [need_clarification]
        notes = parsed.get("notes") or []
        if isinstance(notes, str):
            notes = [notes]
        return _ObservationResult(
            last_observation=last_observation,
            error_summary=error_summary,
            need_clarification=[str(x) for x in need_clarification if x],
            notes=[str(x) for x in notes if x],
        )


def _parse_json_block(raw: object) -> dict:
    if not isinstance(raw, str):
        return {}
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
    try:
        return json.loads(text)
    except Exception:
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                return json.loads(text[start : end + 1])
        except Exception:
            return {}
    return {}


def _last_tool_result(messages: List[BaseMessage]) -> str:
    _last = messages[-1]
    return str(_last.content)
