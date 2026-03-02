from __future__ import annotations

import json
import time
from typing import Annotated, List, Optional, Sequence, Set, TypedDict, cast, Literal
from langchain.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from langchain_core.tools import BaseTool
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from trikernel.orchestration_kernel.runners.protcol import RunnerAPI
from ..logging import get_logger
from ..models import Budget, RunResult, RunnerContext, ToolStepContext
from ..payloads import extract_llm_input
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
from ...tool_kernel.runtime import register_runtime

logger = get_logger(__name__)


class DeepToolLoopRunner(RunnerAPI):
    def __init__(
        self,
        recursion_limit: int = 20,
    ) -> None:
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
                runner_context.large_llm_api,
                runner_context.llm_api,
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
                        "tool_step_context": step_context,
                        "stop": False,
                        "runtime_id": runner_context.conversation_id,
                        "task_id": task.task_id,
                        "memory_context_text": "",
                        "phase": "GET",
                        "phase_goal": "",
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
                response = runner_context.llm_api.invoke(
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


def _initial_step_context(task: Task) -> ToolStepContext:
    payload = task.payload or {}
    budget_payload = payload.get("budget") or {}
    budget = Budget(
        remaining_steps=int(budget_payload.get("remaining_steps", 10)),
        spent_steps=int(budget_payload.get("spent_steps", 0)),
    )
    return ToolStepContext(
        budget=budget,
    )


class ToolLoopState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    tool_set: Set[str]
    tool_step_context: ToolStepContext
    stop: bool
    runtime_id: str
    task_id: str
    memory_context_text: str
    phase: str
    phase_goal: str


def _build_graph(
    large_model: BaseChatModel,
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    user_message: str,
    tools_text: str,
    task_type: str,
    runner_context: RunnerContext,
    store,
):
    graph = StateGraph(ToolLoopState)

    def build_memory(state: ToolLoopState):
        memory_context_text = _build_memory_context(
            runner_context.state_api,
            runner_context.conversation_id,
            user_message,
        )
        return {"memory_context_text": memory_context_text}

    def plan(state: ToolLoopState):
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

    def discover(state):
        messages = _trim_state_messages(state["messages"])
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
        selected = set(runner_context.tool_api.tool_search(str(query)))
        logger.info(f"selected: {selected}")
        return {"tool_set": selected}

    def act(state: ToolLoopState):
        if state["tool_step_context"].budget.remaining_steps <= 0:
            return {
                "messages": [AIMessage(content=_budget_exceeded_text())],
                "stop": True,
            }

        memory_context_text = state.get("memory_context_text", "")
        if task_type == "user_request":
            system, prompt = build_tool_loop_prompt_deep(
                user_message=user_message,
                step_context_text=state["tool_step_context"].to_str(),
                memory_context_text=memory_context_text,
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

        messages = _trim_state_messages(state["messages"])
        if state.get("phase") == "FINISH":
            response = model.invoke(
                [SystemMessage(content=system)]
                + list(messages)
                + [HumanMessage(content=prompt)]
            )
        else:
            # logger.info(f"debug tools: {tools}")
            allowed = _filter_tools(tools, state["tool_set"])
            _allowed_name = [v.name for v in allowed]
            logger.info(f"allowed: {_allowed_name}")
            response = large_model.bind_tools(allowed).invoke(
                [SystemMessage(content=system)]
                + list(messages)
                + [HumanMessage(content=prompt)]
            )
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
        # time.sleep(3)
        return {
            "messages": [response],
            "tool_step_context": state["tool_step_context"],
        }

    tool_node = ToolNode(list(tools), handle_tool_errors=_handle_tool_error)

    def observe(state: ToolLoopState):
        observation = _observe_with_llm(model, state)
        state["tool_step_context"].last_observation = observation.last_observation
        state["tool_step_context"].error_summary = observation.error_summary
        state["tool_step_context"].need_clarification = observation.need_clarification
        state["tool_step_context"].notes = observation.notes
        # time.sleep(3)
        return {"tool_step_context": state["tool_step_context"]}

    def followup(state: ToolLoopState):
        memory_context_text = state.get("memory_context_text", "")
        messages = _trim_state_messages(state["messages"])
        if task_type == "user_request":
            system, prompt = build_tool_loop_followup_prompt(
                user_message=user_message,
                step_context_text=state["tool_step_context"].to_str(),
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

    graph.add_node("build_memory", build_memory)
    graph.add_node("plan", plan)
    graph.add_node("discover", discover)
    graph.add_node("act", act)
    graph.add_node("tools", tool_node)
    graph.add_node("observe", observe)
    graph.add_node("followup", followup)

    def route_after_act(state: ToolLoopState):
        if state.get("stop"):
            return "followup"
        decision = tools_condition(cast(dict[str, object], state))
        if decision == END:
            return "followup"
        return "tools"

    def route_after_observe(state: ToolLoopState):
        if state.get("stop"):
            return "followup"
        return "plan"

    graph.set_entry_point("build_memory")
    graph.add_edge("build_memory", "plan")
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


def keep_last_n_user_turns(
    messages: Sequence[BaseMessage], n_turns: int
) -> Sequence[BaseMessage]:
    _messages = messages[:-1]  ## 一番最後は今回の入力なので取り除く
    count = 0
    start_idx = 0
    for i in range(len(_messages) - 1, -1, -1):
        if isinstance(_messages[i], HumanMessage):
            count += 1
            if count >= n_turns:
                start_idx = i
                break
    return _messages[start_idx:] if count >= n_turns else _messages


def _trim_state_messages(messages: Sequence[BaseMessage]) -> Sequence[BaseMessage]:
    trimed = keep_last_n_user_turns(messages, 2)
    return trim_messages(
        trimed,
        max_tokens=3000,
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
        logger.warning("memory_kernel is None")
        return ""
    profile_text = memory_kernel.get_profile_context(limit=1)
    # logger.info(f"profile_text: {profile_text}")
    semantic_text = memory_kernel.get_semantic_context(query, limit=1)
    # logger.info(f"semantic_text: {semantic_text}")
    episodic_text = memory_kernel.get_episodic_context(query, limit=1)
    # logger.info(f"episodic_text: {episodic_text}")

    return "\n".join(
        part for part in (profile_text, semantic_text, episodic_text) if part
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
    state: ToolLoopState,
    user_message: str,
) -> tuple[str, str]:
    _ctx: ToolStepContext = state["tool_step_context"]

    memory_context_text = state.get("memory_context_text", "")
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


def _observe_with_llm(model: BaseChatModel, state: ToolLoopState) -> _ObservationResult:
    # messages = _trim_state_messages(state["messages"])
    _ctx: ToolStepContext = state["tool_step_context"]
    tool_result = _last_tool_result(state["messages"])

    system, prompt = build_observe_prompt(
        tool_result=tool_result,
        phase=state["phase"],
        phase_goal=state["phase_goal"],
        last_observation=_ctx.last_observation,
        notes=_ctx.notes,
        need_clarification=_ctx.need_clarification,
        error_summary=_ctx.error_summary,
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
