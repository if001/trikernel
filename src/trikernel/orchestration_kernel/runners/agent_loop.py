from __future__ import annotations
import os
import time
from typing import Any


from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)

from langchain.agents import create_agent
from langchain_core.prompts import HumanMessagePromptTemplate
from langgraph.runtime import Runtime
from langchain.agents.middleware import (
    before_model,
    after_model,
    before_agent,
    AgentState,
    LLMToolSelectorMiddleware,
    SummarizationMiddleware,
    ClearToolUsesEdit,
    ContextEditingMiddleware,
    ClearToolUsesEdit,
    FilesystemFileSearchMiddleware,
)

from trikernel.orchestration_kernel.runners.protcol import RunnerAPI
# from deepagents.middleware.filesystem import FilesystemMiddleware

from ..logging import get_logger
from ._shared import history_messages
from ..models import RunResult, RunnerContext
from ..payloads import extract_llm_input, extract_user_message
from .prompts import (
    build_agent_prompt,
)
from ...state_kernel.models import Task


logger = get_logger(__name__)


class AgentLoopRunner(RunnerAPI):
    def __init__(
        self,
        recursion_limit: int = 50,
    ):
        self._recursion_limit = recursion_limit

    def run(self, task: Task, runner_context: RunnerContext) -> RunResult:
        try:
            ctx = runner_context
            user_message = extract_user_message(extract_llm_input(task.payload or {}))
            if not user_message:
                return RunResult(
                    user_output=None,
                    task_state="failed",
                    artifact_refs=[],
                    error={"code": "MISSING_MESSAGE", "message": "message is required"},
                    stream_chunks=[],
                )
            agent = self._build(ctx)
            config = {
                "recursion_limit": self._recursion_limit,
                "configurable": {
                    "thread_id": ctx.conversation_id,
                    "langgraph_user_id": ctx.conversation_id,
                },
            }
            result = agent.invoke(
                {"messages": [HumanMessage(content=user_message)]},
                config,
                # debug=True,
            )
            logger.info(f"raw result: {result}")

            response = result["messages"][-1]
            if response.content == "" and not response.tool_calls:
                in_token_cnt, out_token_cnt, total_token = -1, -1, -1
                if response.usage_metadata:
                    in_token_cnt = response.usage_metadata["input_tokens"]
                    out_token_cnt = response.usage_metadata["output_tokens"]
                    total_token = response.usage_metadata["total_tokens"]
                    logger.error(
                        f"may be token over... in: {in_token_cnt}, out: {out_token_cnt}, total: {total_token}"
                    )
                    return RunResult(
                        user_output=None,
                        task_state="failed",
                        artifact_refs=[],
                        error=None,
                        stream_chunks=[],
                    )

            output = response.content
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

    def _build(self, ctx: RunnerContext):
        @before_agent
        def _sleep(state: AgentState, runtime: Runtime) -> dict[str, Any]:
            logger.info("sleep...")
            time.sleep(5)
            return {}

        @before_agent
        def _inject_long_term_memories(
            state: AgentState, runtime: Runtime
        ) -> dict[str, Any] | None:
            logger.info("run _inject_long_term_memories")
            msgs = state.get("messages", [])
            if not msgs:
                return None

            only_history = history_messages(msgs)

            query = getattr(msgs[-1], "content", "") or ""
            memory_kernel = ctx.state_api.memory_kernel(ctx.conversation_id)
            memory_text = ""
            if memory_kernel is None:
                logger.warning("memory_kernel is None")
            else:
                profile_text = memory_kernel.get_profile_context(limit=1)
                # logger.info(f"profile_text: {profile_text}")
                semantic_text = memory_kernel.get_semantic_context(query, limit=1)
                # logger.info(f"semantic_text: {semantic_text}")
                episodic_text = memory_kernel.get_episodic_context(query, limit=1)
                # logger.info(f"episodic_text: {episodic_text}")
                memory_text = "\n".join(
                    part
                    for part in (profile_text, semantic_text, episodic_text)
                    if part
                )

            system = build_agent_prompt(memory_text)
            return {"messages": [SystemMessage(content=system), *only_history]}

        selector = LLMToolSelectorMiddleware(
            model=ctx.llm_api,
            max_tools=4,
            always_include=[],
        )
        summarization = SummarizationMiddleware(
            model=ctx.llm_api,
            trigger=("messages", 10),
            keep=("messages", 5),
        )
        clearToolResult = ContextEditingMiddleware(
            edits=[
                ClearToolUsesEdit(
                    trigger=2000,
                    keep=3,
                )
            ]
        )
        work_space_dir = os.environ.get("work_space_dir", os.getcwd())
        fsSearch = FilesystemFileSearchMiddleware(
            root_path=work_space_dir,
            use_ripgrep=True,  # ripgrepがあれば使う
            max_file_size_mb=10,  # 大きすぎるファイルはスキップ
        )

        tools = ctx.tool_api.tool_structured_list()
        agent = create_agent(
            model=ctx.large_llm_api,
            tools=tools,
            middleware=[
                _sleep,
                _inject_long_term_memories,
                summarization,
                clearToolResult,
                selector,
                fsSearch,
            ],
            store=ctx.store,
            checkpointer=ctx.message_store.checkpointer,
        )
        return agent
