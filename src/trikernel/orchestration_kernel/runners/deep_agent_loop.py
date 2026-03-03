from __future__ import annotations
import os
from typing import Any


from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)

from langchain.agents import create_agent
from langgraph.runtime import Runtime
from langchain.agents.middleware import (
    FilesystemFileSearchMiddleware,
)
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend

from trikernel.orchestration_kernel.runners.protcol import RunnerAPI


from ..logging import get_logger
from ..models import RunResult, RunnerContext
from ..payloads import extract_llm_input, extract_user_message
from .prompts import (
    build_agent_prompt,
)
from ...state_kernel.models import Task


logger = get_logger(__name__)


class DeepAgentLoopRunner(RunnerAPI):
    def __init__(
        self,
        recursion_limit: int = 20,
    ):
        self._recursion_limit = recursion_limit

    def run(self, task: Task, ctx: RunnerContext) -> RunResult:
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
        work_space_dir = os.environ.get("work_space_dir", os.getcwd())
        fsSearch = FilesystemFileSearchMiddleware(
            root_path=work_space_dir,
            use_ripgrep=True,
            max_file_size_mb=10,
        )

        def make_backend(runtime):
            return CompositeBackend(
                default=StateBackend(runtime),
                routes={"/memories/": StoreBackend(runtime)},
            )

        tools = ctx.tool_api.tool_structured_list()
        logger.info(f"tools: {tools}")
        system = build_agent_prompt()
        agent = create_deep_agent(
            model=ctx.large_llm_api,
            tools=tools,
            system_prompt=system,
            store=ctx.store,
            backend=make_backend,
            checkpointer=ctx.message_store.checkpointer,
            middleware=[fsSearch],
        )
        return agent
