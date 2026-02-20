from __future__ import annotations

from typing import List, Optional, Set

from ..state_kernel.models import Task, utc_now
from ..tool_kernel.models import ToolContext
from .models import LLMToolCall, RunnerContext
from .types import ToolResult

from ..utils.logging import get_logger

logger = get_logger(__name__)


def build_tool_context(runner_context: RunnerContext, task: Task) -> ToolContext:
    return ToolContext(
        runner_id=runner_context.runner_id,
        task_id=task.task_id,
        state_api=runner_context.state_api,
        now=utc_now(),
        llm_api=runner_context.tool_llm_api,
    )


def execute_tool_calls(
    runner_context: RunnerContext,
    task: Task,
    tool_calls: List[LLMToolCall],
    allowed_tools: Optional[Set[str]] = None,
) -> List[ToolResult]:
    tool_results: List[ToolResult] = []
    for call in tool_calls:
        logger.info(f"tool run: {call.tool_name}")
        if allowed_tools is not None and call.tool_name not in allowed_tools:
            tool_results.append(
                {
                    "tool": call.tool_name,
                    "result": {"error_type": "tool_not_allowed"},
                    "tool_call_id": call.tool_call_id,
                }
            )
            continue
        tool_context = build_tool_context(runner_context, task)
        try:
            result = runner_context.tool_api.tool_invoke(
                call.tool_name, call.args, tool_context
            )
            tool_results.append(
                {
                    "tool": call.tool_name,
                    "result": result,
                    "tool_call_id": call.tool_call_id,
                }
            )
        except ValueError as exc:
            tool_results.append(
                {
                    "tool": call.tool_name,
                    "result": {
                        "error_type": "invalid_args",
                        "message": str(exc),
                    },
                    "tool_call_id": call.tool_call_id,
                }
            )
        except Exception as exc:
            tool_results.append(
                {
                    "tool": call.tool_name,
                    "result": {
                        "error_type": "tool_error",
                        "message": str(exc),
                    },
                    "tool_call_id": call.tool_call_id,
                }
            )
    return tool_results
