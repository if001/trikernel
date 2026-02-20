from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .orchestration_kernel.models import RunResult, RunnerContext
from .orchestration_kernel.protocols import LLMAPI, Runner
from .state_kernel.models import Task
from .state_kernel.protocols import StateKernelAPI
from .tool_kernel.protocols import ToolAPI, ToolLLMAPI


@dataclass
class MessageResult:
    message: Optional[str]
    task_state: str
    artifact_refs: List[str]
    error: Optional[Dict[str, object]]
    stream_chunks: List[str]


class TrikernelSession:
    def __init__(
        self,
        state_api: StateKernelAPI,
        tool_api: ToolAPI,
        runner: Runner,
        llm_api: LLMAPI,
        tool_llm_api: Optional[ToolLLMAPI] = None,
        conversation_id: str = "default",
        runner_id: str = "main",
        claim_ttl_seconds: int = 30,
    ) -> None:
        self._state_api = state_api
        self._tool_api = tool_api
        self._runner = runner
        self._llm_api = llm_api
        self._tool_llm_api = tool_llm_api
        self._conversation_id = conversation_id
        self._runner_id = runner_id
        self._claim_ttl_seconds = claim_ttl_seconds

    def send_message(self, message: str, stream: bool = False) -> MessageResult:
        task_id = self._state_api.task_create(
            "user_request", {"user_message": message}
        )
        turn_id = self._state_api.turn_append_user(
            self._conversation_id, message, task_id
        )
        claimed_id = self._state_api.task_claim(
            {"task_id": task_id}, self._runner_id, self._claim_ttl_seconds
        )
        if not claimed_id:
            return MessageResult(
                message=None,
                task_state="failed",
                artifact_refs=[],
                error={"message": "Failed to claim task."},
                stream_chunks=[],
            )
        task = self._state_api.task_get(claimed_id)
        if not task:
            return MessageResult(
                message=None,
                task_state="failed",
                artifact_refs=[],
                error={"message": "Failed to load task."},
                stream_chunks=[],
            )
        result = self._run_task(task, stream=stream)
        assistant_message = result.user_output or ""
        if result.stream_chunks:
            assistant_message = "".join(result.stream_chunks) or assistant_message
        self._finalize_task(task, result)
        self._state_api.turn_set_assistant(
            turn_id,
            assistant_message,
            result.artifact_refs,
            {"task_state": result.task_state},
        )
        return MessageResult(
            message=assistant_message,
            task_state=result.task_state,
            artifact_refs=result.artifact_refs,
            error=result.error,
            stream_chunks=result.stream_chunks,
        )

    def drain_notifications(self) -> List[str]:
        messages: List[str] = []
        while True:
            notification_id = self._state_api.task_claim(
                {"task_type": "notification"}, self._runner_id, self._claim_ttl_seconds
            )
            if not notification_id:
                break
            notification = self._state_api.task_get(notification_id)
            if not notification:
                continue
            payload = notification.payload or {}
            message = payload.get("message")
            if isinstance(message, str) and message:
                messages.append(message)
            self._state_api.task_complete(notification_id)
        return messages

    def _run_task(self, task: Task, stream: bool) -> RunResult:
        context = RunnerContext(
            runner_id=self._runner_id,
            state_api=self._state_api,
            tool_api=self._tool_api,
            llm_api=self._llm_api,
            tool_llm_api=self._tool_llm_api,
            stream=stream,
        )
        return self._runner.run(task, context)

    def _finalize_task(self, task: Task, result: RunResult) -> None:
        if result.task_state == "done":
            self._state_api.task_complete(task.task_id)
        else:
            self._state_api.task_fail(
                task.task_id, result.error or {"message": "failed"}
            )
