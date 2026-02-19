from __future__ import annotations

from typing import Optional

from ui.terminal import TerminalUI
from trikernel.orchestration_kernel import (
    OllamaLLM,
    RunnerContext,
    SingleTurnRunner,
    PDCARunner,
    ToolLoopRunner,
)
from trikernel.state_kernel.kernel import StateKernel
from trikernel.tool_kernel.kernel import ToolKernel
from trikernel.tool_kernel.ollama import ToolOllamaLLM
from trikernel.tool_kernel.registry import register_default_tools


def main() -> None:
    ui = TerminalUI()
    state = StateKernel()
    tool_kernel = ToolKernel()
    register_default_tools(tool_kernel)

    # runner = SingleTurnRunner()
    # runner = PDCARunner()
    runner = ToolLoopRunner()
    llm = OllamaLLM()
    tool_llm = ToolOllamaLLM()

    ui.write_output("Type a message. Empty input exits.")
    while True:
        _drain_notifications(state, ui)
        user_input = ui.read_input("message> ").strip()
        if not user_input:
            break

        task_id = state.task_create("user_request", {"message": user_input})
        turn_id = state.turn_append_user("default", user_input, task_id)
        claimed_id = state.task_claim({"task_id": task_id}, "main", 30)
        if not claimed_id:
            ui.write_output("Failed to claim task.")
            continue
        task = state.task_get(claimed_id)
        if not task:
            ui.write_output("Failed to load task.")
            continue

        context = RunnerContext(
            runner_id="main",
            state_api=state,
            tool_api=tool_kernel,
            llm_api=llm,
            tool_llm_api=tool_llm,
            # stream=True,
            stream=False,
        )
        result = runner.run(task, context)
        if result.stream_chunks:
            ui.write_stream(result.stream_chunks)
            assistant_message = "".join(result.stream_chunks)
        else:
            ui.write_output(result.user_output or "")
            assistant_message = result.user_output or ""
        _finalize_task(state, task.task_id, result.task_state, result.error)
        state.turn_set_assistant(
            turn_id,
            assistant_message,
            result.artifact_refs,
            {"task_state": result.task_state},
        )


def _finalize_task(
    state: StateKernel, task_id: str, task_state: str, error: Optional[dict]
) -> None:
    if task_state == "done":
        state.task_complete(task_id)
    else:
        state.task_fail(task_id, error or {"message": "failed"})


def _drain_notifications(state: StateKernel, ui: TerminalUI) -> None:
    while True:
        notification_id = state.task_claim({"task_type": "notification"}, "main", 30)
        if not notification_id:
            return
        notification = state.task_get(notification_id)
        if not notification:
            continue
        payload = notification.payload or {}
        message = payload.get("message", "")
        if message:
            ui.write_output(message)
        state.task_complete(notification_id)


if __name__ == "__main__":
    main()
