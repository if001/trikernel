from __future__ import annotations

import asyncio
from typing import Optional
import os
from dotenv import load_dotenv

from ui.discord_client_ui import DiscordBot, get_intents, new_bot
from trikernel.orchestration_kernel import (
    OllamaLLM,
    RunnerContext,
    SingleTurnRunner,
    PDCARunner,
    ToolLoopRunner,
    get_logger,
)
from trikernel.state_kernel.kernel import StateKernel
from trikernel.tool_kernel.kernel import ToolKernel
from trikernel.tool_kernel.ollama import ToolOllamaLLM
from trikernel.tool_kernel.registry import register_default_tools


logger = get_logger("discord_client")
load_dotenv()
logger.info("start")
load_dotenv(".env.discord")
TOKEN = os.getenv("DISCORD_BOT_TOKEN")


async def runner_loop(ui: DiscordBot, runner: ToolLoopRunner) -> None:
    llm = OllamaLLM()
    tool_llm = ToolOllamaLLM()
    state = StateKernel()
    tool_kernel = ToolKernel()
    register_default_tools(tool_kernel)

    while not ui.stop_event.is_set():
        await _drain_notifications(state, ui)
        user_input = await ui.read_input()
        if not user_input:
            break

        task_id = state.task_create("user_request", {"message": user_input})
        turn_id = state.turn_append_user("default", user_input, task_id)
        claimed_id = state.task_claim({"task_id": task_id}, "main", 30)
        if not claimed_id:
            await ui.write_output("Failed to claim task.")
            continue
        task = state.task_get(claimed_id)
        if not task:
            await ui.write_output("Failed to load task.")
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
        async with ui.typing():
            result = await asyncio.to_thread(runner.run, task, context)
        assistant_message = ""
        if result.stream_chunks:
            logger.error("not support")
            pass
        else:
            assistant_message = result.user_output or ""
            await ui.write_output(assistant_message)
        _finalize_task(state, task.task_id, result.task_state, result.error)
        state.turn_set_assistant(
            turn_id,
            assistant_message,
            result.artifact_refs,
            {"task_state": result.task_state},
        )


def main() -> None:
    runner = ToolLoopRunner()
    intents = get_intents()
    ui = DiscordBot(intents=intents, runner_loop=lambda bot: runner_loop(bot, runner))
    if not TOKEN:
        raise SystemExit("DISCORD_BOT_TOKEN が未設定です。")
    ui.run(TOKEN)


def _finalize_task(
    state: StateKernel, task_id: str, task_state: str, error: Optional[dict]
) -> None:
    if task_state == "done":
        state.task_complete(task_id)
    else:
        state.task_fail(task_id, error or {"message": "failed"})


async def _drain_notifications(state: StateKernel, ui: DiscordBot) -> None:
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
            await ui.write_output(message)
        state.task_complete(notification_id)


if __name__ == "__main__":
    main()
