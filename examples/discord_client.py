from __future__ import annotations

import asyncio
import os
from dotenv import load_dotenv

from ui.discord_client_ui import DiscordBot, get_intents, new_bot
from trikernel.orchestration_kernel import (
    OllamaLLM,
    SingleTurnRunner,
    PDCARunner,
    ToolLoopRunner,
    get_logger,
)
from trikernel.session import TrikernelSession
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
    session = TrikernelSession(state, tool_kernel, runner, llm, tool_llm)

    while not ui.stop_event.is_set():
        for message in session.drain_notifications():
            await ui.write_output(message)
        user_input = await ui.read_input()
        if not user_input:
            break
        async with ui.typing():
            result = await asyncio.to_thread(session.send_message, user_input, False)
        if result.stream_chunks:
            logger.error("not support")
        else:
            await ui.write_output(result.message or "")


def main() -> None:
    runner = ToolLoopRunner()
    intents = get_intents()
    ui = DiscordBot(intents=intents, runner_loop=lambda bot: runner_loop(bot, runner))
    if not TOKEN:
        raise SystemExit("DISCORD_BOT_TOKEN が未設定です。")
    ui.run(TOKEN)


if __name__ == "__main__":
    main()
