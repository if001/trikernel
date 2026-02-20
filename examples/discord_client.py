from __future__ import annotations

import asyncio
import os
from dotenv import load_dotenv


from ui.discord_client_ui import DiscordBot, get_intents
from trikernel.orchestration_kernel import (
    OllamaLLM,
    SingleTurnRunner,
    PDCARunner,
    ToolLoopRunner,
    get_logger,
)
from trikernel.execution.session import TrikernelSession
from trikernel.state_kernel.kernel import StateKernel
from trikernel.tool_kernel.kernel import ToolKernel
from trikernel.tool_kernel.ollama import ToolOllamaLLM
from trikernel.tool_kernel.registry import register_default_tools

from pathlib import Path

from trikernel.tool_kernel.dsl import build_tools_from_dsl

logger = get_logger("discord_client")
load_dotenv()
logger.info("start")
load_dotenv(".env.discord")
TOKEN = os.getenv("DISCORD_BOT_TOKEN")


DISCORD_CHANNEL_ID = int(os.getenv("DISCORD_CHANNEL_ID", -1))
# Messages in this channel run the main session.
DISCORD_READ_CHANNEL_ID = int(os.getenv("DISCORD_READ_CHANNEL_ID", -1))
# Messages in this channel enqueue work tasks.


from tools.web_tools import web_list, web_page, web_query


def build_web_tools():
    dsl_path = Path(__file__).resolve().parent / "tools" / "web_tools.yaml"
    function_map = {
        "web.query": web_query,
        "web.list": web_list,
        "web.page": web_page,
    }
    tools = build_tools_from_dsl(dsl_path, function_map)
    return tools


async def runner_loop(ui: DiscordBot, runner: ToolLoopRunner) -> None:
    logger.info("runner_loop")
    llm = OllamaLLM()
    tool_llm = ToolOllamaLLM()
    state = StateKernel()
    tool_kernel = ToolKernel()
    register_default_tools(tool_kernel)

    tools = build_web_tools()
    for tool in tools:
        tool_kernel.tool_register_structured(tool)

    session = TrikernelSession(state, tool_kernel, runner, llm, tool_llm)
    session.start_workers()

    try:
        while not ui.stop_event.is_set():
            for notice in session.drain_notifications():
                text = notice.get("message") if isinstance(notice, dict) else notice
                meta = notice.get("meta") if isinstance(notice, dict) else {}
                channel_id = meta.get("channel_id") if isinstance(meta, dict) else None
                await ui.write_output(
                    text or "", channel_id=channel_id or DISCORD_CHANNEL_ID
                )
            channel_id, user_input = await ui.read_input()
            if not user_input:
                break
            if channel_id == DISCORD_READ_CHANNEL_ID:
                logger.info("create_task")
                task_id = session.create_work_task(
                    {
                        "message": (
                            f"{user_input}\n"
                            "urlを読んで、内容を要約し、artifactとして保存し、内容をユーザーに通知してください。"
                        ),
                        "meta": {"channel_id": channel_id},
                    }
                )
                await ui.write_output(
                    f"Queued work task: {task_id}", channel_id=channel_id
                )
                continue
            async with ui.typing():
                result = await asyncio.to_thread(
                    session.send_message, user_input, False
                )
            if result.stream_chunks:
                logger.error("not support")
            else:
                await ui.write_output(result.message or "", channel_id=channel_id)
    finally:
        session.stop_workers()


def main() -> None:
    runner = ToolLoopRunner()
    intents = get_intents()
    ui = DiscordBot(intents=intents, runner_loop=lambda bot: runner_loop(bot, runner))
    if not TOKEN:
        raise SystemExit("DISCORD_BOT_TOKEN が未設定です。")
    ui.run(TOKEN)


if __name__ == "__main__":
    main()
