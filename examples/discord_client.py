from __future__ import annotations

import asyncio
import os
from dotenv import load_dotenv

from ui.discord_client_ui import DiscordBot, get_intents
from trikernel.orchestration_kernel import OllamaLLM, LangGraphToolLoopRunner, get_logger
from trikernel.execution.session import TrikernelSession
from trikernel.state_kernel.kernel import StateKernel
from trikernel.state_kernel.memory_store import build_memory_store
from trikernel.state_kernel.message_store import build_message_store
from trikernel.tool_kernel.kernel import ToolKernel
from trikernel.tool_kernel.ollama import ToolOllamaLLM
from trikernel.tool_kernel.registry import register_default_tools

from tools.web_tools import build_web_tools

logger = get_logger("discord_client")
load_dotenv()
logger.info("start")
load_dotenv(".env.discord")
TOKEN = os.getenv("DISCORD_BOT_TOKEN")

DISCORD_CHANNEL_ID = int(os.getenv("DISCORD_CHANNEL_ID", -1))
DISCORD_READ_CHANNEL_ID = int(os.getenv("DISCORD_READ_CHANNEL_ID", -1))


async def runner_loop(ui: DiscordBot, runner: LangGraphToolLoopRunner) -> None:
    logger.info("runner_loop")
    llm = OllamaLLM()
    tool_llm = ToolOllamaLLM()
    state = StateKernel()
    tool_kernel = ToolKernel(re_index=False)

    async with build_memory_store() as store, build_message_store() as message_store:
        register_default_tools(tool_kernel, store=store)
        for tool in build_web_tools():
            tool_kernel.tool_register(tool)

        session = TrikernelSession(
            state,
            tool_kernel,
            runner,
            llm,
            tool_llm,
            message_store=message_store,
            store=store,
        )
        session.start_workers()

        async def _notification_loop() -> None:
            while not ui.stop_event.is_set():
                notices = session.drain_notifications()
                for notice in notices:
                    text = notice.get("message") if isinstance(notice, dict) else notice
                    meta = notice.get("meta") if isinstance(notice, dict) else {}
                    channel_id = meta.get("channel_id") if isinstance(meta, dict) else None
                    await ui.write_output(
                        text or "", channel_id=channel_id or DISCORD_CHANNEL_ID
                    )
                if not notices:
                    await asyncio.sleep(0.2)

        notification_task = asyncio.create_task(_notification_loop())

        try:
            while not ui.stop_event.is_set():
                logger.info("run_wait")
                channel_id, user_input = await ui.read_input()
                logger.info("read: %s", channel_id)
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
                    result = await asyncio.to_thread(session.send_message, user_input, False)
                if result.stream_chunks:
                    logger.error("not support")
                else:
                    await ui.write_output(result.message or "", channel_id=channel_id)
        finally:
            notification_task.cancel()
            session.stop_workers()


def main() -> None:
    runner = LangGraphToolLoopRunner()
    intents = get_intents()
    ui = DiscordBot(intents=intents, runner_loop=lambda bot: runner_loop(bot, runner))
    if not TOKEN:
        raise SystemExit("DISCORD_BOT_TOKEN が未設定です。")
    ui.run(TOKEN)


if __name__ == "__main__":
    main()
