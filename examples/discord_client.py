from __future__ import annotations

import asyncio
import os
from dotenv import load_dotenv

from trikernel.orchestration_kernel.llm.gemini import newGeiminiClient
from trikernel.orchestration_kernel.llm.ollama import (
    newOllamaClient,
    newOllamaCloudClient,
)
from trikernel.orchestration_kernel.runners._shared import SimpleToolLoopState
from trikernel.orchestration_kernel.runners.agent_loop import AgentLoopRunner
from trikernel.orchestration_kernel.runners.deep_agent_loop import DeepAgentLoopRunner
from trikernel.orchestration_kernel.runners.deep_tool_loop import DeepToolLoopRunner
from trikernel.orchestration_kernel.runners.simple_tool_loop import (
    SimpleGraphToolLoopRunner,
)
from ui.discord_client_ui import DiscordBot, get_intents
from trikernel.orchestration_kernel import get_logger, RunnerAPI
from trikernel.execution.session import TrikernelSession
from trikernel.state_kernel import (
    create_state_kernel,
    build_memory_store,
    build_message_store,
)

from trikernel.tool_kernel.kernel import ToolKernel
from trikernel.tool_kernel.ollama import ToolOllamaLLM
from trikernel.tool_kernel.registry import (
    register_deep_agent_tools,
    register_default_tools,
)

from tools.web_tools import build_web_tools, build_web_tools_for_deep_agent

logger = get_logger("discord_client")
load_dotenv()
logger.info("start")
load_dotenv(".env.discord")
TOKEN = os.getenv("DISCORD_BOT_TOKEN")

DISCORD_CHANNEL_ID = int(os.getenv("DISCORD_CHANNEL_ID", -1))
DISCORD_READ_CHANNEL_ID = int(os.getenv("DISCORD_READ_CHANNEL_ID", -1))


async def runner_loop(ui: DiscordBot, runner: RunnerAPI) -> None:
    logger.info("runner_loop")

    llm = newOllamaClient()
    llm_cloud = newOllamaCloudClient()
    # gemini_llm = newGeiminiClient()

    tool_llm = ToolOllamaLLM()
    tool_kernel = ToolKernel(re_index=False)

    async with build_memory_store() as store, build_message_store() as message_store:
        state = create_state_kernel(store)
        register_default_tools(tool_kernel)
        for tool in build_web_tools():
            tool_kernel.tool_register(tool)

        # register_deep_agent_tools(tool_kernel)
        # for tool in build_web_tools_for_deep_agent():
        #     tool_kernel.tool_register(tool)

        # tool_kernel.debug()

        session = TrikernelSession(
            state_api=state,
            tool_api=tool_kernel,
            runner=runner,
            large_llm_api=llm_cloud,
            llm_api=llm,
            tool_llm_api=tool_llm,
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
                    channel_id = (
                        meta.get("channel_id") if isinstance(meta, dict) else None
                    )
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
                    result = await asyncio.to_thread(
                        session.send_message, user_input, False
                    )
                if result.stream_chunks:
                    logger.error("not support")
                else:
                    await ui.write_output(result.message or "", channel_id=channel_id)
        finally:
            notification_task.cancel()
            session.stop_workers()


def main() -> None:
    runner = SimpleGraphToolLoopRunner()
    # runner = DeepToolLoopRunner()

    ## create_agent/deep_agent, build_toolをagent用にする必要がある
    # runner = AgentLoopRunner()
    # runner = DeepAgentLoopRunner()
    intents = get_intents()
    ui = DiscordBot(intents=intents, runner_loop=lambda bot: runner_loop(bot, runner))
    if not TOKEN:
        raise SystemExit("DISCORD_BOT_TOKEN が未設定です。")
    ui.run(TOKEN)


if __name__ == "__main__":
    main()
