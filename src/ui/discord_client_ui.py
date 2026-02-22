# bot.py
import os
import asyncio
from contextlib import asynccontextmanager
from typing import Awaitable, Callable, Optional, Tuple

import discord
from discord import TextChannel
from dotenv import load_dotenv

from trikernel.orchestration_kernel.logging import get_logger

logger = get_logger(__name__)
load_dotenv(".env.discord")

CHANNEL_ID = int(os.getenv("DISCORD_CHANNEL_ID", -1))
DISCORD_MESSAGE_LIMIT = 2000


def chunk_message(text: str, limit: int = DISCORD_MESSAGE_LIMIT) -> list[str]:
    """Discord 2000 文字制限に合わせて分割（改行優先、最後は強制分割）"""
    chunks, buf = [], ""
    for line in text.splitlines(keepends=True):
        if len(buf) + len(line) > limit:
            chunks.append(buf)
            buf = line
        else:
            buf += line
    if buf:
        chunks.append(buf)
    final = []
    for c in chunks:
        if len(c) <= limit:
            final.append(c)
        else:  # 非常に長い行を強制分割
            for i in range(0, len(c), limit):
                final.append(c[i : i + limit])
    return final


def get_intents():
    intents = discord.Intents.default()
    intents.message_content = (
        True  # メッセージ本文の取得を許可（Bot設定画面でも有効化が必要）
    )
    intents.guilds = True
    intents.messages = True
    return intents


class DiscordBot(discord.Client):
    def __init__(
        self,
        *args,
        runner_loop: Optional[Callable[["DiscordBot"], Awaitable[None]]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.conversation_id = "default"
        self.user_input: asyncio.Queue[Tuple[int, str]] = asyncio.Queue(5)
        self.assistant_output: asyncio.Queue[Tuple[int, str]] = asyncio.Queue(5)
        self.stop_event = asyncio.Event()
        self.channel: Optional[TextChannel] = None
        self._runner_loop = runner_loop
        self._ui_task: Optional[asyncio.Task] = None
        logger.info("discord start...")

    async def on_ready(self):
        logger.info(f"Logged in as {self.user} (id={self.user.id})")

        _channel = self.get_channel(CHANNEL_ID)
        if not _channel:
            raise ValueError("channel not exist")
        if type(_channel) != TextChannel:
            raise ValueError("channel not exist")

        self.channel = _channel

        async def _loop():
            while not self.stop_event.is_set():
                channel_id, _out = await self.assistant_output.get()
                target = self.get_channel(channel_id) if channel_id else _channel
                if not isinstance(target, TextChannel):
                    continue
                for chunk in chunk_message(_out):
                    await target.send(chunk)

        self.send_loop_task = asyncio.create_task(_loop())
        if self._runner_loop and not self._ui_task:
            self._ui_task = asyncio.create_task(self._runner_loop(self))

    async def on_message(self, message: discord.Message):
        # 自分は無視
        if message.author.bot:
            return

        # メンションされていなければ無視
        if not self.user or self.user not in message.mentions:
            return

        # すでにスレッド内→親メッセージ側でスレッドにまとめたいので無視（必要なら変更OK）
        if isinstance(message.channel, discord.Thread):
            return

        ## はじめの改行はmenssionの@だとして取り除く
        user_input = ""
        part_message = message.content.split("\n", 1)
        if len(part_message) > 1:
            user_input = part_message[1]
        else:
            user_input = message.content
        logger.info(f"on message! {message.channel.id}")
        self.user_input.put_nowait((message.channel.id, user_input))

    async def read_input(self) -> Tuple[int, str]:
        return await self.user_input.get()

    def write_stream(self, assistant_output: str):
        raise NotImplementedError()

    async def write_output(
        self, assistant_output: str, channel_id: Optional[int] = None
    ):
        target_id = channel_id or (self.channel.id if self.channel else 0)
        return self.assistant_output.put_nowait((target_id, assistant_output))

    @asynccontextmanager
    async def typing(self):
        if self.channel:
            async with self.channel.typing():
                yield
        else:
            yield

    async def close(self):
        self.stop_event.set()
        self.send_loop_task.cancel()
        if self._ui_task:
            self._ui_task.cancel()
        await super().close()


def new_bot(
    start_ui: Optional[Callable[[DiscordBot], Awaitable[None]]] = None,
) -> DiscordBot:
    return DiscordBot(intents=intents, start_ui=start_ui)
