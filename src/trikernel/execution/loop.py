from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional

from .dispatcher import WorkDispatcher
from .worker import WorkWorker


@dataclass
class LoopConfig:
    poll_interval: float = 0.1


class ExecutionLoop:
    def __init__(
        self,
        dispatcher: WorkDispatcher,
        worker: WorkWorker,
        config: Optional[LoopConfig] = None,
    ) -> None:
        self.dispatcher = dispatcher
        self.worker = worker
        self.config = config or LoopConfig()
        self._stop = asyncio.Event()

    async def run_forever(self) -> None:
        while not self._stop.is_set():
            await self.dispatcher.run_once()
            await self.worker.run_once()
            await asyncio.sleep(self.config.poll_interval)

    def stop(self) -> None:
        self._stop.set()
