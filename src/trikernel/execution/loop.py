from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional

from trikernel.utils.logging import get_logger

from .dispatcher import WorkDispatcher
from .worker import WorkWorker

logger = get_logger(__name__)


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
            try:
                await self.dispatcher.run_once()
                await self.worker.run_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.error("execution loop error", exc_info=True)
            await asyncio.sleep(self.config.poll_interval)

    def stop(self) -> None:
        self._stop.set()
