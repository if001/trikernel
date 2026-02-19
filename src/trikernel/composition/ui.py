from __future__ import annotations

import sys
from typing import AsyncIterable, Iterable


class TerminalUI:
    """Terminal-based input/output that can be swapped later."""

    def read_input(self, prompt: str = "> ") -> str:
        return input(prompt)

    def write_output(self, text: str, end: str = "\n") -> None:
        sys.stdout.write(f"{text}{end}")
        sys.stdout.flush()

    def write_stream(self, chunks: Iterable[str], end: str = "\n") -> None:
        for chunk in chunks:
            sys.stdout.write(chunk)
            sys.stdout.flush()
        if end:
            sys.stdout.write(end)
            sys.stdout.flush()

    async def write_stream_async(self, chunks: AsyncIterable[str], end: str = "\n") -> None:
        async for chunk in chunks:
            sys.stdout.write(chunk)
            sys.stdout.flush()
        if end:
            sys.stdout.write(end)
            sys.stdout.flush()
