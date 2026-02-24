from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional


class ToolLLMBase(ABC):
    @abstractmethod
    def generate(self, prompt: str, tools: Optional[List[Any]] = None) -> str: ...
