from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict


@dataclass(frozen=True)
class UserRequestPayload:
    user_message: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WorkPayload:
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
