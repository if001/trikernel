from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class UserRequestPayload:
    user_message: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WorkPayload:
    message: str
    run_at: Optional[str] = None
    repeat_interval_seconds: Optional[int] = None
    repeat_enabled: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
