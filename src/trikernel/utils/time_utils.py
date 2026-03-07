from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo


def get_timezone() -> ZoneInfo:
    tz_name = os.environ.get("TRIKERNEL_TIMEZONE", "Asia/Tokyo")
    try:
        return ZoneInfo(tz_name)
    except Exception:
        return ZoneInfo("Asia/Tokyo")


def now_iso() -> str:
    return datetime.now(get_timezone()).isoformat()


def to_timezone(value: datetime) -> datetime:
    tz = get_timezone()
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(tz)


def parse_run_at(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("run_at must be ISO8601 format") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def validate_run_at_future(
    value: str,
    *,
    now: datetime | None = None,
    max_future_days: int | None = 365,
) -> datetime:
    parsed = parse_run_at(value)
    now = now or datetime.now(timezone.utc)
    if parsed <= now:
        raise ValueError("run_at must be in the future")
    if max_future_days is not None:
        if parsed > now + timedelta(days=max_future_days):
            raise ValueError("run_at must be within 1 year")
    return parsed
