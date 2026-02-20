from __future__ import annotations

import json
from typing import Any, Dict, Optional

from .models import ToolContext

USER_PROFILE_ARTIFACT_ID = "user_profile"
USER_PROFILE_MEDIA_TYPE = "application/json"


def _require_state_api(context: ToolContext) -> Any:
    if context is None or context.state_api is None:
        raise ValueError("state_api is required in ToolContext")
    return context.state_api


def _load_profile(context: ToolContext) -> Dict[str, Any]:
    state_api = _require_state_api(context)
    artifact = state_api.artifact_read(USER_PROFILE_ARTIFACT_ID)
    if not artifact or not artifact.body:
        return {}
    try:
        return json.loads(artifact.body)
    except json.JSONDecodeError:
        return {}


def user_profile_save(
    name: Optional[str] = None,
    thinking: Optional[str] = None,
    preferences: Optional[str] = None,
    attributes: Optional[str] = None,
    notes: Optional[str] = None,
    merge: bool = True,
    context: ToolContext = None,
) -> Dict[str, Any]:
    state_api = _require_state_api(context)
    profile = _load_profile(context) if merge else {}
    if name is not None:
        profile["name"] = name
    if thinking is not None:
        profile["thinking"] = thinking
    if preferences is not None:
        profile["preferences"] = preferences
    if attributes is not None:
        profile["attributes"] = attributes
    if notes is not None:
        profile["notes"] = notes
    body = json.dumps(profile, ensure_ascii=False)
    state_api.artifact_write_named(
        USER_PROFILE_ARTIFACT_ID,
        USER_PROFILE_MEDIA_TYPE,
        body,
        {"artifact_type": "user_profile"},
    )
    return {"status": "saved", "profile": profile}


def user_profile_load(context: ToolContext = None) -> Dict[str, Any]:
    profile = _load_profile(context)
    return {"profile": profile}


def user_profile_tool_functions() -> Dict[str, Any]:
    return {
        "user.profile.save": user_profile_save,
        "user.profile.load": user_profile_load,
    }
