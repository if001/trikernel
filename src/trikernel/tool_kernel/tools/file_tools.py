from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from ...utils.env import load_env
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import Field
from typing_extensions import Annotated


def _workspace_root() -> Path:
    load_env()
    root = os.environ.get("work_space_dir")
    if not root:
        raise ValueError("work space dir not set")
    return Path(root)


def _resolve_path(path: str) -> Path:
    base = _workspace_root()
    target = Path(path)
    if target.is_absolute():
        return target
    return (base / target).resolve()


def _format_stat(stat: os.stat_result) -> Dict[str, Any]:
    return {
        "size": stat.st_size,
        "mode": stat.st_mode,
        "mtime": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }


def tree(
    path: Annotated[str, Field(default=".", description="Root path.")],
    max_depth: Annotated[int, Field(default=2, description="Max depth to traverse.")],
) -> Dict[str, Any]:
    target = _resolve_path(path)
    result: List[Dict[str, Any]] = []

    def _walk(current: Path, depth: int) -> None:
        if depth > max_depth:
            return
        for entry in sorted(current.iterdir()):
            info = {
                "path": str(entry),
                "is_dir": entry.is_dir(),
            }
            result.append(info)
            if entry.is_dir():
                _walk(entry, depth + 1)

    _walk(target, 0)
    return {"path": str(target), "entries": result}


def stat(
    path: Annotated[str, Field(..., description="Path to stat.")],
) -> Dict[str, Any]:
    target = _resolve_path(path)
    if not target.exists():
        return {"path": str(target), "exists": False}
    return {"path": str(target), "exists": True, "stat": _format_stat(target.stat())}


def find(
    path: Annotated[str, Field(default=".", description="Root path.")],
    pattern: Annotated[str, Field(default="*", description="Glob pattern.")],
) -> Dict[str, Any]:
    target = _resolve_path(path)
    matches = [str(p) for p in target.rglob(pattern)]
    return {"path": str(target), "matches": matches}


def rg(
    path: Annotated[str, Field(default=".", description="Root path.")],
    pattern: Annotated[str, Field(..., description="Regex pattern.")],
    max_results: Annotated[int, Field(default=20, description="Max results.")],
) -> Dict[str, Any]:
    if not pattern:
        return {"matches": []}
    target = _resolve_path(path)
    regex = re.compile(pattern)
    matches: List[Dict[str, Any]] = []
    for file_path in target.rglob("*"):
        if not file_path.is_file():
            continue
        try:
            content = file_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for idx, line in enumerate(content.splitlines(), start=1):
            if regex.search(line):
                matches.append({"path": str(file_path), "line": idx, "text": line})
                if len(matches) >= max_results:
                    return {"matches": matches}
    return {"matches": matches}


def head(
    path: Annotated[str, Field(..., description="File path.")],
    lines: Annotated[int, Field(default=20, description="Number of lines.")],
) -> Dict[str, Any]:
    target = _resolve_path(path)
    try:
        content = target.read_text(encoding="utf-8").splitlines()[:lines]
    except OSError:
        return {"path": str(target), "content": []}
    return {"path": str(target), "content": content}


def tail(
    path: Annotated[str, Field(..., description="File path.")],
    lines: Annotated[int, Field(default=20, description="Number of lines.")],
) -> Dict[str, Any]:
    target = _resolve_path(path)
    try:
        content = target.read_text(encoding="utf-8").splitlines()[-lines:]
    except OSError:
        return {"path": str(target), "content": []}
    return {"path": str(target), "content": content}


def read_file(
    path: Annotated[str, Field(..., description="File path.")],
) -> Dict[str, Any]:
    target = _resolve_path(path)
    try:
        content = target.read_text(encoding="utf-8")
    except OSError:
        return {"path": str(target), "content": ""}
    return {"path": str(target), "content": content}


def build_file_tools() -> List[BaseTool]:
    return [
        StructuredTool.from_function(
            tree,
            name="fs.tree",
            description="List directory structure under a path up to max_depth. Use to understand project layout before reading files.",
        ),
        StructuredTool.from_function(
            stat,
            name="fs.stat",
            description="Get file/directory metadata (size, timestamps). Use to decide whether/what to read.",
        ),
        StructuredTool.from_function(
            find,
            name="fs.find",
            description="Find paths by glob pattern under a root path (e.g., **/*.py). Use for targeted discovery.",
        ),
        StructuredTool.from_function(
            rg,
            name="fs.rg",
            description="Search file contents using regex (ripgrep). Use to locate definitions/usages quickly; keep patterns specific.",
        ),
        StructuredTool.from_function(
            head,
            name="fs.head",
            description="Read first N lines for quick inspection (logs, configs) without loading whole file.",
        ),
        StructuredTool.from_function(
            tail,
            name="fs.tail",
            description="Read last N lines for quick inspection (logs, configs) without loading whole file.",
        ),
        StructuredTool.from_function(
            read_file,
            name="fs.read_file",
            description="Read full file content. Use only after locating the exact file; avoid huge files.",
        ),
    ]
