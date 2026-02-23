from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_core.tools import BaseTool

from ..models import ToolContext
from .structured_tools import build_structured_tool


def _workspace_root() -> Path:
    load_dotenv()
    root = os.environ.get("work_space_dir")
    if not root:
        raise ValueError("work_space_dir is not set")
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists() or not root_path.is_dir():
        raise ValueError("work_space_dir is not a valid directory")
    return root_path


def _resolve_path(path: str) -> Path:
    root = _workspace_root()
    input_path = Path(path) if path else root
    resolved = input_path if input_path.is_absolute() else root / input_path
    resolved = resolved.expanduser().resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("path is outside work_space_dir") from exc
    return resolved


def _ensure_file_size(path: Path, max_bytes: int) -> None:
    if max_bytes <= 0:
        return
    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError("file is too large to read")


class TreeArgs(BaseModel):
    path: str = ""
    max_depth: int = 2
    include_files: bool = True
    max_entries: int = 200


class StatArgs(BaseModel):
    path: str


class FindArgs(BaseModel):
    path: str = ""
    name_pattern: str = "*"
    max_depth: int = 3
    file_type: str = "any"
    max_results: int = 200


class RgArgs(BaseModel):
    pattern: str
    path: str = ""
    ignore_case: bool = False
    max_matches: int = 100
    file_glob: Optional[str] = None


class HeadArgs(BaseModel):
    path: str
    lines: int = 10
    max_bytes: int = 1_000_000


class TailArgs(BaseModel):
    path: str
    lines: int = 10
    max_bytes: int = 1_000_000


class ReadFileArgs(BaseModel):
    path: str
    max_bytes: int = 1_000_000


def tree(
    path: str = "",
    max_depth: int = 2,
    include_files: bool = True,
    max_entries: int = 200,
    *,
    context: ToolContext,
) -> Dict[str, Any]:
    _ = context
    root = _resolve_path(path)
    if not root.exists():
        return {"error": "path_not_found"}
    lines: List[str] = []
    entry_count = 0

    def walk(current: Path, depth: int) -> None:
        nonlocal entry_count
        if depth > max_depth or entry_count >= max_entries:
            return
        try:
            entries = sorted(current.iterdir(), key=lambda p: (not p.is_dir(), p.name))
        except PermissionError:
            lines.append(f"{current}: [permission denied]")
            entry_count += 1
            return
        for entry in entries:
            if entry_count >= max_entries:
                return
            rel = entry.relative_to(root)
            if entry.is_dir():
                lines.append(f"{rel}/")
                entry_count += 1
                walk(entry, depth + 1)
            elif include_files:
                lines.append(str(rel))
                entry_count += 1

    if root.is_dir():
        walk(root, 0)
    else:
        lines.append(root.name)
    return {
        "path": str(root),
        "entries": lines,
        "truncated": entry_count >= max_entries,
    }


def stat(
    path: str,
    *,
    context: ToolContext,
) -> Dict[str, Any]:
    _ = context
    target = _resolve_path(path)
    if not target.exists():
        return {"error": "path_not_found"}
    info = target.stat()
    return {
        "path": str(target),
        "is_file": target.is_file(),
        "is_dir": target.is_dir(),
        "size": info.st_size,
        "modified_at": datetime.fromtimestamp(
            info.st_mtime, tz=timezone.utc
        ).isoformat(),
    }


def find(
    path: str = "",
    name_pattern: str = "*",
    max_depth: int = 3,
    file_type: str = "any",
    max_results: int = 200,
    *,
    context: ToolContext,
) -> Dict[str, Any]:
    _ = context
    root = _resolve_path(path)
    if not root.exists():
        return {"error": "path_not_found"}
    results: List[str] = []
    for current_root, dirs, files in os.walk(root):
        rel_root = Path(current_root).relative_to(root)
        depth = len(rel_root.parts)
        if depth > max_depth:
            dirs[:] = []
            continue
        candidates: List[Path] = []
        if file_type in ("any", "dir"):
            candidates.extend(Path(current_root) / d for d in dirs)
        if file_type in ("any", "file"):
            candidates.extend(Path(current_root) / f for f in files)
        for candidate in candidates:
            if len(results) >= max_results:
                return {"paths": results, "truncated": True}
            if not candidate.match(name_pattern):
                continue
            results.append(str(candidate.relative_to(root)))
    return {"paths": results, "truncated": False}


def rg(
    pattern: str,
    path: str = "",
    ignore_case: bool = False,
    max_matches: int = 100,
    file_glob: Optional[str] = None,
    *,
    context: ToolContext,
) -> Dict[str, Any]:
    _ = context
    root = _resolve_path(path)
    if not root.exists():
        return {"error": "path_not_found"}
    flags = re.IGNORECASE if ignore_case else 0
    try:
        regex = re.compile(pattern, flags)
    except re.error as exc:
        return {"error": f"invalid_pattern: {exc}"}
    matches: List[Dict[str, Any]] = []
    files: List[Path]
    if root.is_file():
        files = [root]
    else:
        glob_pattern = file_glob or "**/*"
        files = [p for p in root.glob(glob_pattern) if p.is_file()]
    for file_path in files:
        try:
            content = file_path.read_text(
                encoding="utf-8", errors="ignore"
            ).splitlines()
        except OSError:
            continue
        for idx, line in enumerate(content, start=1):
            if regex.search(line):
                matches.append(
                    {
                        "path": str(file_path.relative_to(root)),
                        "line_number": idx,
                        "line": line,
                    }
                )
                if len(matches) >= max_matches:
                    return {"matches": matches, "truncated": True}
    return {"matches": matches, "truncated": False}


def head(
    path: str,
    lines: int = 10,
    max_bytes: int = 1_000_000,
    *,
    context: ToolContext,
) -> Dict[str, Any]:
    _ = context
    target = _resolve_path(path)
    if not target.exists() or not target.is_file():
        return {"error": "file_not_found"}
    try:
        _ensure_file_size(target, max_bytes)
        with target.open("r", encoding="utf-8", errors="ignore") as handle:
            output = [handle.readline().rstrip("\n") for _ in range(max(lines, 0))]
    except ValueError as exc:
        return {"error": str(exc)}
    return {"path": str(target), "lines": output}


def tail(
    path: str,
    lines: int = 10,
    max_bytes: int = 1_000_000,
    *,
    context: ToolContext,
) -> Dict[str, Any]:
    _ = context
    target = _resolve_path(path)
    if not target.exists() or not target.is_file():
        return {"error": "file_not_found"}
    try:
        _ensure_file_size(target, max_bytes)
        content = target.read_text(encoding="utf-8", errors="ignore").splitlines()
        output = content[-max(lines, 0) :] if lines else []
    except ValueError as exc:
        return {"error": str(exc)}
    return {"path": str(target), "lines": output}


def read_file(
    path: str,
    max_bytes: int = 1_000_000,
    *,
    context: ToolContext,
) -> Dict[str, Any]:
    _ = context
    target = _resolve_path(path)
    if not target.exists() or not target.is_file():
        return {"error": "file_not_found"}
    try:
        _ensure_file_size(target, max_bytes)
        content = target.read_text(encoding="utf-8", errors="ignore")
    except ValueError as exc:
        return {"error": str(exc)}
    return {"path": str(target), "content": content}


def build_file_tools() -> List[tuple[BaseTool, Any]]:
    return [
        (
            build_structured_tool(
                tree,
                name="fs.tree",
                description="List files and directories under a path.",
                args_schema=TreeArgs,
            ),
            tree,
        ),
        (
            build_structured_tool(
                stat,
                name="fs.stat",
                description="Get file or directory metadata.",
                args_schema=StatArgs,
            ),
            stat,
        ),
        (
            build_structured_tool(
                find,
                name="fs.find",
                description="Find files or directories by glob pattern.",
                args_schema=FindArgs,
            ),
            find,
        ),
        (
            build_structured_tool(
                rg,
                name="fs.rg",
                description="Search file contents with a regex.",
                args_schema=RgArgs,
            ),
            rg,
        ),
        (
            build_structured_tool(
                head,
                name="fs.head",
                description="Read the first N lines of a file.",
                args_schema=HeadArgs,
            ),
            head,
        ),
        (
            build_structured_tool(
                tail,
                name="fs.tail",
                description="Read the last N lines of a file.",
                args_schema=TailArgs,
            ),
            tail,
        ),
        (
            build_structured_tool(
                read_file,
                name="fs.read_file",
                description="Read a file's content.",
                args_schema=ReadFileArgs,
            ),
            read_file,
        ),
    ]
