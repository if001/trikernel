from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol

from ..models import Artifact


class ArtifactStoreProtocol(Protocol):
    def write(
        self, media_type: str, body: str, metadata: Dict[str, Any]
    ) -> Artifact: ...

    def read(self, artifact_id: str) -> Optional[Artifact]: ...

    def write_named(
        self, artifact_id: str, media_type: str, body: str, metadata: Dict[str, Any]
    ) -> Artifact: ...

    def list(self) -> List[Artifact]: ...

    def search(self, query: Dict[str, Any]) -> Iterable[Artifact]: ...

    def artifact_path(self, artifact_id: str) -> Path: ...

