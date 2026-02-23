from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    NamespaceMatchType,
    PutOp,
    SearchOp,
    SearchItem,
)

from trikernel.state_kernel.models import utc_now
from trikernel.utils.search import HybridSearchIndex

Namespace = Tuple[str, ...]


@dataclass(frozen=True)
class MemoryStoreConfig:
    data_dir: Path
    data_file: Path
    index_dir: Path
    enable_search: bool
    embed_model: str
    base_url: str


def load_memory_store_config(data_dir: Optional[Path] = None) -> MemoryStoreConfig:
    load_dotenv()
    base_dir = data_dir or Path(".state")
    file_path = Path(
        os.environ.get(
            "TRIKERNEL_MEMORY_STORE_PATH", str(base_dir / "memory_store.json")
        )
    )
    index_dir = Path(
        os.environ.get(
            "TRIKERNEL_MEMORY_INDEX_DIR", str(base_dir / "memory_index")
        )
    )
    enable_search = os.environ.get("TRIKERNEL_MEMORY_SEARCH", "1") != "0"
    embed_model = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    return MemoryStoreConfig(
        data_dir=base_dir,
        data_file=file_path,
        index_dir=index_dir,
        enable_search=enable_search,
        embed_model=embed_model,
        base_url=base_url,
    )


class JsonFileMemoryStore(BaseStore):
    def __init__(self, config: MemoryStoreConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._items: Dict[Namespace, Dict[str, Item]] = {}
        self._index = self._init_index(config) if config.enable_search else None
        self._load()

    def setup(self) -> None:
        return None

    async def asetup(self) -> None:
        return None

    def get(self, namespace: Namespace, key: str) -> Optional[Item]:
        with self._lock:
            return self._items.get(namespace, {}).get(key)

    def put(
        self,
        namespace: Namespace,
        key: str,
        value: Dict[str, Any],
        *,
        index: bool | None = None,
        ttl: Optional[int] = None,
    ) -> Item:
        _ = ttl
        with self._lock:
            ns_map = self._items.setdefault(namespace, {})
            now = utc_now()
            existing = ns_map.get(key)
            created_at = existing.created_at if existing else now
            item = Item(
                namespace=namespace,
                key=key,
                value=value,
                created_at=created_at,
                updated_at=now,
            )
            ns_map[key] = item
            self._persist()
            if index is not False:
                self._index_item(item)
            return item

    def delete(self, namespace: Namespace, key: str) -> None:
        with self._lock:
            ns_map = self._items.get(namespace)
            if not ns_map or key not in ns_map:
                return
            del ns_map[key]
            if not ns_map:
                self._items.pop(namespace, None)
            self._persist()
            if self._index:
                self._index.upsert_document(
                    Document(page_content="", metadata={"id": _doc_id(namespace, key)}),
                    _doc_id(namespace, key),
                    force=True,
                )

    def search(
        self,
        namespace: Namespace,
        query: Optional[str] = None,
        *,
        limit: int = 10,
        offset: int = 0,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchItem]:
        items = list(self._items.get(namespace, {}).values())
        if filter:
            items = [
                item
                for item in items
                if all(item.value.get(key) == value for key, value in filter.items())
            ]
        if not query:
            sliced = items[offset : offset + limit]
            return [SearchItem(item=item, score=1.0) for item in sliced]

        if self._index:
            docs = self._index.search(query, k=limit, metadata_filter=None)
            results: List[SearchItem] = []
            for doc in docs:
                metadata = doc.metadata or {}
                ns = tuple(metadata.get("namespace", []))
                key = metadata.get("key")
                if ns != namespace or not isinstance(key, str):
                    continue
                item = self._items.get(ns, {}).get(key)
                if item:
                    results.append(SearchItem(item=item, score=1.0))
            return results[offset : offset + limit]

        query_lower = query.lower()
        matched = [
            item
            for item in items
            if query_lower in json.dumps(item.value, ensure_ascii=False).lower()
        ]
        sliced = matched[offset : offset + limit]
        return [SearchItem(item=item, score=0.1) for item in sliced]

    def list_namespaces(
        self,
        *,
        prefix: Optional[Namespace] = None,
        max_depth: Optional[int] = None,
        match_type: NamespaceMatchType = "prefix",
        limit: int = 100,
        offset: int = 0,
    ) -> List[Namespace]:
        namespaces = list(self._items.keys())
        if prefix:
            if match_type == "prefix":
                namespaces = [ns for ns in namespaces if ns[: len(prefix)] == prefix]
            else:
                namespaces = [ns for ns in namespaces if ns == prefix]
        if max_depth is not None:
            namespaces = [ns[:max_depth] for ns in namespaces]
        namespaces = sorted(set(namespaces))
        return namespaces[offset : offset + limit]

    def batch(self, ops: List[object]) -> List[Any]:
        results: List[Any] = []
        for op in ops:
            if isinstance(op, GetOp):
                results.append(self.get(op.namespace, op.key))
            elif isinstance(op, PutOp):
                results.append(
                    self.put(
                        op.namespace,
                        op.key,
                        op.value,
                        index=getattr(op, "index", None),
                        ttl=getattr(op, "ttl", None),
                    )
                )
            elif isinstance(op, SearchOp):
                results.append(
                    self.search(
                        op.namespace,
                        op.query,
                        limit=getattr(op, "limit", 10),
                        offset=getattr(op, "offset", 0),
                        filter=getattr(op, "filter", None),
                    )
                )
            elif isinstance(op, ListNamespacesOp):
                results.append(
                    self.list_namespaces(
                        prefix=getattr(op, "prefix", None),
                        max_depth=getattr(op, "max_depth", None),
                        match_type=getattr(op, "match_type", "prefix"),
                        limit=getattr(op, "limit", 100),
                        offset=getattr(op, "offset", 0),
                    )
                )
            else:
                results.append(None)
        return results

    async def aget(self, namespace: Namespace, key: str) -> Optional[Item]:
        return self.get(namespace, key)

    async def aput(
        self,
        namespace: Namespace,
        key: str,
        value: Dict[str, Any],
        *,
        index: bool | None = None,
        ttl: Optional[int] = None,
    ) -> Item:
        return self.put(namespace, key, value, index=index, ttl=ttl)

    async def adelete(self, namespace: Namespace, key: str) -> None:
        self.delete(namespace, key)

    async def asearch(
        self,
        namespace: Namespace,
        query: Optional[str] = None,
        *,
        limit: int = 10,
        offset: int = 0,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchItem]:
        return self.search(
            namespace, query, limit=limit, offset=offset, filter=filter
        )

    async def alist_namespaces(
        self,
        *,
        prefix: Optional[Namespace] = None,
        max_depth: Optional[int] = None,
        match_type: NamespaceMatchType = "prefix",
        limit: int = 100,
        offset: int = 0,
    ) -> List[Namespace]:
        return self.list_namespaces(
            prefix=prefix,
            max_depth=max_depth,
            match_type=match_type,
            limit=limit,
            offset=offset,
        )

    async def abatch(self, ops: List[object]) -> List[Any]:
        return self.batch(ops)

    def _load(self) -> None:
        path = self._config.data_file
        if not path.exists():
            return
        raw = json.loads(path.read_text(encoding="utf-8"))
        for item_data in raw:
            namespace = tuple(item_data.get("namespace", []))
            key = item_data["key"]
            item = Item(
                namespace=namespace,
                key=key,
                value=item_data.get("value", {}),
                created_at=item_data.get("created_at", utc_now()),
                updated_at=item_data.get("updated_at", utc_now()),
            )
            self._items.setdefault(namespace, {})[key] = item
            self._index_item(item)

    def _persist(self) -> None:
        path = self._config.data_file
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = []
        for namespace, items in self._items.items():
            for item in items.values():
                payload.append(
                    {
                        "namespace": list(namespace),
                        "key": item.key,
                        "value": item.value,
                        "created_at": item.created_at,
                        "updated_at": item.updated_at,
                    }
                )
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _index_item(self, item: Item) -> None:
        if not self._index:
            return
        doc_id = _doc_id(item.namespace, item.key)
        content = json.dumps(item.value, ensure_ascii=False)
        doc = Document(
            page_content=content,
            metadata={"id": doc_id, "namespace": list(item.namespace), "key": item.key},
        )
        self._index.upsert_document(doc, doc_id, force=True)

    @staticmethod
    def _init_index(config: MemoryStoreConfig) -> HybridSearchIndex:
        embeddings = OllamaEmbeddings(
            model=config.embed_model,
            base_url=config.base_url,
        )
        config.index_dir.mkdir(parents=True, exist_ok=True)
        return HybridSearchIndex(config.index_dir, "memory", embeddings)


def _doc_id(namespace: Namespace, key: str) -> str:
    return "/".join(namespace) + f"/{key}"


def load_memory_store(data_dir: Optional[Path] = None) -> JsonFileMemoryStore:
    config = load_memory_store_config(data_dir)
    return JsonFileMemoryStore(config)
