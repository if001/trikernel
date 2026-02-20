from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class HybridSearchIndex:
    def __init__(self, persist_dir: Path, name: str, embeddings: Embeddings) -> None:
        self._persist_dir = persist_dir
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._name = name
        self._embeddings = embeddings
        self._docs: List[Document] = []
        self._faiss: Optional[FAISS] = None
        self._bm25: Optional[BM25Retriever] = None
        self._load()

    def set_documents(self, docs: Iterable[Document]) -> None:
        self._docs = list(docs)
        self._rebuild_indexes()
        self._persist_docs()

    def upsert_document(
        self, doc: Document, doc_id: str, *, force: bool = False
    ) -> None:
        if not force and self.has_id(doc_id):
            return
        self._docs = [
            existing for existing in self._docs if existing.metadata.get("id") != doc_id
        ]
        self._docs.append(doc)
        self._rebuild_indexes()
        self._persist_docs()

    def has_id(self, doc_id: str) -> bool:
        return any(doc.metadata.get("id") == doc_id for doc in self._docs)

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        if not query:
            return self._filter_docs(self._docs, metadata_filter)[:k]
        retrievers = []
        bm25 = self._ensure_bm25()
        if bm25:
            bm25.k = k
            retrievers.append(bm25)
        if self._faiss:
            retrievers.append(self._faiss.as_retriever(search_kwargs={"k": k}))
        if not retrievers:
            return []
        weights = [0.5] * len(retrievers)
        ensemble = EnsembleRetriever(retrievers=retrievers, weights=weights)
        docs = ensemble.invoke(query)
        return self._filter_docs(docs, metadata_filter)[:k]

    def _ensure_bm25(self) -> Optional[BM25Retriever]:
        if not self._docs:
            return None
        self._bm25 = BM25Retriever.from_documents(self._docs)
        return self._bm25

    def _rebuild_indexes(self) -> None:
        if not self._docs:
            self._faiss = None
            self._bm25 = None
            return
        self._faiss = FAISS.from_documents(self._docs, self._embeddings)
        self._bm25 = BM25Retriever.from_documents(self._docs)
        self._persist_faiss()

    def _persist_faiss(self) -> None:
        if not self._faiss:
            return
        self._faiss.save_local(str(self._faiss_dir()))

    def _persist_docs(self) -> None:
        payload = [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in self._docs
        ]
        self._docs_path().write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _load(self) -> None:
        if self._docs_path().exists():
            raw = json.loads(self._docs_path().read_text(encoding="utf-8"))
            self._docs = [
                Document(
                    page_content=item["page_content"],
                    metadata=item.get("metadata") or {},
                )
                for item in raw
            ]
        if self._faiss_dir().exists():
            self._faiss = FAISS.load_local(
                str(self._faiss_dir()),
                self._embeddings,
                allow_dangerous_deserialization=True,
            )

    def _filter_docs(
        self, docs: Iterable[Document], metadata_filter: Optional[Dict[str, Any]]
    ) -> List[Document]:
        if not metadata_filter:
            return list(docs)
        filtered: List[Document] = []
        for doc in docs:
            metadata = doc.metadata or {}
            matched = True
            for key, value in metadata_filter.items():
                if metadata.get(key) != value:
                    matched = False
                    break
            if matched:
                filtered.append(doc)
        return filtered

    def _docs_path(self) -> Path:
        return self._persist_dir / f"{self._name}_docs.json"

    def _faiss_dir(self) -> Path:
        return self._persist_dir / f"{self._name}_faiss"
