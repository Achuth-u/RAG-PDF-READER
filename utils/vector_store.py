"""Local FAISS vector store management."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Create, load, update, and query a local FAISS index."""

    def __init__(self, store_dir: Path, embeddings) -> None:
        self.store_dir = store_dir
        self.embeddings = embeddings
        self.store_dir.mkdir(parents=True, exist_ok=True)

    @property
    def index_exists(self) -> bool:
        """Return whether a persisted FAISS index exists."""

        return (self.store_dir / "index.faiss").exists() and (
            self.store_dir / "index.pkl"
        ).exists()

    def load(self) -> FAISS | None:
        """Load the local index if available."""

        if not self.index_exists:
            return None

        logger.info("Loading FAISS index from %s", self.store_dir)
        return FAISS.load_local(
            str(self.store_dir),
            self._require_embeddings(),
            allow_dangerous_deserialization=True,
        )

    def create_or_update(self, chunks: list[Document]) -> FAISS:
        """Create a new index or append chunks to the existing index."""

        if not chunks:
            raise ValueError("No chunks were generated from the uploaded PDFs.")

        vector_store = self.load()
        if vector_store is None:
            logger.info("Creating FAISS index with %s chunks", len(chunks))
            vector_store = FAISS.from_documents(
                chunks,
                self._require_embeddings(),
                distance_strategy=DistanceStrategy.COSINE,
            )
        else:
            logger.info("Adding %s chunks to existing FAISS index", len(chunks))
            vector_store.add_documents(chunks)

        vector_store.save_local(str(self.store_dir))
        return vector_store

    def similarity_search(self, query: str, top_k: int) -> list[Document]:
        """Retrieve the most relevant chunks for a query."""

        vector_store = self.load()
        if vector_store is None:
            raise ValueError("No vector database found. Upload PDFs first.")

        return vector_store.similarity_search(query, k=top_k)

    def _require_embeddings(self):
        """Return embeddings or fail with a clear developer-facing error."""

        if self.embeddings is None:
            raise ValueError("Embeddings are required for this vector store operation.")
        return self.embeddings

    def clear(self) -> None:
        """Remove the local index."""

        if self.store_dir.exists():
            shutil.rmtree(self.store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Vector store cleared: %s", self.store_dir)
