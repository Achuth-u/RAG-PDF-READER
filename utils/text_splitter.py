"""Document chunking utilities."""

from __future__ import annotations

import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

__all__ = ["DocumentChunker"]

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Split documents into overlapping chunks."""

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("Chunk overlap must be smaller than chunk size.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def split(self, documents: list[Document]) -> list[Document]:
        """Return chunked documents with stable chunk indexes."""

        chunks = self.splitter.split_documents(documents)
        for index, chunk in enumerate(chunks, start=1):
            chunk.metadata["chunk_id"] = index

        logger.info(
            "Chunk creation complete: source_pages=%s | chunks=%s | size=%s | overlap=%s",
            len(documents),
            len(chunks),
            self.chunk_size,
            self.chunk_overlap,
        )
        return chunks
