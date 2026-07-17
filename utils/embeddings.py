"""Embedding model factory."""

from __future__ import annotations

import logging

from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


class EmbeddingFactory:
    """Create sentence-transformer embeddings used by FAISS."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def create(self) -> HuggingFaceEmbeddings:
        """Load the embedding model with normalized vectors for cosine search."""

        logger.info("Loading embedding model: %s", self.model_name)
        return HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
