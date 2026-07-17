"""RAG orchestration with Ollama."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from utils.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

FALLBACK_ANSWER = "I couldn't find this information in the uploaded documents."

SYSTEM_PROMPT = """You are an AI assistant.

Answer ONLY using the provided context.
If the answer is not found in the context, say:
"I couldn't find this information in the uploaded documents."

Do not use outside knowledge. Include a concise answer before sources.

Context:
{context}

Question:
{question}

Answer:"""


@dataclass(frozen=True)
class RAGResponse:
    """Generated answer plus retrieved source chunks."""

    answer: str
    sources: list[Document]
    retrieval_time: float
    response_time: float


class RAGChain:
    """Retrieve relevant chunks and answer with a local Ollama model."""

    def __init__(
        self,
        vector_store: VectorStoreManager,
        model: str,
        temperature: float,
        top_k: int,
        ollama_base_url: str,
        num_predict: int,
    ) -> None:
        self.vector_store = vector_store
        self.model = model
        self.temperature = temperature
        self.top_k = top_k
        self.ollama_base_url = ollama_base_url
        self.num_predict = num_predict
        self.prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)

    def ask(self, question: str) -> RAGResponse:
        """Answer a user question using retrieved context only."""

        if not question.strip():
            raise ValueError("Please enter a question.")

        retrieval_started = time.perf_counter()
        sources = self.vector_store.similarity_search(question, self.top_k)
        retrieval_time = time.perf_counter() - retrieval_started

        context = self._format_context(sources)
        if not context.strip():
            return RAGResponse(FALLBACK_ANSWER, [], retrieval_time, 0.0)

        llm = ChatOllama(
            model=self.model,
            temperature=self.temperature,
            base_url=self.ollama_base_url,
            num_predict=self.num_predict,
        )
        messages = self.prompt.format_messages(context=context, question=question)

        response_started = time.perf_counter()
        try:
            response = llm.invoke(messages)
        except Exception as exc:
            logger.exception("Ollama response failed")
            raise RuntimeError(
                "Ollama is not reachable or the selected model is not installed. "
                f"Run `ollama serve`, then `ollama pull {self.model}`."
            ) from exc

        response_time = time.perf_counter() - response_started
        answer = str(response.content).strip() or FALLBACK_ANSWER

        logger.info(
            "RAG response complete: model=%s | top_k=%s | retrieval=%.2fs | response=%.2fs",
            self.model,
            self.top_k,
            retrieval_time,
            response_time,
        )
        return RAGResponse(answer, sources, retrieval_time, response_time)

    @staticmethod
    def _format_context(documents: list[Document]) -> str:
        blocks = []
        for index, document in enumerate(documents, start=1):
            source = document.metadata.get("source", "Unknown document")
            page = document.metadata.get("page", "Unknown page")
            content = document.page_content.strip()
            blocks.append(f"[Source {index}: {source}, page {page}]\n{content}")
        return "\n\n".join(blocks)
