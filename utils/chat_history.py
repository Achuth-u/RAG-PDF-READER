"""Chat history export helpers."""

from __future__ import annotations

from datetime import datetime


class ChatHistoryManager:
    """Serialize Streamlit chat messages for download."""

    @staticmethod
    def to_markdown(messages: list[dict]) -> str:
        """Export messages and citations as Markdown."""

        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [f"# PDF RAG Chat Export\n\nExported: {created_at}\n"]

        for message in messages:
            role = "User" if message["role"] == "user" else "Assistant"
            lines.append(f"\n## {role}\n\n{message['content']}\n")
            sources = message.get("sources") or []
            if sources:
                lines.append("\n### Sources\n")
                for source in sources:
                    lines.append(
                        f"- {source['document']} | Page {source['page']} | "
                        f"Chunk {source.get('chunk_id', 'N/A')}\n"
                    )

        return "".join(lines)

    @staticmethod
    def source_payload(documents) -> list[dict]:
        """Convert LangChain source documents into JSON-friendly dictionaries."""

        payload = []
        for document in documents:
            payload.append(
                {
                    "document": document.metadata.get("source", "Unknown document"),
                    "page": document.metadata.get("page", "Unknown page"),
                    "chunk_id": document.metadata.get("chunk_id", "N/A"),
                    "excerpt": document.page_content[:700],
                }
            )
        return payload
