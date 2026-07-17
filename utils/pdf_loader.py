"""PDF loading, cleaning, and metadata extraction."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PDFUploadInfo:
    """Metadata shown in the upload summary."""

    filename: str
    pages: int
    size_bytes: int
    saved_path: Path

    @property
    def size_mb(self) -> float:
        """Return file size in megabytes."""

        return self.size_bytes / (1024 * 1024)


class PDFLoader:
    """Load PDFs into LangChain documents with page-level metadata."""

    def __init__(self, upload_dir: Path, raw_text_dir: Path | None = None) -> None:
        self.upload_dir = upload_dir
        self.raw_text_dir = raw_text_dir
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        if self.raw_text_dir is not None:
            self.raw_text_dir.mkdir(parents=True, exist_ok=True)

    def save_upload(self, uploaded_file) -> PDFUploadInfo:
        """Persist a Streamlit uploaded PDF and return summary metadata."""

        filename = Path(uploaded_file.name).name
        if not filename.lower().endswith(".pdf"):
            raise ValueError(f"{filename} is not a PDF file.")

        saved_path = self.upload_dir / filename
        saved_path.write_bytes(uploaded_file.getbuffer())
        size_bytes = saved_path.stat().st_size
        pages = self._count_pages(saved_path)
        logger.info("PDF uploaded: %s | pages=%s | bytes=%s", filename, pages, size_bytes)

        return PDFUploadInfo(
            filename=filename,
            pages=pages,
            size_bytes=size_bytes,
            saved_path=saved_path,
        )

    def load_pdf(self, path: Path) -> list[Document]:
        """Extract text from each non-empty page in a PDF."""

        try:
            reader = PdfReader(str(path))
        except Exception as exc:
            logger.exception("Failed to open PDF: %s", path)
            raise ValueError(f"Could not read {path.name}. The PDF may be corrupted.") from exc

        documents: list[Document] = []
        for page_index, page in enumerate(reader.pages, start=1):
            raw_text = page.extract_text() or ""
            text = self.clean_text(raw_text)
            if not text:
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": path.name,
                        "page": page_index,
                        "path": str(path),
                    },
                )
            )

        if not documents:
            raise ValueError(f"{path.name} does not contain extractable text.")

        self._save_raw_text(path, documents)
        logger.info("PDF loaded: %s | extracted_pages=%s", path.name, len(documents))
        return documents

    def load_many(self, paths: list[Path]) -> list[Document]:
        """Load multiple PDFs and continue only when all are valid."""

        documents: list[Document] = []
        for path in paths:
            documents.extend(self.load_pdf(path))
        return documents

    @staticmethod
    def clean_text(text: str) -> str:
        """Normalize whitespace while preserving paragraph boundaries."""

        text = text.replace("\x00", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" *\n *", "\n", text)
        return text.strip()

    @staticmethod
    def _count_pages(path: Path) -> int:
        try:
            return len(PdfReader(str(path)).pages)
        except Exception as exc:
            logger.exception("Failed to count PDF pages: %s", path)
            raise ValueError(f"Could not inspect {path.name}.") from exc

    def _save_raw_text(self, path: Path, documents: list[Document]) -> None:
        """Persist extracted text for inspection and portfolio transparency."""

        if self.raw_text_dir is None:
            return

        sections = []
        for document in documents:
            page = document.metadata.get("page", "Unknown page")
            sections.append(f"\n\n--- Page {page} ---\n\n{document.page_content}")

        raw_text_path = self.raw_text_dir / f"{path.stem}.txt"
        raw_text_path.write_text("".join(sections).strip(), encoding="utf-8")
