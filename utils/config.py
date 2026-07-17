"""Application configuration and path helpers."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploaded_pdfs"
RAW_TEXT_DIR = DATA_DIR / "raw_text"
DATABASE_DIR = BASE_DIR / "database"
VECTOR_STORE_DIR = DATABASE_DIR / "vector_store"
LOG_DIR = BASE_DIR / "logs"
CSS_FILE = BASE_DIR / "css" / "style.css"


def _int_from_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _float_from_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def _tuple_from_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    values = tuple(
        value.strip()
        for value in os.getenv(name, "").split(",")
        if value.strip()
    )
    return values or default


@dataclass(frozen=True)
class AppConfig:
    """Runtime defaults for the Streamlit app."""

    app_title: str = os.getenv("APP_TITLE", "AI PDF RAG Chatbot")
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    default_llm_model: str = os.getenv("DEFAULT_LLM_MODEL", "tinyllama")
    allowed_llm_models: tuple[str, ...] = _tuple_from_env(
        "ALLOWED_LLM_MODELS",
        ("tinyllama", "llama3", "mistral", "gemma"),
    )
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    ollama_num_predict: int = _int_from_env("OLLAMA_NUM_PREDICT", 512)
    chunk_size: int = _int_from_env("CHUNK_SIZE", 1000)
    chunk_overlap: int = _int_from_env("CHUNK_OVERLAP", 200)
    top_k: int = _int_from_env("TOP_K", 4)
    temperature: float = _float_from_env("TEMPERATURE", 0.1)
    max_upload_mb: int = _int_from_env("MAX_UPLOAD_MB", 200)


CONFIG = AppConfig()


def ensure_directories() -> None:
    """Create application directories if they do not exist."""

    for path in (UPLOAD_DIR, RAW_TEXT_DIR, VECTOR_STORE_DIR, LOG_DIR):
        path.mkdir(parents=True, exist_ok=True)


def configure_logging() -> None:
    """Configure file and console logging once."""

    ensure_directories()
    log_path = LOG_DIR / "app.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
