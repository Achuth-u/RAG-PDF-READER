"""Streamlit application for PDF question answering with local RAG."""

from __future__ import annotations

import logging
import shutil
import time

import streamlit as st

from utils.chat_history import ChatHistoryManager
from utils.config import (
    CONFIG,
    CSS_FILE,
    RAW_TEXT_DIR,
    UPLOAD_DIR,
    VECTOR_STORE_DIR,
    configure_logging,
    ensure_directories,
)
from utils.embeddings import EmbeddingFactory
from utils.ollama_status import get_ollama_status, is_model_available
from utils.pdf_loader import PDFLoader, PDFUploadInfo
from utils.rag_chain import RAGChain
from utils.text_splitter import DocumentChunker
from utils.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


def app(environ, start_response):
    """Minimal WSGI fallback for hosts that auto-detect Python web apps."""

    status = "200 OK"
    headers = [("Content-Type", "text/html; charset=utf-8")]
    body = (
        "<!doctype html>"
        "<html><head><title>AI PDF RAG Chatbot</title></head>"
        "<body>"
        "<h1>AI PDF RAG Chatbot</h1>"
        "<p>This project is a Streamlit app. Start it with:</p>"
        "<pre>streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT --server.headless=true</pre>"
        "</body></html>"
    ).encode("utf-8")

    if environ.get("PATH_INFO") == "/healthz":
        body = b"ok"

    start_response(status, headers)
    return [body]


application = app
handler = app


def load_css() -> None:
    """Load custom CSS if present."""

    if CSS_FILE.exists():
        css = CSS_FILE.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings(model_name: str):
    """Cache the embedding model across Streamlit reruns."""

    return EmbeddingFactory(model_name).create()


@st.cache_data(ttl=10, show_spinner=False)
def get_cached_ollama_status(base_url: str):
    """Cache Ollama status briefly to keep reruns responsive."""

    return get_ollama_status(base_url)


def initialise_state() -> None:
    """Create Streamlit session state defaults."""

    defaults = {
        "messages": [],
        "uploaded_infos": [],
        "total_pages": 0,
        "total_chunks": 0,
        "last_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def format_bytes(size_bytes: int) -> str:
    """Format bytes for display."""

    size_mb = size_bytes / (1024 * 1024)
    return f"{size_mb:.2f} MB"


def render_header() -> None:
    """Render the top navigation/header."""

    st.markdown(
        """
        <div class="topbar">
            <div>
                <p class="eyebrow">Local RAG Workspace</p>
                <h1>AI PDF Question Answering</h1>
            </div>
            <div class="status-pill">Ollama + FAISS + LangChain</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard(
    uploaded_count: int,
    total_pages: int,
    total_chunks: int,
    selected_model: str,
    vector_ready: bool,
) -> None:
    """Render dashboard status cards."""

    values = [
        ("PDFs", str(uploaded_count)),
        ("Pages", str(total_pages)),
        ("Chunks", str(total_chunks)),
        ("Embeddings", CONFIG.embedding_model.split("/")[-1]),
        ("LLM", selected_model),
        ("Vector Store", "Ready" if vector_ready else "Empty"),
    ]
    columns = st.columns(6)
    for column, (label, value) in zip(columns, values):
        with column:
            st.markdown(
                f"""
                <div class="metric-card">
                    <span>{label}</span>
                    <strong>{value}</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )


def display_upload_summary(infos: list[PDFUploadInfo]) -> None:
    """Show uploaded file metadata."""

    if not infos:
        return

    st.success("Upload successful")
    for info in infos:
        st.markdown(
            f"""
            <div class="upload-card">
                <strong>{info.filename}</strong>
                <span>{info.pages} pages | {format_bytes(info.size_bytes)}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def process_uploads(
    uploaded_files,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """Save PDFs, extract text, split chunks, and update FAISS."""

    if not uploaded_files:
        st.warning("Choose at least one PDF before processing.")
        return

    loader = PDFLoader(UPLOAD_DIR, RAW_TEXT_DIR)
    progress = st.progress(0)
    status = st.empty()

    try:
        status.info("Reading PDFs...")
        upload_infos = [loader.save_upload(file) for file in uploaded_files]
        progress.progress(25)

        status.info("Extracting text...")
        documents = loader.load_many([info.saved_path for info in upload_infos])
        progress.progress(45)

        status.info("Splitting text...")
        chunker = DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = chunker.split(documents)
        progress.progress(65)

        status.info("Generating embeddings and updating FAISS...")
        started = time.perf_counter()
        embeddings = get_embeddings(CONFIG.embedding_model)
        vector_manager = VectorStoreManager(VECTOR_STORE_DIR, embeddings)
        vector_manager.create_or_update(chunks)
        elapsed = time.perf_counter() - started
        progress.progress(100)

        st.session_state.uploaded_infos = upload_infos
        st.session_state.total_pages += sum(info.pages for info in upload_infos)
        st.session_state.total_chunks += len(chunks)
        st.session_state.last_error = None
        logger.info("Embedding/vector update finished in %.2fs", elapsed)
        status.success(f"Vector database updated in {elapsed:.2f}s")
        display_upload_summary(upload_infos)
    except Exception as exc:
        st.session_state.last_error = str(exc)
        logger.exception("Upload processing failed")
        status.error(str(exc))
    finally:
        progress.empty()


def clear_database() -> None:
    """Clear vector store and uploaded PDF files."""

    if VECTOR_STORE_DIR.exists():
        shutil.rmtree(VECTOR_STORE_DIR)
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

    if UPLOAD_DIR.exists():
        for pdf_path in UPLOAD_DIR.glob("*.pdf"):
            pdf_path.unlink(missing_ok=True)

    if RAW_TEXT_DIR.exists():
        for text_path in RAW_TEXT_DIR.glob("*.txt"):
            text_path.unlink(missing_ok=True)

    st.session_state.uploaded_infos = []
    st.session_state.total_pages = 0
    st.session_state.total_chunks = 0
    st.toast("Database cleared")
    logger.info("Database and uploaded PDFs cleared")


def render_sources(sources: list[dict]) -> None:
    """Render source citations under an answer."""

    if not sources:
        return

    with st.expander("Sources", expanded=False):
        for index, source in enumerate(sources, start=1):
            st.markdown(
                f"**{index}. {source['document']}** | Page {source['page']} | "
                f"Chunk {source.get('chunk_id', 'N/A')}"
            )
            st.caption(source.get("excerpt", ""))


def render_chat_history() -> None:
    """Render previous messages."""

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message.get("sources") or [])


def answer_question(
    question: str,
    selected_model: str,
    temperature: float,
    top_k: int,
    embeddings,
) -> None:
    """Run RAG for a user question and update chat state."""

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Generating answer..."):
            try:
                vector_manager = VectorStoreManager(VECTOR_STORE_DIR, embeddings)
                rag = RAGChain(
                    vector_store=vector_manager,
                    model=selected_model,
                    temperature=temperature,
                    top_k=top_k,
                    ollama_base_url=CONFIG.ollama_base_url,
                    num_predict=CONFIG.ollama_num_predict,
                )
                response = rag.ask(question)
                sources = ChatHistoryManager.source_payload(response.sources)
                st.markdown(response.answer)
                render_sources(sources)
                st.caption(
                    f"Retrieval: {response.retrieval_time:.2f}s | "
                    f"Response: {response.response_time:.2f}s"
                )
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response.answer,
                        "sources": sources,
                    }
                )
            except Exception as exc:
                logger.exception("Question answering failed")
                message = str(exc)
                st.error(message)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": message,
                        "sources": [],
                    }
                )


def sidebar_controls():
    """Render sidebar controls and return selected settings."""

    with st.sidebar:
        st.title("Controls")

        uploaded_files = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            help=f"Maximum recommended upload: {CONFIG.max_upload_mb} MB",
        )

        default_model = (
            CONFIG.default_llm_model
            if CONFIG.default_llm_model in CONFIG.allowed_llm_models
            else CONFIG.allowed_llm_models[0]
        )
        if st.session_state.get("selected_llm_model") not in CONFIG.allowed_llm_models:
            st.session_state.selected_llm_model = default_model

        selected_model = st.selectbox(
            "Model",
            CONFIG.allowed_llm_models,
            index=CONFIG.allowed_llm_models.index(default_model),
            key="selected_llm_model",
        )
        chunk_size = st.slider(
            "Chunk Size",
            min_value=300,
            max_value=2500,
            value=CONFIG.chunk_size,
            step=100,
        )
        chunk_overlap = st.slider(
            "Chunk Overlap",
            min_value=0,
            max_value=min(800, chunk_size - 50),
            value=min(CONFIG.chunk_overlap, chunk_size - 50),
            step=50,
        )
        top_k = st.slider("Top K", min_value=1, max_value=10, value=CONFIG.top_k, step=1)
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=CONFIG.temperature,
            step=0.05,
        )

        st.divider()
        process_clicked = st.button("Process PDFs", type="primary", use_container_width=True)
        clear_db_clicked = st.button("Clear Database", use_container_width=True)
        clear_chat_clicked = st.button("Clear Chat", use_container_width=True)

        st.divider()
        vector_ready = VectorStoreManager(VECTOR_STORE_DIR, None).index_exists
        ollama_status = get_cached_ollama_status(CONFIG.ollama_base_url)
        model_ready = is_model_available(selected_model, ollama_status.models)
        st.write("System Status")
        st.caption(f"Vector store: {'Ready' if vector_ready else 'Empty'}")
        st.caption(f"Embedding model: {CONFIG.embedding_model}")
        st.caption(f"Selected model: {selected_model}")
        st.caption(f"Ollama URL: {CONFIG.ollama_base_url}")
        if ollama_status.reachable:
            st.caption(f"Ollama: {'Ready' if model_ready else 'Model missing'}")
            if ollama_status.models:
                st.caption(f"Installed models: {', '.join(ollama_status.models)}")
        else:
            st.caption("Ollama: Not reachable")
            if ollama_status.error:
                st.caption(ollama_status.error)

        if st.session_state.last_error:
            st.error(st.session_state.last_error)

        if clear_db_clicked:
            clear_database()
            st.rerun()

        if clear_chat_clicked:
            st.session_state.messages = []
            st.rerun()

        export = ChatHistoryManager.to_markdown(st.session_state.messages)
        st.download_button(
            "Download Chat",
            data=export,
            file_name="pdf-rag-chat.md",
            mime="text/markdown",
            use_container_width=True,
        )

    return (
        uploaded_files,
        selected_model,
        chunk_size,
        chunk_overlap,
        top_k,
        temperature,
        process_clicked,
    )


def main() -> None:
    """Run the Streamlit app."""

    ensure_directories()
    configure_logging()

    st.set_page_config(
        page_title=CONFIG.app_title,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    load_css()
    initialise_state()

    (
        uploaded_files,
        selected_model,
        chunk_size,
        chunk_overlap,
        top_k,
        temperature,
        process_clicked,
    ) = sidebar_controls()

    vector_manager = VectorStoreManager(VECTOR_STORE_DIR, None)

    render_header()
    render_dashboard(
        uploaded_count=len(list(UPLOAD_DIR.glob("*.pdf"))),
        total_pages=st.session_state.total_pages,
        total_chunks=st.session_state.total_chunks,
        selected_model=selected_model,
        vector_ready=vector_manager.index_exists,
    )

    if process_clicked:
        process_uploads(uploaded_files, chunk_size, chunk_overlap)

    st.markdown('<section class="chat-shell">', unsafe_allow_html=True)
    render_chat_history()
    st.markdown("</section>", unsafe_allow_html=True)

    prompt = st.chat_input("Ask a question about your uploaded PDFs")
    if prompt:
        if not vector_manager.index_exists:
            st.warning("Upload and process at least one PDF before asking questions.")
        else:
            ollama_status = get_cached_ollama_status(CONFIG.ollama_base_url)
            if not ollama_status.reachable:
                st.warning(
                    "Ollama is not reachable. Start Ollama with `ollama serve` "
                    f"or set `OLLAMA_BASE_URL` to a running server. Current URL: "
                    f"`{CONFIG.ollama_base_url}`"
                )
            elif not is_model_available(selected_model, ollama_status.models):
                st.warning(
                    f"`{selected_model}` is not installed in Ollama. "
                    f"Run `ollama pull {selected_model}` and try again."
                )
            else:
                embeddings = get_embeddings(CONFIG.embedding_model)
                answer_question(prompt, selected_model, temperature, top_k, embeddings)

    st.markdown(
        """
        <footer>
            Built for local, citation-first document QA. Answers are constrained to uploaded PDFs.
        </footer>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
