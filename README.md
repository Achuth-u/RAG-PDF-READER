# AI PDF Question Answering Chatbot Using RAG

A beginner-friendly Retrieval-Augmented Generation web app for asking questions about uploaded PDFs. It uses Streamlit for the interface, LangChain for the RAG pipeline, sentence-transformers for embeddings, FAISS for local vector search, and Ollama for local LLM answers.

The assistant is instructed to answer only from the uploaded documents and show source citations with document names, page numbers, and retrieved excerpts.

## Features

- Upload and process multiple PDF files.
- Extract page-level text with metadata.
- Save extracted raw text locally for inspection.
- Clean whitespace and skip empty pages.
- Split documents with `RecursiveCharacterTextSplitter`.
- Generate local embeddings with `sentence-transformers/all-MiniLM-L6-v2`.
- Store and update a local FAISS vector database.
- Ask questions through a modern Streamlit chat interface.
- Select `llama3`, `mistral`, or `gemma` from the sidebar.
- Configure chunk size, overlap, top K, and temperature.
- Display source citations for every answer.
- Clear chat, clear database, and download chat history as Markdown.
- Log upload, chunking, retrieval, response timing, and errors.

## Project Structure

```text
.
+-- app.py
+-- requirements.txt
+-- README.md
+-- .env
+-- .streamlit/
|   +-- config.toml
+-- assets/
+-- css/
|   +-- style.css
+-- data/
|   +-- uploaded_pdfs/
|   +-- raw_text/
+-- database/
|   +-- vector_store/
+-- logs/
+-- utils/
    +-- chat_history.py
    +-- config.py
    +-- embeddings.py
    +-- pdf_loader.py
    +-- rag_chain.py
    +-- text_splitter.py
    +-- vector_store.py
```

## Requirements

- Python 3.10 or newer
- Ollama installed and running
- At least one local Ollama model pulled:

```bash
ollama pull llama3
ollama pull mistral
ollama pull gemma
```

The embedding model may download the first time it runs, so an internet connection is needed for the initial setup unless the model is already cached locally.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If you already installed dependencies before `torchvision` was added, update the environment:

```bash
pip install torch torchvision
```

Start Ollama in a separate terminal if it is not already running:

```bash
ollama serve
```

Run the app:

```bash
streamlit run app.py
```

Open the local URL Streamlit prints, usually `http://localhost:8501`.

## Environment Variables

Defaults live in `.env`:

```env
APP_TITLE="AI PDF RAG Chatbot"
DEFAULT_LLM_MODEL="tinyllama"
ALLOWED_LLM_MODELS="tinyllama,llama3,mistral,gemma"
OLLAMA_BASE_URL="http://127.0.0.1:11434"
OLLAMA_NUM_PREDICT=512
EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K=4
TEMPERATURE=0.1
MAX_UPLOAD_MB=200
```

## How RAG Works

1. Users upload one or more PDFs.
2. The app extracts text page by page.
3. Text is cleaned and split into overlapping chunks.
4. Each chunk is converted into an embedding.
5. FAISS stores embeddings locally for fast similarity search.
6. A user question is embedded and matched against the top K chunks.
7. The retrieved chunks are inserted into a strict prompt.
8. Ollama generates an answer using only the retrieved context.
9. The answer is shown with source citations.

## Architecture

```text
PDF Upload
  -> PDF Loader
  -> Text Extraction
  -> Chunking
  -> Embeddings
  -> FAISS Vector Store
  -> Question
  -> Similarity Search
  -> Prompt Template
  -> Ollama
  -> Answer + Sources
```

## Error Handling

The app catches and reports common problems:

- Corrupted or unreadable PDFs
- Empty PDFs with no extractable text
- Missing vector database
- Chunk overlap larger than chunk size
- Ollama not running
- Selected Ollama model not installed
- Embedding or vector store errors

Logs are written to `logs/app.log`.

## Deployment Notes

### Streamlit Community Cloud

Streamlit Community Cloud does not run a local Ollama service by default. To deploy there, replace `ChatOllama` with a hosted model endpoint or connect to a reachable Ollama server.

### Render

Use a Python web service for Streamlit and provide access to an Ollama service. For production, keep the vector database on persistent disk.

### Railway

Run Streamlit as the web process and configure an Ollama service separately. Add persistent storage for `database/vector_store`.

### Docker

Build the image:

```bash
docker build -t pdf-rag-chatbot .
```

Run it:

```bash
docker run -p 8501:8501 pdf-rag-chatbot
```

For Ollama inside Docker-based deployments, run Ollama as a separate service and make it reachable from the Streamlit container.

## Future Improvements

- PDF preview with highlighted source passages
- OCR for scanned PDFs
- Keyword search alongside vector search
- Saved sessions and multiple collections
- Document summaries
- Quiz and flashcard generators
- Voice input and voice output
- Feedback buttons and admin analytics

## License

This project is provided as a portfolio-ready educational starter. Add your preferred license before publishing.
