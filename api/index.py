"""Vercel fallback entrypoint.

The full application is a Streamlit app and should be deployed on a host that
supports long-running Streamlit processes. This handler keeps Vercel builds from
failing when it auto-detects a Python web app.
"""

from __future__ import annotations


def app(environ, start_response):
    """Return a minimal response for Vercel serverless deployments."""

    status = "200 OK"
    path = environ.get("PATH_INFO", "")
    headers = [("Content-Type", "text/html; charset=utf-8")]

    if path == "/healthz":
        body = b"ok"
    else:
        body = (
            "<!doctype html>"
            "<html><head><title>AI PDF RAG Chatbot</title></head>"
            "<body>"
            "<h1>AI PDF RAG Chatbot</h1>"
            "<p>This project is a Streamlit app. Deploy it with:</p>"
            "<pre>streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT --server.headless=true</pre>"
            "</body></html>"
        ).encode("utf-8")

    start_response(status, headers)
    return [body]


application = app
handler = app
