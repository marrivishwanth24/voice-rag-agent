"""
rag_client.py — thin client for the existing RAG Agent Platform's public API.

This project deliberately does NOT reimplement retrieval. It calls the
already-deployed, already-tested rag-agent-platform over HTTP — the same
/query contract its React frontend uses — instead of duplicating pgvector
search, reranking, or embedding logic. Keeps this project fully independent
(separate repo, separate deploy, separate failure domain) while reusing real,
proven infrastructure rather than rebuilding it.

https://github.com/marrivishwanth24/rag-agent-platform
"""

import os
import re

import httpx

RAG_API_URL = os.getenv("RAG_API_URL", "https://rag-agent-platform-production.up.railway.app")

# A session ID on the RAG platform that already has real documents uploaded
# under it (via its own /upload endpoint). This project doesn't manage
# documents itself — it's a voice front-end onto an existing document space.
RAG_SESSION_ID = os.getenv("RAG_SESSION_ID")

# The RAG platform's SynthesisAgent writes markdown for a browser UI (bold,
# headings, bullet lists). None of that should be read aloud by a TTS engine,
# so it's stripped here rather than asking the other project to change its
# prompt for a client it doesn't know about.
_MARKDOWN_PATTERNS = [
    (re.compile(r"\*\*(.+?)\*\*"), r"\1"),          # **bold**
    (re.compile(r"\*(.+?)\*"), r"\1"),               # *italic*
    (re.compile(r"^#{1,6}\s*", re.MULTILINE), ""),   # # headings
    (re.compile(r"^[-*]\s+", re.MULTILINE), ""),     # - bullet markers
    (re.compile(r"`([^`]+)`"), r"\1"),               # `inline code`
]


def strip_markdown(text: str) -> str:
    for pattern, repl in _MARKDOWN_PATTERNS:
        text = pattern.sub(repl, text)
    return re.sub(r"\n{2,}", " ", text).strip()


class RagClientError(RuntimeError):
    """Raised when the RAG platform can't be reached or isn't configured."""


async def query_documents(question: str, document_ids: list[str] | None = None) -> str:
    """Ask the RAG platform a question; return a plain-text, markdown-stripped answer.

    Consumes the same SSE stream the browser frontend does, but collects it
    into a single string instead of forwarding tokens one at a time — the
    voice loop needs the full answer before it can speak it (Phase 1 is
    turn-based, not real-time streaming; see README for the Phase 2 plan).
    """
    if not RAG_SESSION_ID:
        raise RagClientError(
            "RAG_SESSION_ID is not configured — set it to a session on the RAG "
            "platform that already has documents uploaded under it."
        )

    url = f"{RAG_API_URL}/query"
    headers = {"X-Session-ID": RAG_SESSION_ID, "Content-Type": "application/json"}
    body = {"question": question, "document_ids": document_ids or []}

    chunks: list[str] = []
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.stream("POST", url, headers=headers, json=body) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[len("data: "):]
                    if payload.startswith("[CITATIONS]") or payload == "[DONE]":
                        continue
                    chunks.append(payload)
    except httpx.HTTPError as e:
        raise RagClientError(f"RAG platform request failed: {e}") from e

    return strip_markdown("".join(chunks))
