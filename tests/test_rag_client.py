"""Tests for rag_client: markdown stripping (pure) and the SSE-to-text client
(mocked httpx — no real network call to the RAG platform)."""

from unittest.mock import patch

import pytest

from app.rag_client import RagClientError, query_documents, strip_markdown


def test_strip_markdown_removes_bold_and_italic():
    assert strip_markdown("This is **bold** and *italic* text.") == "This is bold and italic text."


def test_strip_markdown_removes_headings_and_bullets():
    text = "# Heading\n- first point\n- second point"
    assert "#" not in strip_markdown(text)
    assert "- " not in strip_markdown(text)


def test_strip_markdown_removes_inline_code():
    assert strip_markdown("Run `pytest` to test.") == "Run pytest to test."


def test_strip_markdown_collapses_blank_lines():
    assert strip_markdown("Line one.\n\n\nLine two.") == "Line one. Line two."


# ── query_documents (mocked SSE stream) ─────────────────────────────────────

class _FakeStreamResponse:
    def __init__(self, lines):
        self._lines = lines

    def raise_for_status(self):
        pass

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _FakeStreamCM:
    def __init__(self, lines):
        self._lines = lines

    async def __aenter__(self):
        return _FakeStreamResponse(self._lines)

    async def __aexit__(self, *exc_info):
        return False


class _FakeAsyncClient:
    def __init__(self, lines):
        self._lines = lines

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    def stream(self, method, url, headers=None, json=None):
        return _FakeStreamCM(self._lines)


def _sse_lines(*tokens, citations="[]"):
    lines = [f"data: {t}" for t in tokens]
    lines.append(f"data: [CITATIONS]{citations}")
    lines.append("data: [DONE]")
    return lines


@pytest.mark.asyncio
async def test_query_documents_assembles_and_cleans_stream(monkeypatch):
    monkeypatch.setattr("app.rag_client.RAG_SESSION_ID", "session-123")
    lines = _sse_lines("The ", "**answer**", " is here.")

    with patch("httpx.AsyncClient", lambda *a, **kw: _FakeAsyncClient(lines)):
        result = await query_documents("What is the answer?")

    assert result == "The answer is here."


@pytest.mark.asyncio
async def test_query_documents_raises_without_session_id(monkeypatch):
    monkeypatch.setattr("app.rag_client.RAG_SESSION_ID", None)
    with pytest.raises(RagClientError):
        await query_documents("anything")


@pytest.mark.asyncio
async def test_query_documents_ignores_citations_and_done_markers(monkeypatch):
    monkeypatch.setattr("app.rag_client.RAG_SESSION_ID", "session-123")
    lines = _sse_lines("Just the text.", citations='[{"filename":"a.pdf"}]')

    with patch("httpx.AsyncClient", lambda *a, **kw: _FakeAsyncClient(lines)):
        result = await query_documents("q")

    assert "CITATIONS" not in result
    assert "DONE" not in result
    assert result == "Just the text."
