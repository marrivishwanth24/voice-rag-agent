"""Tests for voice_agent's per-call conversation handling.

Mocks query_documents (the RAG platform call) and the Claude client, so these
run without hitting any real API.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.rag_client import RagClientError
import app.voice_agent as voice_agent
from app.voice_agent import end_call, handle_turn


def _claude_response(text):
    return SimpleNamespace(content=[SimpleNamespace(text=text)])


@pytest.fixture(autouse=True)
def _clean_conversations():
    """Each test gets an empty conversation store, regardless of call_sid reuse."""
    voice_agent._conversations.clear()
    yield
    voice_agent._conversations.clear()


@pytest.mark.asyncio
async def test_handle_turn_returns_synthesized_answer():
    with patch("app.voice_agent.query_documents", new=AsyncMock(return_value="Paris is the capital.")), \
         patch.object(voice_agent, "_client") as mock_client:
        mock_client.messages.create = AsyncMock(return_value=_claude_response("It's Paris."))

        answer = await handle_turn("CA123", "What is the capital of France?")

    assert answer == "It's Paris."


@pytest.mark.asyncio
async def test_handle_turn_degrades_gracefully_when_rag_platform_unreachable():
    with patch("app.voice_agent.query_documents", new=AsyncMock(side_effect=RagClientError("down"))), \
         patch.object(voice_agent, "_client") as mock_client:
        mock_client.messages.create = AsyncMock(return_value=_claude_response("I don't have that information."))

        answer = await handle_turn("CA123", "anything")

    assert answer == "I don't have that information."
    # Claude still gets called -- told there's no context, not left hanging.
    mock_client.messages.create.assert_called_once()


@pytest.mark.asyncio
async def test_handle_turn_accumulates_history_across_turns():
    with patch("app.voice_agent.query_documents", new=AsyncMock(return_value="ctx")), \
         patch.object(voice_agent, "_client") as mock_client:
        mock_client.messages.create = AsyncMock(return_value=_claude_response("ok"))

        await handle_turn("CA123", "first question")
        await handle_turn("CA123", "second question")

    # user+assistant per turn -> 4 messages after 2 turns
    assert len(voice_agent._conversations["CA123"]) == 4


@pytest.mark.asyncio
async def test_handle_turn_keeps_separate_history_per_call():
    with patch("app.voice_agent.query_documents", new=AsyncMock(return_value="ctx")), \
         patch.object(voice_agent, "_client") as mock_client:
        mock_client.messages.create = AsyncMock(return_value=_claude_response("ok"))

        await handle_turn("CALL-A", "q1")
        await handle_turn("CALL-B", "q2")

    assert "CALL-A" in voice_agent._conversations
    assert "CALL-B" in voice_agent._conversations
    assert len(voice_agent._conversations["CALL-A"]) == 2
    assert len(voice_agent._conversations["CALL-B"]) == 2


def test_end_call_clears_history():
    voice_agent._conversations["CA999"] = [{"role": "user", "content": "x"}]
    end_call("CA999")
    assert "CA999" not in voice_agent._conversations


def test_end_call_is_safe_on_unknown_call_sid():
    end_call("never-seen")  # must not raise
