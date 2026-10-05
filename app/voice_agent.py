"""
voice_agent.py — conversation orchestration for a phone call.

Each turn: take the caller's transcribed question, retrieve grounded context
from the RAG platform (via rag_client), then ask Claude for a short,
spoken-style answer. This is a deliberately separate synthesis step from the
RAG platform's own SynthesisAgent — that one writes markdown with citations
for a browser; this one writes 1-3 plain spoken sentences for a phone call.
Same principle as the RAG platform's MCP server: reuse the retrieval, don't
reuse a synthesis prompt tuned for a different medium.
"""

import os

import anthropic

from app.rag_client import query_documents, RagClientError

_client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

_VOICE_SYSTEM = (
    "You are a helpful voice assistant answering phone calls. You'll be given "
    "a caller's question and context retrieved from their documents. "
    "Answer in 1-3 short sentences, in a natural, conversational spoken style. "
    "No markdown, no bullet points, no headings, no asterisks — this will be "
    "read aloud by a text-to-speech engine. If the context doesn't answer the "
    "question, say so briefly and naturally rather than guessing."
)

# Simple in-memory per-call conversation history: CallSid -> list of turns.
# Fine for a single-instance MVP; swap for Redis or a DB before running
# multiple server instances (in-memory state won't be shared across them).
_conversations: dict[str, list[dict]] = {}

MAX_HISTORY_TURNS = 6  # bounds prompt size / cost as a call goes on


async def handle_turn(call_sid: str, caller_question: str) -> str:
    """Process one turn of a call: retrieve, synthesize a spoken reply, return it."""
    history = _conversations.setdefault(call_sid, [])

    try:
        context = await query_documents(caller_question)
    except RagClientError:
        context = ""  # degrade gracefully -- Claude is told context is empty below

    user_msg = (
        f"Caller's question: {caller_question}\n\n"
        f"Retrieved context:\n{context or '(no relevant context found)'}"
    )
    history.append({"role": "user", "content": user_msg})

    resp = await _client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        system=_VOICE_SYSTEM,
        messages=history[-MAX_HISTORY_TURNS:],
    )
    answer = resp.content[0].text
    history.append({"role": "assistant", "content": answer})

    return answer


def end_call(call_sid: str) -> None:
    """Clean up conversation history when a call ends."""
    _conversations.pop(call_sid, None)
