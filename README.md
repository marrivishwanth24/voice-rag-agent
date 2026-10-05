# Voice RAG Agent

A phone number that answers questions about your documents. Call in, ask a
question, get a spoken, grounded answer — a voice front-end onto the
[RAG Agent Platform](https://github.com/marrivishwanth24/rag-agent-platform),
not a reimplementation of it.

## What it actually does

- **Answers phone calls** via Twilio and listens for a spoken question.
- **Retrieves grounded context** by calling the RAG platform's live `/query`
  API — the same two-stage pgvector + Voyage rerank pipeline its web app
  uses. This project owns zero retrieval logic; it's a client.
- **Synthesizes a short, spoken-style answer** with a separate Claude call
  tuned for speech (1–3 sentences, no markdown) — distinct from the RAG
  platform's own markdown/citation-formatted synthesis, which isn't meant to
  be read aloud.
- **Holds a conversation** — per-call history, so a caller can ask a
  follow-up question without repeating context.

## Architecture

```
Caller dials the Twilio number
      │
      ▼
Twilio <Gather input="speech">  (Twilio's own speech-to-text)
      │  POSTs the transcript to this app
      ▼
FastAPI webhook (/voice/process)
      │
      ├── rag_client.query_documents()
      │     → POST to the RAG platform's /query (SSE), collected + markdown-stripped
      │
      └── voice_agent.handle_turn()
            → Claude, voice-tuned system prompt → 1-3 spoken sentences
      │
      ▼
TwiML <Say> speaks the answer, then <Gather> listens for a follow-up
```

This is **Phase 1**: turn-based, using Twilio's built-in speech recognition
and TTS (`<Gather>`/`<Say>`) — no raw audio streaming, no extra STT/TTS
accounts needed to get an end-to-end call working. See **Roadmap** below for
Phase 2 (real-time streaming + barge-in).

## Why call the RAG platform's API instead of reimplementing retrieval?

This project and the RAG platform are deliberately separate — separate
repos, separate deploys, separate failure domains. Reusing its public
`/query` endpoint (the same contract its React frontend uses) means this
project has zero duplicated pgvector/embedding/reranking logic, and any
retrieval-quality improvement made there is immediately inherited here with
no code change.

## Why a separate synthesis step, not just read `/query`'s answer aloud?

The RAG platform's `SynthesisAgent` writes markdown — headings, bold text,
bullet lists — formatted for a browser UI with a citations panel. None of
that reads naturally aloud. `rag_client.strip_markdown()` cleans the raw
text, and `voice_agent.handle_turn()` does its own short Claude call with a
voice-specific system prompt, so the final answer is conversational and
brief rather than a wall of browser-formatted text read verbatim.

## Project Structure

```
voice-rag-agent/
├── app/
│   ├── main.py          ← FastAPI app, Twilio webhooks (/voice/incoming, /voice/process, /voice/status)
│   ├── voice_agent.py    ← Per-call conversation state + voice-tuned Claude synthesis
│   └── rag_client.py     ← HTTP client for the RAG platform's /query API + markdown stripping
├── tests/
│   ├── test_main.py        ← Webhook/TwiML tests
│   ├── test_voice_agent.py ← Conversation handling tests
│   └── test_rag_client.py  ← Markdown stripping + SSE client tests (mocked)
├── requirements.txt
├── .env.example
└── pytest.ini
```

## Getting Started

### Prerequisites
- Python 3.11+
- An Anthropic API key
- A session on the [RAG platform](https://rag-frontend-topaz.vercel.app) with at least one document already uploaded
- A [Twilio](https://twilio.com) account (free trial works for testing)

### Local setup

```bash
git clone <this-repo>
cd voice-rag-agent
pip install -r requirements.txt
cp .env.example .env   # fill in ANTHROPIC_API_KEY and RAG_SESSION_ID
uvicorn app.main:app --reload
```

### Getting a `RAG_SESSION_ID`

1. Open the [RAG platform's live demo](https://rag-frontend-topaz.vercel.app), upload a PDF.
2. Open your browser's dev tools → Application/Storage → `localStorage` →
   copy the `sessionId` value.
3. Put it in `.env` as `RAG_SESSION_ID`.

### Connecting a real phone number (Twilio)

1. [console.twilio.com](https://console.twilio.com) → sign up (free trial
   includes credit and a free number).
2. **Phone Numbers → Buy a Number** (trial accounts can call/receive from
   your own verified number without paying).
3. Deploy this app somewhere publicly reachable (Railway, Render, Fly.io —
   or `ngrok http 8000` for local testing).
4. In the Twilio number's **Voice Configuration**, set "A call comes in" to
   **Webhook**, `https://<your-deployed-url>/voice/incoming`, HTTP POST.
5. Call the number.

## Running Tests

```bash
pip install -r requirements.txt
pytest
```

20 tests — Twilio webhook/TwiML generation, conversation state handling, and
the RAG-platform client (markdown stripping + SSE parsing), all mocked at
the network boundary. No live account needed to run the suite.

## Roadmap (Phase 2)

Phase 1 proves the concept end-to-end with the least moving parts. The
honest next step for production quality:

- **Twilio Media Streams** (raw audio over WebSocket) instead of `<Gather>`,
  so processing starts before the caller finishes speaking.
- **Deepgram** streaming STT, **ElevenLabs** streaming TTS — both support
  token/audio-chunk streaming, cutting time-to-first-response well below
  what round-tripping through Twilio's own STT/TTS allows.
- **Barge-in**: let the caller interrupt mid-answer, cancel the in-flight
  response cleanly instead of talking over them.
- Swap the in-memory `_conversations` dict for Redis once running more than
  one server instance (state needs to be shared, not per-process).

## Author

**Vishwanth Marri**
- GitHub: [github.com/marrivishwanth24](https://github.com/marrivishwanth24)
