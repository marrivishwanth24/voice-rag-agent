"""
main.py — FastAPI app handling Twilio voice webhooks.

Phase 1 MVP: turn-based, using Twilio's own built-in speech recognition
(<Gather input="speech">) and text-to-speech (<Say>). No raw audio streaming,
no separate STT/TTS provider accounts needed to get an end-to-end call
working. Phase 2 upgrades to Twilio Media Streams + Deepgram + ElevenLabs
for lower latency and mid-response interruption (barge-in) — see README.
"""

import logging

from fastapi import FastAPI, Form
from fastapi.responses import Response
from twilio.twiml.voice_response import Gather, VoiceResponse

from app.voice_agent import end_call, handle_turn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Voice RAG Agent")

GREETING = "Hi, I can answer questions about your documents. What would you like to know?"
NO_INPUT = "Sorry, I didn't catch that. Could you say that again?"
GOODBYE = "Thanks for calling. Goodbye."


def _gather(prompt: str) -> Gather:
    gather = Gather(input="speech", action="/voice/process", speech_timeout="auto", method="POST")
    gather.say(prompt)
    return gather


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/voice/incoming")
async def voice_incoming():
    """First webhook Twilio hits when a call connects."""
    vr = VoiceResponse()
    vr.append(_gather(GREETING))
    # Reached only if Gather times out with zero input at all.
    vr.say(GOODBYE)
    vr.hangup()
    return Response(content=str(vr), media_type="application/xml")


@app.post("/voice/process")
async def voice_process(CallSid: str = Form(...), SpeechResult: str = Form(default="")):
    """Handles the caller's transcribed speech, replies, and listens again."""
    vr = VoiceResponse()

    if not SpeechResult.strip():
        vr.append(_gather(NO_INPUT))
        vr.say(GOODBYE)
        vr.hangup()
        return Response(content=str(vr), media_type="application/xml")

    try:
        answer = await handle_turn(CallSid, SpeechResult)
    except Exception:
        logger.exception("Error handling call turn for %s", CallSid)
        answer = "Sorry, something went wrong on my end. Please try again."

    vr.append(_gather(answer))
    # Reached if the caller doesn't say anything else after the answer.
    vr.say(GOODBYE)
    vr.hangup()
    return Response(content=str(vr), media_type="application/xml")


@app.post("/voice/status")
async def voice_status(CallSid: str = Form(...), CallStatus: str = Form(...)):
    """Twilio status callback — clean up conversation state when a call ends."""
    if CallStatus in ("completed", "failed", "busy", "no-answer", "canceled"):
        end_call(CallSid)
    return Response(status_code=204)
