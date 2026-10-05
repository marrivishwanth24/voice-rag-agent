"""Tests for the Twilio webhook endpoints — verifies the generated TwiML,
not real calls. handle_turn/end_call are mocked."""

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_incoming_call_greets_and_gathers():
    r = client.post("/voice/incoming")
    assert r.status_code == 200
    assert "<Gather" in r.text
    assert "documents" in r.text  # part of the greeting


def test_process_with_empty_speech_reprompts_then_hangs_up():
    r = client.post("/voice/process", data={"CallSid": "CA1", "SpeechResult": ""})
    assert r.status_code == 200
    assert "didn't catch that" in r.text
    assert "<Hangup" in r.text


def test_process_with_speech_calls_handle_turn_and_speaks_answer():
    with patch("app.main.handle_turn", new=AsyncMock(return_value="The answer is 42.")) as mock_handle:
        r = client.post("/voice/process", data={"CallSid": "CA2", "SpeechResult": "What is the answer?"})

    assert r.status_code == 200
    assert "The answer is 42." in r.text
    assert "<Gather" in r.text  # keeps listening for a follow-up
    mock_handle.assert_called_once_with("CA2", "What is the answer?")


def test_process_handles_handle_turn_failure_without_crashing():
    with patch("app.main.handle_turn", new=AsyncMock(side_effect=RuntimeError("boom"))):
        r = client.post("/voice/process", data={"CallSid": "CA3", "SpeechResult": "anything"})

    assert r.status_code == 200  # degrades to a spoken apology, not a 500
    assert "went wrong" in r.text


def test_status_completed_ends_call():
    with patch("app.main.end_call") as mock_end:
        r = client.post("/voice/status", data={"CallSid": "CA4", "CallStatus": "completed"})

    assert r.status_code == 204
    mock_end.assert_called_once_with("CA4")


def test_status_in_progress_does_not_end_call():
    with patch("app.main.end_call") as mock_end:
        r = client.post("/voice/status", data={"CallSid": "CA5", "CallStatus": "in-progress"})

    assert r.status_code == 204
    mock_end.assert_not_called()
