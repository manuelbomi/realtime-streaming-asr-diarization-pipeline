"""Integration test for the actual FastAPI WebSocket endpoint -- exercises
routing, binary-frame receipt, and JSON-text-frame responses through
Starlette's TestClient, with the real ASR model swapped for FakeTranscriber
so this runs in milliseconds with no model download.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

import api.ws_server as ws_server
from tests.conftest import FakeTranscriber


def test_healthz():
    client = TestClient(ws_server.app)
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_websocket_stream_returns_transcript_events(two_speaker_pcm, monkeypatch):
    fake = FakeTranscriber("hello")
    monkeypatch.setattr(ws_server, "get_transcriber", lambda: fake)

    client = TestClient(ws_server.app)
    with client.websocket_connect("/ws/stream") as ws:
        # Send in a few chunks, like a real client would.
        chunk = 4000
        for i in range(0, len(two_speaker_pcm), chunk):
            ws.send_bytes(two_speaker_pcm[i : i + chunk])
        ws.send_bytes(ws_server.EOS_MAGIC)

        events = []
        while True:
            try:
                msg = ws.receive_text()
            except Exception:
                break
            events.append(json.loads(msg))
            if len(events) == 4:
                break

    assert len(events) == 4
    for e in events:
        assert set(e.keys()) == {"speaker", "text", "start_ts", "end_ts", "is_final"}
        assert e["text"] == "hello"
