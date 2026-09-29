"""FastAPI WebSocket server: send it raw 16kHz mono int16 PCM audio, get back
one JSON TranscriptEvent (see api/events.py) per finalized speech segment.

Run it:
    uvicorn api.ws_server:app --host 0.0.0.0 --port 8000

Talk to it:
    ws://localhost:8000/ws/stream
    -> client sends binary frames of raw PCM16 audio, any chunk size
    -> server sends back text frames, each one JSON-encoded TranscriptEvent
    -> client closes the socket (or sends the 4-byte magic b"EOS\\0") to
       signal end-of-stream, which flushes any in-progress segment

See scripts/stream_wav_file.py for a working client, and docs/PRODUCTION.md
for how this single-process, single-model-instance design needs to change to
serve many concurrent calls (worker pool, GPU batching, backpressure).
"""

from __future__ import annotations

import json
import logging
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from api.pipeline import ConnectionPipeline
from asr.streaming import WhisperTranscriber

logger = logging.getLogger("ws_server")

EOS_MAGIC = b"EOS\0"

# Loaded once per process, on first use, and shared read-only across all
# connections -- loading a Whisper model per connection would make every
# call pay multi-second model-load latency. See docs/PRODUCTION.md for why
# this single shared instance is a concurrency bottleneck under real
# multi-call load, and how a worker-pool / batched-inference design fixes it.
_transcriber: WhisperTranscriber | None = None


def get_transcriber() -> WhisperTranscriber:
    global _transcriber
    if _transcriber is None:
        model_size = os.environ.get("ASR_MODEL_SIZE", "tiny.en")
        device = os.environ.get("ASR_DEVICE", "cpu")
        compute_type = os.environ.get("ASR_COMPUTE_TYPE", "int8")
        logger.info(
            "loading faster-whisper model=%s device=%s compute_type=%s", model_size, device, compute_type
        )
        _transcriber = WhisperTranscriber(model_size, device=device, compute_type=compute_type)
    return _transcriber


app = FastAPI(title="realtime-streaming-asr-diarization-pipeline")


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.websocket("/ws/stream")
async def stream(ws: WebSocket) -> None:
    await ws.accept()
    pipeline = ConnectionPipeline(get_transcriber())

    try:
        while True:
            chunk = await ws.receive_bytes()
            if chunk == EOS_MAGIC:
                break
            for event in pipeline.push_audio(chunk):
                await ws.send_text(json.dumps(event.to_dict()))
    except WebSocketDisconnect:
        return
    finally:
        for event in pipeline.flush():
            try:
                await ws.send_text(json.dumps(event.to_dict()))
            except RuntimeError:
                # socket already closed on the client side -- nothing to send to
                break
