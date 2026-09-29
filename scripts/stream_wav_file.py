"""Stream a WAV file to the WebSocket server at real-time pace (or run the
pipeline locally with --offline, no server/network needed), and print each
TranscriptEvent as it arrives.

This only accepts 16kHz mono 16-bit PCM WAV input, matching what the rest of
this pipeline assumes throughout (see asr/streaming.py). If you have a
different format (stereo, 44.1kHz, mp3, a call recording exported some other
way), convert it first, e.g.:

    ffmpeg -i input.mp3 -ar 16000 -ac 1 -sample_fmt s16 output.wav

Usage:
    # against a running `uvicorn api.ws_server:app` server
    python -m scripts.stream_wav_file --wav tests/fixtures/two_speaker_conversation.wav

    # locally, no server/network, using the same pipeline code the server uses
    python -m scripts.stream_wav_file --wav tests/fixtures/two_speaker_conversation.wav --offline
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import wave
from pathlib import Path

EOS_MAGIC = b"EOS\0"
EXPECTED_SAMPLE_RATE = 16_000


def _load_pcm16_mono_16k(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wf:
        if wf.getframerate() != EXPECTED_SAMPLE_RATE or wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            raise ValueError(
                f"{path} is {wf.getframerate()}Hz, {wf.getnchannels()}ch, "
                f"{wf.getsampwidth() * 8}-bit -- this tool requires 16kHz mono 16-bit PCM. "
                "Convert with: ffmpeg -i input.wav -ar 16000 -ac 1 -sample_fmt s16 output.wav"
            )
        return wf.readframes(wf.getnframes())


def _chunk(pcm: bytes, chunk_ms: int) -> list[bytes]:
    bytes_per_chunk = int(EXPECTED_SAMPLE_RATE * (chunk_ms / 1000.0)) * 2  # 2 bytes/sample
    return [pcm[i : i + bytes_per_chunk] for i in range(0, len(pcm), bytes_per_chunk)]


async def stream_to_server(host: str, port: int, wav_path: Path, chunk_ms: int) -> None:
    import websockets

    pcm = _load_pcm16_mono_16k(wav_path)
    chunks = _chunk(pcm, chunk_ms)
    uri = f"ws://{host}:{port}/ws/stream"
    print(f"connecting to {uri} ...")

    async with websockets.connect(uri) as ws:
        async def sender():
            for c in chunks:
                await ws.send(c)
                await asyncio.sleep(chunk_ms / 1000.0)  # real-time pace
            await ws.send(EOS_MAGIC)

        async def receiver():
            async for message in ws:
                event = json.loads(message)
                span = f"{event['start_ts']:6.2f}-{event['end_ts']:6.2f}"
                print(f"[{span}] {event['speaker']}: {event['text']}")

        send_task = asyncio.create_task(sender())
        try:
            await receiver()
        finally:
            send_task.cancel()


def run_offline(wav_path: Path) -> None:
    """No network, no server process -- runs the exact same ConnectionPipeline
    the WebSocket server uses, directly in this process. Useful for a quick
    demo or for debugging the pipeline without the networking layer involved.
    """
    from api.pipeline import run_batch
    from asr.streaming import WhisperTranscriber

    pcm = _load_pcm16_mono_16k(wav_path)
    chunks = _chunk(pcm, chunk_ms=200)

    print("loading faster-whisper model (tiny.en, cpu, int8) ...")
    t0 = time.monotonic()
    transcriber = WhisperTranscriber("tiny.en", device="cpu", compute_type="int8")
    print(f"model loaded in {time.monotonic() - t0:.1f}s")

    t0 = time.monotonic()
    events = run_batch(chunks, transcriber)
    elapsed = time.monotonic() - t0
    audio_duration = len(pcm) / 2 / EXPECTED_SAMPLE_RATE
    rtf = elapsed / audio_duration
    print(f"processed {audio_duration:.1f}s of audio in {elapsed:.1f}s (realtime factor {rtf:.2f}x)")

    for event in events:
        print(f"[{event.start_ts:6.2f}-{event.end_ts:6.2f}] {event.speaker}: {event.text}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--chunk-ms", type=int, default=100, help="audio chunk size sent per WS frame, real-time paced"
    )
    parser.add_argument("--offline", action="store_true", help="run the pipeline locally, no server/network")
    args = parser.parse_args()

    if args.offline:
        run_offline(args.wav)
    else:
        asyncio.run(stream_to_server(args.host, args.port, args.wav, args.chunk_ms))


if __name__ == "__main__":
    main()
