"""Wires VAD segmentation + ASR + diarization into one per-connection
pipeline. Kept separate from ws_server.py so it can be unit-tested without
spinning up FastAPI/WebSockets at all -- feed it bytes, get back events.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Protocol

from api.events import TranscriptEvent
from asr.streaming import SpeechSegment, VadSegmenter
from diarization.simple_diarizer import IncrementalSpeakerClusterer, extract_features


class Transcriber(Protocol):
    """Structural type so tests can pass a fake/mocked transcriber without
    importing (or downloading) faster-whisper at all."""

    def transcribe(self, audio) -> str: ...  # noqa: ANN001, D102


class ConnectionPipeline:
    """One instance per live connection (WebSocket, or a batch file replay).
    Not shared/reused across connections -- it holds per-speaker clustering
    state (`IncrementalSpeakerClusterer`) that's only meaningful within a
    single conversation.
    """

    def __init__(
        self,
        transcriber: Transcriber,
        *,
        sample_rate: int = 16_000,
        segmenter: VadSegmenter | None = None,
        speaker_clusterer: IncrementalSpeakerClusterer | None = None,
    ) -> None:
        self._transcriber = transcriber
        self._sample_rate = sample_rate
        self._segmenter = segmenter or VadSegmenter()
        self._speakers = speaker_clusterer or IncrementalSpeakerClusterer()

    def push_audio(self, chunk: bytes) -> Iterator[TranscriptEvent]:
        for segment in self._segmenter.push(chunk):
            event = self._process_segment(segment)
            if event is not None:
                yield event

    def flush(self) -> Iterator[TranscriptEvent]:
        for segment in self._segmenter.flush():
            event = self._process_segment(segment)
            if event is not None:
                yield event

    def _process_segment(self, segment: SpeechSegment) -> TranscriptEvent | None:
        audio = segment.to_float32()
        text = self._transcriber.transcribe(audio)
        if not text:
            # An empty transcript (silence that slipped past VAD, a non-verbal
            # sound) isn't useful downstream -- drop it rather than emit noise.
            return None

        embedding = extract_features(audio, self._sample_rate)
        speaker_id = self._speakers.assign(embedding)

        return TranscriptEvent(
            speaker=f"speaker_{speaker_id}",
            text=text,
            start_ts=round(segment.start_ts, 3),
            end_ts=round(segment.end_ts, 3),
            is_final=True,
        )


def run_batch(
    audio_chunks: Iterable[bytes],
    transcriber: Transcriber,
    *,
    sample_rate: int = 16_000,
) -> list[TranscriptEvent]:
    """Convenience entry point for offline/test use: feed a full list of
    chunks and get back the complete list of events, including whatever
    flush() finalizes at the end. Used by scripts/stream_wav_file.py's
    non-networked --offline mode and by the test suite.
    """
    pipeline = ConnectionPipeline(transcriber, sample_rate=sample_rate)
    events: list[TranscriptEvent] = []
    for chunk in audio_chunks:
        events.extend(pipeline.push_audio(chunk))
    events.extend(pipeline.flush())
    return events
