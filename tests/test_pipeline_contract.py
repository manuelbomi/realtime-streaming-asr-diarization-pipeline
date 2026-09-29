"""Tests the event contract the whole repo exists to deliver: for every
finalized speech segment, exactly one JSON-serializable object with keys
{speaker, text, start_ts, end_ts, is_final}, nothing more, nothing less.
A downstream consumer (e.g. a real-time signal-detection service) codes
against this shape -- these tests exist to make an accidental field
rename/removal fail CI instead of a downstream integration silently.
"""

from __future__ import annotations

import json

from api.pipeline import ConnectionPipeline, run_batch
from tests.conftest import FakeTranscriber

EXPECTED_KEYS = {"speaker", "text", "start_ts", "end_ts", "is_final"}


def test_event_shape_matches_contract(two_speaker_pcm):
    events = run_batch([two_speaker_pcm], FakeTranscriber("hi there"))
    assert len(events) == 4
    for event in events:
        d = event.to_dict()
        assert set(d.keys()) == EXPECTED_KEYS
        assert isinstance(d["speaker"], str) and d["speaker"].startswith("speaker_")
        assert isinstance(d["text"], str) and d["text"]
        assert isinstance(d["start_ts"], float)
        assert isinstance(d["end_ts"], float)
        assert d["is_final"] is True
        json.dumps(d)  # must be trivially JSON-serializable


def test_empty_transcript_segments_are_dropped(two_speaker_pcm):
    events = run_batch([two_speaker_pcm], FakeTranscriber(""))
    assert events == []


def test_same_synthetic_voice_gets_same_speaker_label(two_speaker_pcm):
    events = run_batch([two_speaker_pcm], FakeTranscriber("hi"))
    speakers = [e.speaker for e in events]
    assert speakers[0] == speakers[2]  # both speaker-A segments
    assert speakers[1] == speakers[3]  # both speaker-B segments
    assert speakers[0] != speakers[1]


def test_transcriber_only_called_once_per_finalized_segment(two_speaker_pcm):
    fake = FakeTranscriber("hi")
    run_batch([two_speaker_pcm], fake)
    assert len(fake.calls) == 4  # exactly the 4 VAD-finalized segments


def test_arbitrary_chunk_boundaries_dont_change_the_result(two_speaker_pcm):
    """The caller can push audio in whatever chunk sizes it received it in
    (a WebSocket frame, a microphone callback buffer) -- the emitted events
    must not depend on where those chunk boundaries happen to fall.
    """
    fake_a, fake_b = FakeTranscriber("hi"), FakeTranscriber("hi")

    pipeline_a = ConnectionPipeline(fake_a)
    events_a = list(pipeline_a.push_audio(two_speaker_pcm)) + list(pipeline_a.flush())

    pipeline_b = ConnectionPipeline(fake_b)
    events_b = []
    chunk = 777  # deliberately not frame-aligned
    for i in range(0, len(two_speaker_pcm), chunk):
        events_b.extend(pipeline_b.push_audio(two_speaker_pcm[i : i + chunk]))
    events_b.extend(pipeline_b.flush())

    assert [e.speaker for e in events_a] == [e.speaker for e in events_b]
    assert [round(e.start_ts, 1) for e in events_a] == [round(e.start_ts, 1) for e in events_b]
