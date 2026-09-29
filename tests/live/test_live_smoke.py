"""REAL faster-whisper model, downloaded and run for real. Excluded from the
default `pytest` run (see pyproject.toml's `addopts = -m "not live"`) because
it downloads ~75MB of model weights on first run and takes several seconds
even on CPU -- too slow/flaky for every CI push, but valuable to run:

    pytest -m live tests/live/

before a release, or whenever asr/streaming.py's WhisperTranscriber wrapper
changes, to confirm the real model integration still works end-to-end
(not just the mocked contract tests).

What this intentionally does NOT assert: exact transcript text. The fixture
audio is synthetic tone bursts (see scripts/synthesize_test_audio.py), not
real speech, so Whisper's output on it is undefined by design -- usually
empty, sometimes a short hallucinated token. This test only asserts that the
real model loads, runs without raising, and returns a `str` (this pipeline's
whole contract with the ASR layer), which is exactly what a code change that
broke the faster-whisper integration would fail.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.live


def test_real_whisper_model_loads_and_transcribes_without_error(two_speaker_pcm):
    from api.pipeline import run_batch
    from asr.streaming import WhisperTranscriber

    transcriber = WhisperTranscriber("tiny.en", device="cpu", compute_type="int8")
    events = run_batch([two_speaker_pcm], transcriber)

    # However many segments produced non-empty text, every one must respect
    # the contract type -- this is the real assertion.
    for event in events:
        assert isinstance(event.text, str)
        assert event.text != ""  # run_batch already drops empty ones
