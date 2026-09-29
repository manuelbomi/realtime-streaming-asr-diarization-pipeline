# ADR 0002: VAD-gated chunking over fixed-size chunking

## Status

Accepted

## Context

Any streaming ASR pipeline has to decide, somehow, how much audio to hand to
the transcription model at once and when. There are two obvious strategies:

1. **Fixed-size chunking** -- transcribe every N seconds of audio,
   regardless of whether anyone is talking.
2. **VAD-gated chunking** (what this repo does) -- use a voice-activity
   detector to find utterance boundaries, and transcribe each complete
   utterance once it ends.

## Decision

Use `webrtcvad` to detect speech/silence per 30ms frame
(`asr/streaming.py::VadSegmenter`), buffer while someone is talking, and
finalize + transcribe the buffered segment after a configurable run of
trailing silence (`silence_timeout_ms`, default 500ms).

## Consequences

- **Transcripts align to actual utterance boundaries**, not arbitrary time
  windows. Fixed-size chunking will regularly cut a sentence in half at a
  chunk boundary, handing Whisper two incomplete fragments instead of one
  coherent segment -- and Whisper, like most ASR models, does measurably
  worse on truncated/fragmented audio than on a complete utterance.
- **No wasted transcription calls on silence.** A fixed-size chunker
  transcribes every window whether or not anyone spoke in it; on a real
  call, a large fraction of wall-clock time is silence (pauses, listening,
  hold time), so this can easily 2-3x the number of (wasted) inference calls
  for no benefit.
- **Latency is bounded by pause length, not a fixed clock.** A short
  utterance ("yes", "no", "hold on") gets transcribed almost immediately
  after `silence_timeout_ms` elapses; a long uninterrupted monologue is
  force-finalized at `max_segment_ms` (default 20s) so latency never grows
  unbounded and no single Whisper call has to eat an arbitrary amount of
  audio.
- **The real cost: it's still segment-level, not token-level, streaming.**
  This design fundamentally cannot show a word appearing on screen while
  someone is mid-sentence -- see the "Limitations & extension points"
  section of the README. That's a deliberate scope boundary of a
  Whisper-based pipeline (a batch model transcribing VAD-bounded chunks),
  not a defect in the chunking strategy itself.
- **One more tunable knob than fixed-size chunking**, and it's a real
  trade-off, not a free lunch: `silence_timeout_ms` too short clips
  sentences on natural pauses (a comma-length pause can trigger a false
  finalize); too long adds dead air to every segment's latency. 500ms is a
  reasonable default for conversational speech, tuned by ear against this
  repo's synthetic fixture and worth re-tuning against real call audio for
  a specific deployment (fast interrupty conversation vs. slower deliberate
  speech behave differently).

## Alternatives considered

- **Fixed-size chunking (e.g. every 5s).** Simpler, no VAD dependency, but
  produces materially worse transcripts on chunk-boundary-cut sentences and
  wastes inference calls on silence -- rejected for a pipeline whose whole
  purpose is feeding clean, complete-utterance segments downstream.
- **A genuinely incremental/streaming ASR model** (e.g. a
  chunk-wise-attention streaming Conformer, whisper_streaming's re-decode
  approach). Would enable true token-level partial hypotheses
  (`is_final: false` events -- see `api/events.py`), at the cost of a much
  more complex model-serving story than "call faster-whisper once per
  finalized segment." Documented as a real future extension point, not
  implemented here to keep the tutorial's core mechanism (VAD -> segment ->
  transcribe) legible.
