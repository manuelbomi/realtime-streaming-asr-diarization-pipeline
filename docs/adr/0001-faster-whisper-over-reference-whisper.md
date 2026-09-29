# ADR 0001: faster-whisper over the openai/whisper reference implementation

## Status

Accepted

## Context

OpenAI's reference `whisper` package is the canonical implementation and the
easiest to find tutorials for, but it runs the model in plain PyTorch. This
pipeline is built around the VAD-gated segment-level design in
`asr/streaming.py`: every finalized speech segment triggers one synchronous
transcription call, and the whole point of the design is to get that
transcript back with as little added latency as possible.

## Decision

Use `faster-whisper`, which re-implements Whisper's inference on top of
CTranslate2 (a C++ inference engine originally built for translation models)
instead of plain PyTorch, and use `compute_type="int8"` for CPU inference.

## Consequences

- **Faster CPU inference.** CTranslate2's int8 quantized kernels are
  substantially faster than PyTorch's default fp32 path on CPU -- this
  matters here specifically because the design goal is "transcript arrives
  shortly after the speaker stops talking," not "transcribe a batch of audio
  files overnight." The `tiny.en`/int8 combination measured in this repo's
  README loaded in ~5s and processed the test fixture at roughly 4x
  real-time on CPU in this sandbox -- see the README for the exact numbers
  and how they were measured.
- **Same model weights, same accuracy characteristics.** faster-whisper loads
  the same published Whisper checkpoints (`tiny`, `tiny.en`, `base`, `base.en`,
  `small`, ... `large-v3`) -- this is a faster *engine*, not a different or
  degraded model. Swapping model sizes is a one-argument change
  (`WhisperTranscriber(model_size=...)`), not a re-architecture.
- **An extra native dependency.** CTranslate2 ships prebuilt wheels for the
  common platforms (which is what made it viable to install and run in this
  Windows/Python 3.10 sandbox with no extra build tooling), but it's still a
  compiled dependency, which is one more thing to get right in a Docker base
  image than pure-Python PyTorch.
- **GPU path is available but not exercised here.** faster-whisper supports
  `device="cuda"` with float16 compute for GPU serving; this repo's default
  (`device="cpu"`, `compute_type="int8"`) targets a laptop/CI-friendly setup.
  See docs/PRODUCTION.md for what changes to actually serve concurrent calls
  at scale.

## Alternatives considered

- **openai/whisper (reference PyTorch implementation).** Simpler dependency
  footprint, but meaningfully slower on CPU, which directly fights the
  latency goal this whole pipeline is designed around.
- **whisper.cpp.** Even faster CPU-only inference via a hand-written C/C++
  kernel, but a separate binary/FFI boundary rather than a `pip install`-able
  Python package, which raises the bar for a tutorial repo meant to be
  readable end-to-end in Python. Worth revisiting for an embedded/edge
  deployment target.
