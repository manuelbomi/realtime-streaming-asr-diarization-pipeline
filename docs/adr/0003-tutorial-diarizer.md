# ADR 0003: a lightweight hand-rolled diarizer as the tutorial default

## Status

Accepted

## Context

Speaker diarization ("who spoke when") normally means a pretrained neural
speaker-embedding model (e.g. `pyannote.audio`'s pipelines, or a standalone
embedding model like ECAPA-TDNN) plus a real clustering/resegmentation
algorithm capable of handling overlapping speech. `pyannote.audio`'s
pipelines are gated on HuggingFace (require accepting a license + an access
token) and pull down multi-hundred-MB model weights on first use.

This repo's goal is that someone can clone it and have the whole pipeline
running, readable, and testable in minutes, with no account/token/large
download required.

## Decision

Ship a from-scratch, dependency-light diarizer
(`diarization/simple_diarizer.py`): a hand-rolled MFCC-style spectral
feature (log-mel-filterbank energies -> DCT, computed with nothing but
numpy/scipy) averaged over each VAD-finalized segment, clustered either in
batch (`cluster_offline`, `sklearn.cluster.AgglomerativeClustering`) or
incrementally for a live stream (`IncrementalSpeakerClusterer`, nearest-
centroid assignment).

## Consequences

- **Zero extra downloads, runs anywhere numpy/scipy/scikit-learn run.**
  Verified in this repo's own test suite (`tests/test_diarizer.py`) against
  the synthetic two-speaker fixture: the two "speaker A" segments and the
  two "speaker B" segments cluster correctly at the shipped default
  (`distance_threshold=5.0` / `new_speaker_distance=5.0`) -- see the README
  for the real measured pairwise distances used to calibrate that default.
- **Segment-level only -- cannot resolve overlapping speech.** If two people
  talk over each other inside a single VAD-bounded segment, this diarizer
  attributes the *whole segment* to whichever voice dominates the average
  spectrum. A real call-center or meeting recording has overlapping speech
  regularly (interruptions, backchannels like "mm-hm"); this is a named,
  understood limitation, not an oversight.
- **A hand-tuned distance threshold, not a calibrated model.** The default
  values were chosen by inspecting actual pairwise distances on one
  synthetic fixture (documented in the README) -- they are a reasonable
  starting point, not a value guaranteed to generalize to different
  microphones, codecs, or telephony compression. Production use requires
  re-tuning against real labeled audio for the target deployment.
- **The swap point is exactly one function.** Everything downstream of
  `extract_features()` -- both clusterers -- operates on a plain fixed-length
  numpy vector and has no idea whether it came from a hand-rolled MFCC or a
  256-dim neural speaker embedding. Upgrading to `pyannote.audio` in
  production means replacing the body of `extract_features()` (or adding a
  parallel `extract_features_neural()` and pointing the pipeline at it) --
  no change needed to `IncrementalSpeakerClusterer`, `cluster_offline`, or
  anything in `api/pipeline.py`.

## Alternatives considered

- **`pyannote.audio` out of the box, as the only option.** Better accuracy
  and native overlapping-speech handling, but a HuggingFace-gated model
  download and heavier dependency footprint (torch + torchaudio + the
  pipeline package) work against a tutorial repo's "clone and run in
  minutes" goal. Documented here as *the* recommended production upgrade
  path, not rejected outright -- see docs/PRODUCTION.md.
- **No diarization at all (single-speaker transcript only).** Simpler, but
  removes an entire architectural stage (feature extraction -> clustering ->
  stable speaker IDs) that's exactly what this repo exists to teach, and
  breaks the `speaker` field in the downstream event contract
  (`api/events.py`) that a consumer would reasonably expect to be populated.
