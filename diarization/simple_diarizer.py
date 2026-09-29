"""Tutorial-grade "who spoke this segment" diarization.

Read this docstring before you reach for this module in anything that
matters. Production diarization (pyannote.audio, NeMo's speaker-diarization
pipelines, etc.) uses a neural speaker-embedding model (trained on tens of
thousands of speakers with a metric-learning loss) plus a real clustering /
resegmentation algorithm (VBx, spectral clustering with eigengap heuristics)
that can also handle *overlapping* speech -- two people talking at once,
common in real calls.

What's here is deliberately simpler, on purpose, for a tutorial:

  1. A hand-rolled, dependency-light "spectral fingerprint" feature
     (log-mel-filterbank energies -> DCT, i.e. an MFCC-style vector,
     average-pooled over the segment) computed with nothing but numpy/scipy.
     No pretrained embedding model, no model download, runs anywhere.
  2. Segment-level assignment only -- it decides "who spoke this whole
     segment," not "who spoke each 20ms frame." Two people talking over each
     other inside one VAD-bounded segment will get attributed to whichever
     voice dominates the segment's average spectrum. That's a real, named
     limitation, not an oversight.

This trades accuracy for zero external dependencies and instant startup,
which is the right trade for learning the *architecture* of a diarization
stage (feature extraction -> clustering -> stable speaker IDs) before you
pay the cost (model download, GPU, tuning) of a production-grade model.

**The production swap point** is exactly one function: replace
`extract_features()` with a call to a pretrained speaker-embedding model
(e.g. `pyannote.audio.Pipeline` or a standalone embedding model like
pyannote's `speechbrain/spkrec-ecapa-voxceleb`). Everything downstream --
`IncrementalSpeakerClusterer`, `cluster_offline` -- operates on a plain
numpy vector and doesn't know or care how it was produced, precisely so that
swap is a one-function change. See docs/adr/0003-tutorial-diarizer.md.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import dct
from sklearn.cluster import AgglomerativeClustering

_EPS = 1e-10


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(n_filters: int, n_fft: int, sample_rate: int) -> np.ndarray:
    """Triangular mel filterbank, shape (n_filters, n_fft // 2 + 1).

    This is the same construction librosa/python_speech_features use
    internally, written out by hand so this module has zero dependency on
    an audio-feature library -- just numpy/scipy, which the rest of the repo
    already needs.
    """
    low_mel, high_mel = _hz_to_mel(np.array([0.0, sample_rate / 2.0]))
    mel_points = np.linspace(low_mel, high_mel, n_filters + 2)
    hz_points = _mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    fbank = np.zeros((n_filters, n_fft // 2 + 1))
    for m in range(1, n_filters + 1):
        f_left, f_center, f_right = bin_points[m - 1], bin_points[m], bin_points[m + 1]
        f_center = max(f_center, f_left + 1)
        f_right = max(f_right, f_center + 1)
        for k in range(f_left, f_center):
            if 0 <= k < fbank.shape[1]:
                fbank[m - 1, k] = (k - f_left) / (f_center - f_left)
        for k in range(f_center, f_right):
            if 0 <= k < fbank.shape[1]:
                fbank[m - 1, k] = (f_right - k) / (f_right - f_center)
    return fbank


def extract_features(
    audio: np.ndarray,
    sample_rate: int,
    *,
    n_filters: int = 26,
    n_mfcc: int = 13,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
) -> np.ndarray:
    """A segment-level MFCC-style spectral fingerprint, average-pooled over
    every analysis frame in the segment. Returns a fixed-length vector of
    shape (n_mfcc,) regardless of segment duration, which is what lets us
    compare segments of different lengths with a plain distance metric.
    """
    if audio.ndim != 1:
        raise ValueError("expected mono audio")

    frame_len = max(1, int(sample_rate * frame_ms / 1000.0))
    hop_len = max(1, int(sample_rate * hop_ms / 1000.0))
    n_fft = 1
    while n_fft < frame_len:
        n_fft *= 2

    if len(audio) < frame_len:
        # Pad very short segments so we can still extract at least one frame.
        audio = np.pad(audio, (0, frame_len - len(audio)))

    window = np.hanning(frame_len)
    fbank = _mel_filterbank(n_filters, n_fft, sample_rate)

    frame_mfccs = []
    for start in range(0, len(audio) - frame_len + 1, hop_len):
        frame = audio[start : start + frame_len] * window
        spectrum = np.abs(np.fft.rfft(frame, n=n_fft)) ** 2
        mel_energies = fbank @ spectrum
        log_mel = np.log(mel_energies + _EPS)
        mfcc = dct(log_mel, type=2, norm="ortho")[:n_mfcc]
        frame_mfccs.append(mfcc)

    if not frame_mfccs:
        return np.zeros(n_mfcc, dtype=np.float64)

    return np.mean(np.stack(frame_mfccs), axis=0)


def cluster_offline(
    embeddings: list[np.ndarray],
    *,
    distance_threshold: float = 5.0,
) -> list[int]:
    """Batch clustering of a known, finite list of segment embeddings into
    speaker labels (0, 1, 2, ...), with the number of speakers determined
    automatically from `distance_threshold` rather than fixed in advance.

    Use this for offline/batch analysis (e.g. "diarize this whole recorded
    call"). For a live stream where segments arrive one at a time and you
    can't wait for the call to end, use IncrementalSpeakerClusterer instead.

    `distance_threshold` is in the same units as the MFCC feature vectors
    (Euclidean distance between average-log-mel-DCT vectors) -- it has no
    universal "right" value; tune it against a few minutes of labeled
    real speech for your microphone/codec before trusting it in production.
    """
    if len(embeddings) == 0:
        return []
    if len(embeddings) == 1:
        return [0]

    X = np.stack(embeddings)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        linkage="average",
        metric="euclidean",
    )
    labels = clustering.fit_predict(X)
    return [int(label) for label in labels]


class IncrementalSpeakerClusterer:
    """Online, single-pass speaker assignment for a live stream: segments
    arrive one at a time and must be labeled immediately, with no ability to
    revisit earlier decisions once more speakers show up.

    Strategy: nearest-centroid assignment. Each known speaker is represented
    by the running mean of the feature vectors assigned to them so far. A new
    segment is assigned to the closest existing centroid if it's within
    `new_speaker_distance`; otherwise it starts a new speaker. This is a
    simple, greedy approximation of the batch clustering above -- it can't
    "change its mind" the way offline clustering can, which is the real cost
    of running online instead of after the fact.
    """

    def __init__(self, *, new_speaker_distance: float = 5.0, max_speakers: int = 8) -> None:
        self._new_speaker_distance = new_speaker_distance
        self._max_speakers = max_speakers
        self._centroids: list[np.ndarray] = []
        self._counts: list[int] = []

    def assign(self, embedding: np.ndarray) -> int:
        if not self._centroids:
            self._centroids.append(embedding.copy())
            self._counts.append(1)
            return 0

        distances = [float(np.linalg.norm(embedding - c)) for c in self._centroids]
        best_idx = int(np.argmin(distances))

        if distances[best_idx] <= self._new_speaker_distance or len(self._centroids) >= self._max_speakers:
            n = self._counts[best_idx]
            self._centroids[best_idx] = (self._centroids[best_idx] * n + embedding) / (n + 1)
            self._counts[best_idx] += 1
            return best_idx

        self._centroids.append(embedding.copy())
        self._counts.append(1)
        return len(self._centroids) - 1

    @property
    def num_speakers(self) -> int:
        return len(self._centroids)
