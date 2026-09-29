from __future__ import annotations

import numpy as np

from asr.streaming import VadSegmenter
from diarization.simple_diarizer import (
    IncrementalSpeakerClusterer,
    cluster_offline,
    extract_features,
)


def test_extract_features_is_fixed_length_regardless_of_segment_duration():
    short = np.random.default_rng(0).normal(size=4_000).astype(np.float32)  # 0.25s
    long = np.random.default_rng(0).normal(size=32_000).astype(np.float32)  # 2s
    f_short = extract_features(short, sample_rate=16_000)
    f_long = extract_features(long, sample_rate=16_000)
    assert f_short.shape == f_long.shape == (13,)


def test_cluster_offline_separates_two_clearly_distinct_voices(two_speaker_pcm):
    """Real end-to-end check: VAD-segment the two_speaker fixture, extract
    features from each real segment, and confirm offline clustering
    correctly groups the two A-segments together and the two B-segments
    together (the exact same pairwise-distance behavior observed manually
    while calibrating the default distance_threshold -- see the README and
    the module docstring in diarization/simple_diarizer.py).
    """
    seg = VadSegmenter()
    segments = list(seg.push(two_speaker_pcm)) + list(seg.flush())
    assert len(segments) == 4

    features = [extract_features(s.to_float32(), sample_rate=16_000) for s in segments]
    labels = cluster_offline(features)  # default distance_threshold=5.0

    assert labels[0] == labels[2]  # both speaker-A segments
    assert labels[1] == labels[3]  # both speaker-B segments
    assert labels[0] != labels[1]  # the two voices are distinct clusters


def test_incremental_clusterer_matches_offline_clustering(two_speaker_pcm):
    seg = VadSegmenter()
    segments = list(seg.push(two_speaker_pcm)) + list(seg.flush())
    features = [extract_features(s.to_float32(), sample_rate=16_000) for s in segments]

    clusterer = IncrementalSpeakerClusterer()
    labels = [clusterer.assign(f) for f in features]

    assert labels[0] == labels[2]
    assert labels[1] == labels[3]
    assert labels[0] != labels[1]
    assert clusterer.num_speakers == 2


def test_incremental_clusterer_caps_at_max_speakers():
    clusterer = IncrementalSpeakerClusterer(new_speaker_distance=0.01, max_speakers=3)
    # Each vector is far from the others relative to the tiny threshold, so
    # without the cap this would create a new speaker every call.
    rng = np.random.default_rng(1)
    for _ in range(10):
        clusterer.assign(rng.normal(scale=100, size=13))
    assert clusterer.num_speakers <= 3


def test_cluster_offline_handles_degenerate_inputs():
    assert cluster_offline([]) == []
    assert cluster_offline([np.zeros(13)]) == [0]
