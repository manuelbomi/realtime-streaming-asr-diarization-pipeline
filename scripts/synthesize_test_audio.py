"""Generate small synthetic WAV fixtures for tests and demos.

Important honesty note, right up front: this does NOT generate real speech.
It generates amplitude-modulated, harmonically-structured tone bursts loud
enough in the speech-relevant frequency range to reliably trigger a
real-speech-tuned VAD (webrtcvad), separated by silence, with two distinct
spectral "voices" so the diarizer's clustering has something structurally
different to tell apart. It is a stand-in for exercising the *pipeline
plumbing* (chunking, segment boundaries, event contract, clustering
mechanics) in CI without shipping copyrighted or third-party speech audio in
this repo.

For an actual demo of transcription quality or diarization accuracy, supply
your own real speech WAV file (16kHz mono recommended; scripts/stream_wav_file.py
will resample/downmix anything soundfile can read) -- synthetic tones will
transcribe to empty/garbage text from Whisper by design, since they aren't
speech.
"""

from __future__ import annotations

import argparse
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE_HZ = 16_000


def _voice_burst(duration_s: float, fundamental_hz: float, sample_rate: int, seed: int) -> np.ndarray:
    """A harmonically-structured, amplitude-modulated tone burst standing in
    for one "voice." Different `fundamental_hz` values give the diarizer's
    spectral feature extractor genuinely different fingerprints to cluster.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration_s * sample_rate)) / sample_rate

    # A few harmonics with decaying amplitude, like a simplified vowel formant
    # structure, plus a slow amplitude envelope so it isn't a flat pure tone
    # (real speech energy rises and falls; a dead-flat tone is easier for a
    # VAD to mis-flag as a machine hum rather than speech).
    signal = np.zeros_like(t)
    for harmonic, amp in enumerate((1.0, 0.6, 0.35, 0.2), start=1):
        signal += amp * np.sin(2 * np.pi * fundamental_hz * harmonic * t)

    envelope_hz = 3.0  # ~syllable-rate amplitude modulation
    envelope = 0.6 + 0.4 * np.sin(2 * np.pi * envelope_hz * t - np.pi / 2)
    signal *= envelope

    signal += rng.normal(scale=0.03, size=signal.shape)  # a little breathiness/noise

    # Short fade in/out so segment edges aren't a hard click (which can itself
    # look like a VAD-triggering transient at the wrong spot).
    fade_len = min(len(signal) // 8, int(0.02 * sample_rate))
    if fade_len > 0:
        fade = np.linspace(0.0, 1.0, fade_len)
        signal[:fade_len] *= fade
        signal[-fade_len:] *= fade[::-1]

    signal /= np.max(np.abs(signal)) + 1e-9
    return signal * 0.9  # leave a little headroom below full scale


def build_two_speaker_conversation(sample_rate: int = SAMPLE_RATE_HZ) -> np.ndarray:
    """speaker A, speaker B, speaker A, speaker B -- with clean silence gaps
    long enough (700ms > the 500ms default silence_timeout) to force four
    separate finalized segments, so tests can assert on segment count/order
    and on the diarizer re-identifying the two A segments as one speaker and
    the two B segments as another.
    """
    silence = np.zeros(int(0.7 * sample_rate))

    def speaker_a(seed: int) -> np.ndarray:
        return _voice_burst(1.2, fundamental_hz=140.0, sample_rate=sample_rate, seed=seed)

    def speaker_b(seed: int) -> np.ndarray:
        return _voice_burst(1.2, fundamental_hz=260.0, sample_rate=sample_rate, seed=seed)

    parts = [
        silence,
        speaker_a(seed=1),
        silence,
        speaker_b(seed=2),
        silence,
        speaker_a(seed=3),
        silence,
        speaker_b(seed=4),
        silence,
    ]
    return np.concatenate(parts).astype(np.float32)


def write_wav(path: Path, audio_f32: np.ndarray, sample_rate: int = SAMPLE_RATE_HZ) -> None:
    int16 = np.clip(audio_f32 * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("tests/fixtures/two_speaker_conversation.wav"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    audio = build_two_speaker_conversation()
    write_wav(args.out, audio)
    print(f"wrote {args.out} ({len(audio) / SAMPLE_RATE_HZ:.2f}s @ {SAMPLE_RATE_HZ}Hz)")


if __name__ == "__main__":
    main()
