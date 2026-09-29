"""Voice-activity-gated segmentation + streaming transcription.

The core idea: Whisper (and faster-whisper) is a *batch* model -- it transcribes
a fixed chunk of audio in one forward pass. It has no native notion of "keep
listening and update your guess as more audio arrives" the way some purpose-built
streaming ASR systems do. So a "streaming" experience with Whisper is really:

    1. Decide, frame by frame, whether someone is currently talking (VAD).
    2. Buffer audio while they're talking.
    3. The instant they stop (a run of trailing silence), treat that buffered
       chunk as "final" and hand the whole thing to Whisper in one call.

This gives you segment-level streaming (a result every time someone finishes a
sentence/thought) rather than token-level streaming (a result updating word by
word while they're still talking). That's a real, load-bearing limitation of
this design -- see the "Limitations & extension points" section in the README
and docs/adr/0002-vad-gated-chunking.md before assuming this behaves like a
live captioning system.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator

import numpy as np
import webrtcvad

# --- Audio format this whole pipeline assumes ------------------------------
# 16 kHz, mono, 16-bit signed little-endian PCM. This is the sweet spot for
# webrtcvad (which only accepts 8/16/32/48 kHz) and for Whisper (which
# resamples everything to 16 kHz internally anyway, so feeding it 16 kHz
# avoids a wasted resample and matches what the model was trained on).
SAMPLE_RATE_HZ = 16_000
SAMPLE_WIDTH_BYTES = 2  # int16

# webrtcvad only accepts 10, 20, or 30 ms frames. 30 ms is the most tolerant
# of the three (fewer false "silence" flags on soft speech onsets/offsets) at
# the cost of slightly coarser timing resolution -- an acceptable trade for a
# segment-level (not token-level) pipeline.
FRAME_DURATION_MS = 30
FRAME_BYTES = int(SAMPLE_RATE_HZ * (FRAME_DURATION_MS / 1000.0)) * SAMPLE_WIDTH_BYTES


@dataclasses.dataclass(frozen=True)
class SpeechSegment:
    """One VAD-finalized chunk of audio, ready to hand to an ASR model."""

    pcm16: bytes  # raw 16 kHz mono int16 PCM, the whole segment
    start_ts: float  # seconds, relative to the start of the stream
    end_ts: float  # seconds, relative to the start of the stream

    def to_float32(self) -> np.ndarray:
        """Convert to the [-1.0, 1.0] float32 mono array faster-whisper expects."""
        int16 = np.frombuffer(self.pcm16, dtype=np.int16)
        return (int16.astype(np.float32) / 32768.0).copy()


class VadSegmenter:
    """Feed it raw PCM16 bytes in arbitrary-sized chunks; get back finalized
    SpeechSegments whenever a run of speech is followed by enough trailing
    silence.

    This is a state machine, not a pure function, because real audio doesn't
    arrive pre-cut into utterances -- it arrives as a continuous byte stream
    (from a microphone, a WebSocket, a telephony media stream) and something
    has to decide *where the boundaries are*. That's this class's whole job.

    Usage:
        seg = VadSegmenter()
        for chunk in audio_source:            # any chunk size, any timing
            for finalized in seg.push(chunk):  # usually 0, occasionally 1+
                handle(finalized)
        for finalized in seg.flush():          # end of stream: emit whatever's left
            handle(finalized)
    """

    def __init__(
        self,
        *,
        aggressiveness: int = 2,
        # How much trailing silence ends a segment. Too short and you cut
        # someone off mid-sentence on a natural pause; too long and every
        # segment drags in a second-plus of dead air before you get a
        # transcript. 500ms is a reasonable default for conversational speech.
        silence_timeout_ms: int = 500,
        # Segments shorter than this are almost always a VAD false-positive
        # (a cough, a mouse click picked up by a bad mic) -- drop them rather
        # than pay for a Whisper call that will return noise or an empty string.
        min_segment_ms: int = 250,
        # Force-finalize very long speech runs (e.g. someone monologuing for
        # 30+ seconds without a natural pause) so a) the caller starts getting
        # results instead of waiting indefinitely, and b) a single Whisper call
        # never has to eat an unbounded amount of audio.
        max_segment_ms: int = 20_000,
    ) -> None:
        if aggressiveness not in (0, 1, 2, 3):
            raise ValueError("webrtcvad aggressiveness must be 0-3")
        self._vad = webrtcvad.Vad(aggressiveness)
        self._silence_timeout_frames = max(1, silence_timeout_ms // FRAME_DURATION_MS)
        self._min_segment_frames = max(1, min_segment_ms // FRAME_DURATION_MS)
        self._max_segment_frames = max(1, max_segment_ms // FRAME_DURATION_MS)

        self._byte_buffer = bytearray()  # unprocessed bytes, not yet frame-aligned
        self._speech_frames: list[bytes] = []  # frames belonging to the in-progress segment
        self._speech_only_frame_count = 0  # excludes trailing-silence padding frames
        self._trailing_silence_frames = 0
        self._in_speech = False
        self._frames_seen_total = 0  # for absolute timestamps
        self._segment_start_frame_idx: int | None = None

    def push(self, chunk: bytes) -> Iterator[SpeechSegment]:
        """Feed in newly-arrived raw PCM16 bytes. Yields zero or more
        finalized SpeechSegments (almost always zero; occasionally one, if
        this chunk happens to contain the trailing edge of an utterance)."""
        self._byte_buffer.extend(chunk)
        while len(self._byte_buffer) >= FRAME_BYTES:
            frame = bytes(self._byte_buffer[:FRAME_BYTES])
            del self._byte_buffer[:FRAME_BYTES]
            yield from self._consume_frame(frame)

    def flush(self) -> Iterator[SpeechSegment]:
        """Call at end-of-stream. Finalizes any in-progress segment, ignoring
        the silence-timeout requirement (there's no more audio coming)."""
        if self._in_speech and self._speech_frames:
            seg = self._finalize_segment()
            if seg is not None:
                yield seg
        # Any leftover partial frame in the byte buffer (< FRAME_BYTES) is
        # discarded -- it's sub-30ms of audio, below any useful resolution.

    def _consume_frame(self, frame: bytes) -> Iterator[SpeechSegment]:
        is_speech = self._vad.is_speech(frame, SAMPLE_RATE_HZ)
        self._frames_seen_total += 1

        if is_speech:
            if not self._in_speech:
                self._in_speech = True
                self._segment_start_frame_idx = self._frames_seen_total - 1
            self._speech_frames.append(frame)
            self._speech_only_frame_count += 1
            self._trailing_silence_frames = 0

            if len(self._speech_frames) >= self._max_segment_frames:
                seg = self._finalize_segment()
                if seg is not None:
                    yield seg
            return

        # Non-speech frame.
        if self._in_speech:
            # Keep a little trailing silence in the buffer so words aren't
            # clipped right at their natural decay -- Whisper does better with
            # a small pad of context on both edges.
            self._speech_frames.append(frame)
            self._trailing_silence_frames += 1
            if self._trailing_silence_frames >= self._silence_timeout_frames:
                seg = self._finalize_segment()
                if seg is not None:
                    yield seg

    def _finalize_segment(self) -> SpeechSegment | None:
        frames = self._speech_frames
        start_idx = self._segment_start_frame_idx
        speech_only_frames = self._speech_only_frame_count
        self._speech_frames = []
        self._speech_only_frame_count = 0
        self._trailing_silence_frames = 0
        self._in_speech = False
        self._segment_start_frame_idx = None

        # Judge "too short" against actual speech duration, not the total
        # buffered duration -- the buffer also includes the trailing-silence
        # pad appended above for Whisper context, which on its own can easily
        # exceed min_segment_frames even for a single-frame VAD false-positive.
        if speech_only_frames < self._min_segment_frames or start_idx is None:
            return None  # too short -- almost certainly a VAD false-positive

        pcm16 = b"".join(frames)
        start_ts = start_idx * (FRAME_DURATION_MS / 1000.0)
        end_ts = (start_idx + len(frames)) * (FRAME_DURATION_MS / 1000.0)
        return SpeechSegment(pcm16=pcm16, start_ts=start_ts, end_ts=end_ts)


class WhisperTranscriber:
    """Thin wrapper around faster-whisper so the rest of the pipeline depends
    on a two-line interface (`transcribe(float32 mono 16kHz) -> str`) instead
    of the full faster-whisper API surface. Swap this class out (e.g. for a
    hosted ASR API, or a larger/GPU model) without touching VadSegmenter or
    the WebSocket server.
    """

    def __init__(
        self,
        model_size: str = "tiny.en",
        *,
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        # Imported lazily so `python -m pytest -k "not live"` and anything
        # that only needs VadSegmenter/diarization doesn't pay the cost of
        # importing ctranslate2 or downloading model weights.
        from faster_whisper import WhisperModel

        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio: np.ndarray) -> str:
        """audio: float32 mono PCM at SAMPLE_RATE_HZ, range [-1.0, 1.0]."""
        segments, _info = self._model.transcribe(
            audio,
            language="en",
            vad_filter=False,  # we already VAD-gated upstream; don't double-gate
            beam_size=1,  # greedy decoding -- lower latency, adequate for short segments
        )
        return " ".join(s.text.strip() for s in segments).strip()
