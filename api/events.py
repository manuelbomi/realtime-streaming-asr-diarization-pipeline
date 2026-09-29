"""The wire format this whole repo exists to produce.

Every finalized speech segment becomes exactly one JSON object with this
shape, sent as a WebSocket text frame:

    {"speaker": "speaker_0", "text": "...", "start_ts": 1.23, "end_ts": 3.41, "is_final": true}

`start_ts`/`end_ts` are seconds relative to the start of the WebSocket
connection (not wall-clock time), so a client can align them against its own
playback position without needing a shared clock with the server.

`is_final` is always `true` in this implementation -- see the module
docstring in asr/streaming.py for why (Whisper is a batch model; this
pipeline only ever emits a segment once VAD has decided it's finished). The
field exists anyway, and defaults to `true`, because it's part of the
contract a downstream real-time signal-detection consumer should code
against from day one: a future upgrade to a genuinely incremental ASR engine
could start emitting `is_final: false` partial hypotheses on the *same*
connection without breaking any consumer that already checks the flag.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class TranscriptEvent:
    speaker: str
    text: str
    start_ts: float
    end_ts: float
    is_final: bool = True

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
