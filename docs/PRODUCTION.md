# Taking this from a tutorial repo to a production call-audio pipeline

This document is deliberately concrete about what changes, and honest about
what's a measured number versus a design target. Numbers labeled "measured"
were observed running this exact repo in the sandbox it was built in
(Windows, CPU, Python 3.10, `faster-whisper` `tiny.en`/int8); everything else
is labeled "design target" and should be re-measured against your own
hardware and audio before you rely on it.

## 1. Latency budget

The end-to-end latency a user experiences for one utterance is roughly:

```
silence_timeout_ms (wait for trailing silence to confirm they're done)
  + ASR inference time for that segment
  + network/serialization overhead
```

**Measured** in this sandbox: loading `tiny.en` (CPU, int8) took ~5.1s
(a one-time cost at process startup, not per segment); transcribing the
~8.3s synthetic test fixture (4 short segments) took ~2.0s total inference
time, i.e. faster-whisper ran at roughly **0.24x real-time** (processing
time was about a quarter of audio duration) on ordinary CPU hardware with no
GPU. That means the per-segment inference cost for a typical 1-2s utterance
was well under a second in this environment.

Combined with the default `silence_timeout_ms=500`, a realistic **design
target** for one utterance's end-to-end latency (silence-to-transcript) on
CPU with `tiny.en` is roughly 0.5-1.5s for short utterances. This is a
target to validate against your real audio and hardware, not a guarantee --
it will vary with segment length, CPU, and concurrent load (see below).

**Levers, in order of impact:**

1. **`silence_timeout_ms`** -- the single biggest fixed cost for short
   utterances. Lowering it trades latency for a higher risk of clipping
   sentences on natural pauses (see docs/adr/0002).
2. **Model size** -- `tiny.en`/`base.en` for lowest latency;
   `small.en`/`medium.en` for better accuracy at real added latency cost.
   Benchmark this trade-off against your own accuracy requirements and real
   call audio before choosing -- this repo did not benchmark accuracy
   (synthetic tones have no ground-truth transcript to compare against).
3. **GPU inference** (`device="cuda"`, `compute_type="float16"`) --
   faster-whisper supports this directly; not exercised in this repo's
   CPU-only sandbox, but the standard next step for meaningfully lower
   per-segment latency at scale.
4. **Network/serialization** -- this repo's WebSocket JSON-per-event
   protocol is simple and debuggable but not the lowest-overhead option;
   see "Telephony ingestion" below for the binary/protobuf alternative worth
   considering at scale.

## 2. Scaling to many concurrent calls

`api/ws_server.py`'s current design loads **one shared `WhisperModel`
instance per process** (`get_transcriber()`), called synchronously from
every connection's coroutine. CTranslate2 releases the GIL during inference,
so concurrent asyncio connections don't fully serialize behind each other --
but they do contend for the same CPU (or GPU) resource, so throughput is
still bounded by how many transcriptions that one model instance can
actually process per second, not by how many WebSocket connections are open.

For real concurrent-call volume:

- **Horizontal scaling**: run N stateless replicas of this FastAPI service
  behind a load balancer that supports sticky WebSocket sessions (each call
  stays pinned to one replica for its duration, since `ConnectionPipeline`
  holds per-call state). Kubernetes + a `Service` with session affinity, or
  an ALB with sticky sessions, both work.
- **A dedicated ASR worker pool**: instead of transcribing inline in the
  WebSocket handler, push finalized segments onto a queue (Redis Streams,
  SQS, a Kafka topic) and have a separate pool of GPU worker processes pull
  segments, batch them, transcribe, and publish results back. This
  decouples "how many calls are connected" from "how many transcriptions
  are running at once" and enables **batched GPU inference** -- feeding
  several segments from different calls into one forward pass, which is
  dramatically more GPU-efficient than one-segment-at-a-time. faster-whisper
  and vLLM-style batching frameworks both support this pattern; it's the
  standard architecture for serving many concurrent low-latency ASR streams.
- **Autoscaling**: scale the worker pool on queue depth (a strong scaling
  signal, unlike CPU%, since load is bursty by nature -- calls start/stop
  independently) rather than raw request rate.

## 3. Upgrading the diarizer to `pyannote.audio`

`diarization/simple_diarizer.py` is intentionally a tutorial-grade,
dependency-light stand-in (see docs/adr/0003). The exact swap point:

```python
# Current (asr/streaming.py + diarization/simple_diarizer.py):
embedding = extract_features(audio, sample_rate)  # hand-rolled MFCC-style vector
speaker_id = speaker_clusterer.assign(embedding)

# Production swap: replace extract_features()'s body (or add a parallel
# extract_features_neural()) to call a pretrained speaker-embedding model,
# e.g.:
from pyannote.audio import Model, Inference
embedding_model = Model.from_pretrained("pyannote/embedding", use_auth_token=HF_TOKEN)
inference = Inference(embedding_model, window="whole")
embedding = inference({"waveform": audio_tensor, "sample_rate": sample_rate})
# -> feed this embedding into the SAME IncrementalSpeakerClusterer/cluster_offline
```

Nothing downstream of `extract_features()` needs to change -- both
clusterers operate on a plain fixed-length numpy vector regardless of how it
was produced. What you gain: far better speaker discrimination (a model
trained via metric learning on tens of thousands of speakers, vs. a
hand-rolled spectral fingerprint), and -- if you adopt `pyannote.audio`'s
full pipeline rather than just its embedding model -- native handling of
**overlapping speech**, which the segment-level design here cannot do at
all (see docs/adr/0003's "Consequences").

## 4. Telephony ingestion

This repo's WebSocket server assumes something upstream is already handing
it raw 16kHz mono PCM16 chunks. Getting there from a real phone call
requires a telephony media-streaming integration, generically:

- **SIP/RTP ingestion**: a media gateway (e.g. FreeSWITCH, Asterisk, or a
  cloud telephony provider's built-in media-streaming feature) terminates
  the call's RTP audio and forwards it as a stream to your service. Most
  telephony providers offer this as a managed feature -- for example,
  **Twilio Media Streams** forwards a live call's audio to a WebSocket you
  control, as base64-encoded mu-law-encoded audio at 8kHz by default (which
  you'd decode and upsample to 16kHz before feeding this pipeline -- mu-law
  decode is a lookup table, upsampling is a standard `scipy.signal.resample`
  call). Any similar "forward this call's audio to my WebSocket" telephony
  feature slots in the same way.
- **Two audio streams, not one**: real call-center telephony typically gives
  you the agent and customer legs as *separate* audio streams (or a stereo
  file with one speaker per channel), which sidesteps diarization
  entirely for that split -- you already know who's who per channel. If
  you only have a single mixed-down mono stream (e.g. a room mic, a
  single-channel recording), that's exactly when this repo's diarization
  stage earns its keep.

## 5. Consent, retention, and PII-in-audio

Generic compliance framing -- validate against your actual jurisdiction and
regulatory obligations, this is not legal advice:

- **Recording consent**: most jurisdictions require some form of
  disclosure/consent before recording or transcribing a call (one-party vs.
  all-party consent laws vary by jurisdiction and, in the US, by state).
  This belongs in the call-handling layer upstream of this pipeline (an
  IVR announcement, a documented consent flow), not something the ASR
  service itself can determine.
- **Audio and transcripts both carry PII.** A transcript of a phone call
  routinely contains names, account numbers, addresses, and other sensitive
  data spoken aloud -- the same data-handling obligations that apply to
  structured PII (encryption at rest/in transit, access controls, retention
  limits, right-to-deletion workflows) apply to the raw audio and the
  transcript text produced here.
- **Retention policy**: this repo doesn't persist audio or transcripts
  anywhere by default (it streams events over the WebSocket and keeps no
  disk/database state) -- any production deployment that adds
  persistence (call recordings, transcript logs, the golden-dataset/
  evaluation tooling a downstream signal-detection service would add) needs
  an explicit, enforced retention/deletion policy, not an implicit "keep
  everything forever" default.
- **Diarized speaker labels are not identity.** `speaker_0`/`speaker_1` are
  stable *within one call* but carry no cross-call identity guarantee (this
  repo's clusterer resets per connection, by design -- see
  `ConnectionPipeline`'s docstring). Don't conflate a diarization label with
  a verified speaker identity in any downstream compliance-sensitive logic.
