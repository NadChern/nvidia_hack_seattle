# speech

Repository-standard `inference` service for the Visual Memory Assistant.

Owns speech in both directions for the wearer: transcribing what they say (STT, Parakeet) and speaking the assistant's answers back to them (TTS, Kokoro). It never touches WebRTC or LiveKit directly — audio in and audio out both go exclusively through the Media Gateway's relay, consumed and produced through `MediaClient` from `packages/media-contract`. It is `inference`-kind in `service.toml` because it owns two pinned models — see `model-manifest.toml` for the exact revisions and artifact sizes, not restated here so the pin lives in one place.

**Status: Stage D done — this service is at the repository's Definition of Done.** It ingests real audio into contiguous PCM segments, resamples them to the rate each model expects, transcribes them for real (Parakeet, streamed over a WebSocket), and synthesizes real speech back (Kokoro, over HTTP). Structured/redacted logging, validated startup config, `/health/live` + `/health/ready`, and a working Linux ARM64 Dockerfile are all in place. See "What's built vs. not yet" and "Integration surface" below for exactly what that does and doesn't include.

## Development stages

This is this service's own build plan — a staging convention the Speech owner uses to track progress against, not a repo-wide convention. Don't assume other services in this repository number their work the same way.

| Stage | What it delivers | Status |
|---|---|---|
| A | Audio ingestion: relay audio → contiguous PCM segments, no models | Done |
| B · Part 1 | Resampling + typed `SpeechToText` boundary + a stub (no real model) | Done |
| B · Part 2 | Real Parakeet STT adapter wired behind the `SpeechToText` interface | Done |
| C · Part 1 | Real Kokoro TTS adapter wired behind a new `TextToSpeech` interface | Done |
| C · Part 2 | `POST /v1/synthesize` — TTS exposed over HTTP | Done |
| C · Part 3 | `WS /v1/stt/{session_id}` — STT exposed as a streaming WebSocket | Done |
| D | Finish to Definition of Done (privacy-safe structured logging, validated config, health checks, ARM64 Dockerfile, all gates) | Done |

- Linux CUDA uses Parakeet TDT 0.6B v3 through Transformers and Kokoro-82M through its Torch runtime. The portable `Dockerfile` still runs stubs by design; `Dockerfile.cuda` installs the locked CUDA group for the physical GN100. Both backends remain behind the same `SpeechToText`/`TextToSpeech` interfaces as the MLX development path.
- Whether Speech calls the Memory query API directly or hands a transcript to a separate Agent layer is still an open question — see `role-prompts/Speech.md`. That question, the return-audio transport, and wiring this service into `compose.yaml` are all explicitly out of Stage D's scope; see "Integration surface."

## Design and philosophy

### Why this service only ever talks to `MediaClient`

The Media Gateway holds the only inference LiveKit subscription on purpose — it keeps inference decode cost, token surface, and WebRTC complexity in one place. The read-only operator viewer is the explicit non-inference carve-out. Speech honors its boundary completely: it never joins a room, never holds a token, and never parses the relay's wire framing itself. Every byte in comes from `MediaClient` iterating `ws://<gateway>/v1/stream/audio`, and (once return audio exists) every byte out will go back out through the same client. If something the relay should provide isn't exposed by `MediaClient`, the answer is to ask the Media Gateway's owner to add it, not to reach past the client and hand-parse the stream — see `role-prompts/Speech.md`'s hard rules for why that line is firm.

### The `pts_samples` continuity principle

The relay coalesces audio into `audio_chunk` messages, each carrying a `sequence` number and a `pts_samples` value — the cumulative sample count since the current epoch began. It is tempting to assume a stream is continuous because its messages arrive in order, but `sequence` only proves messages weren't reordered, not that no audio was lost between them. The Media Gateway's own audio path can shed no audio at the socket level (`docs/12-Media-Relay-Contract.md` is explicit that audio is never dropped there), but network conditions between gateway and consumer, or a consumer that briefly falls behind, can still produce a real gap in what a consumer actually receives. `pts_samples` is what catches that: if a chunk's `pts_samples` is greater than the previous chunk's `pts_samples + samples`, samples went missing in between, and `sequence` alone would never reveal it. An undetected gap here isn't a cosmetic bug — handing a transcriber two unrelated stretches of audio stitched together as if they were continuous corrupts the transcript with no error anywhere in the pipeline to catch it. `continuity.py` exists to make that detection mechanical rather than something every future call site has to remember to reimplement correctly.

### Epoch resets discard fragments, they don't finish them

`epoch_started` is the reset signal for a stream: per the Media Relay Contract, a consumer must discard tracker state and any partially accumulated data on it. The subtlety worth stating plainly is *why* that has to mean discard and not finish-and-emit: the only way a builder can still have unflushed audio in it when a fresh `epoch_started` arrives is if the previous epoch never reached its own clean `epoch_ended` — in practice, a reconnect cut it short. That leftover is an interrupted fragment, not a verified-complete segment, and emitting it as one would silently hand a real gap-filled or truncated stretch of audio downstream as if it were trustworthy. Treating `epoch_started` as "throw the fragment away and start clean" is what makes the same handler correct for both an ordinary epoch change and a reconnect: `MediaClient` re-sends a synthetic `epoch_started` for every still-active epoch after reconnecting, so there is no separate "we just reconnected" signal to special-case — one correct `epoch_started` handler already covers it. `tests/test_ingest.py::test_ingest_resets_cleanly_across_a_reconnect` exists specifically because an earlier version of this got the direction backwards (see the Tests section below).

### The `AudioSegment` boundary

`ingest.py` yields `AudioSegment` — a frozen Pydantic v2 model carrying `session_id`, `epoch_id`, `sample_rate`, `channels`, `sample_format`, `pts_samples_start`, `samples`, `first_sample_captured_at`, and the raw `pcm` bytes for one contiguous run of audio with no detected gap inside it. It deliberately does none of the following yet: it does not decode the PCM into a numpy array, does not resample it, does not play it back, and does not transcribe it. Each of those is a real later stage of this role with its own design questions (a decode/resample step has to pick a target format before Parakeet can even run), and folding any of them into the ingestion boundary now would make that boundary responsible for decisions it isn't ready to make.

### If a spoken turn is clipped or delayed

Four settings and boundaries decide whether a spoken sentence survives
segmentation intact. Symptoms map to them as follows:

| Symptom | Setting | Why |
|---|---|---|
| The sentence is cut off before you finish it | `VMA_STT_UTTERANCE_SILENCE_MS` (1000) | A pause longer than this ends the utterance. 700 ms was tuned on a desk mic and cut wearers off at natural breaths. |
| The **start** of "hey memory" is missing | `VMA_STT_PREROLL_MS` (600) | The trim that removes dead air measures from where the detector grew *confident*, not where the sound began. The margin kept is `preroll − attack delay`, so a slow attack eats the first phoneme. At 300 ms a measured 320 ms attack kept **−20 ms**. |
| Words drop out mid-sentence | `VMA_STT_VAD_THRESHOLD` (0.5) | Noise suppression and AGC on a head-worn mic flatten inter-word dips below the threshold. |
| The first words become an immediate turn after waiting quietly | `VMA_STT_UTTERANCE_MAX_SECONDS` (8) | This emergency ceiling starts at the first detected speech frame, never at the previous boundary; idle microphone time cannot spend the spoken-turn budget. |

The Agent receives every completed non-empty transcript; Speech is responsible
only for finding audio boundaries, not for classifying the user's intent.

### Known limits, flagged rather than worked around

Two real gaps exist right now, both intentionally left unaddressed rather than papered over:

- **The `1011 audio_backpressure` close is invisible to this service.** The Media Relay Contract documents that the gateway closes a subscriber's socket with code `1011` if that subscriber's audio queue overflows, and that this should be treated as a real signal, not a generic disconnect. `MediaClient`'s public API does not support that distinction today: only close code `1008` gets special handling (raised as `MediaClientError`); every other close, including `1011`, is caught generically inside `MediaClient.__aiter__` and logged by the client's own logger as "disconnected, reconnecting," with no way for a consumer to recover the original close code or reason. `ingest.py`'s module docstring documents this in full. It has been raised as something to ask Alex about, since the same special-case pattern already used for `1008` could plausibly extend to `1011` — it has not been worked around by reaching into `MediaClient` internals or hand-parsing the socket, both of which are against this role's hard rules.
- **Sample rates don't natively line up, though resampling now exists.** The gateway's audio relay defaults to 48 kHz (`services/media-gateway/src/media_gateway/config.py`, itself configurable). Parakeet checkpoints commonly expect 16 kHz mono input; Kokoro commonly outputs 24 kHz. None of those three numbers match on the wire. `config.py`'s `expected_audio_sample_rate` setting lets `ingest.py` detect and log a mismatch per chunk rather than silently assume 48 kHz is correct, and `resample.py` (Stage B Part 1) converts audio to `stt_target_sample_rate` before it reaches a real model. What's still open isn't *whether* resampling happens — it does — but *who should own it*: this service resampling locally, or the gateway's own rate changing for this consumer, is still an open question for the team, not a decision made unilaterally here.

## What's built vs. not yet

**Stage A (done):** real audio ingestion into contiguous PCM segments, with no models anywhere in the path. Concretely: `continuity.py`'s gap detection, `ingest.py`'s epoch-aware, reconnect-safe, gap-splitting consumer built on `MediaClient`, and `config.py`'s two settings for the gateway's audio URL and expected sample rate. All of it is exercised end to end against a recorded fixture with no gateway, no LiveKit, and no network running.

**Stage B (done):** resampling, a typed STT boundary, and a real STT backend. `resample.py` converts an `AudioSegment`'s PCM to the rate `config.py`'s `stt_target_sample_rate` setting names, using `soxr` for correct resampling rather than naive decimation. `stt.py` defines the `Transcript` result shape and a `SpeechToText` interface. `parakeet_backend.py`'s `ParakeetMlxSpeechToText` implements it with actual `parakeet-mlx` inference — see `model-manifest.toml` for the pinned revision and artifact details, not restated here so the pin lives in one place.

**Stage C (done):** the mirror image for text-to-speech, plus both directions exposed over HTTP/WebSocket. `tts.py` defines `SpeechAudio` and a `TextToSpeech` interface; `kokoro_backend.py`'s `KokoroMlxTextToSpeech` implements it with real `mlx-audio`/Kokoro inference. `POST /v1/synthesize` and `WS /v1/stt/{session_id}` (both described under "Integration surface" below) are how a running `speech` process actually does something — `main.py` now wires `ingest_segments`/`SpeechToText`/`TextToSpeech` together as part of the FastAPI app, not just as independently-tested pieces.

**Model runtimes are optional hardware profiles.** `parakeet-mlx` and `mlx-audio` are Darwin-only (`uv sync --group mlx`); Torch, Transformers/Parakeet, and Kokoro are Linux CUDA-only (`uv sync --group cuda`). Every hardware import is lazy, so the portable image and offline tests import cleanly without either group. `main.py` selects MLX, then usable CUDA, then stubs once at startup. `StubSpeechToText`/`StubTextToSpeech` remain deliberately available for CPU CI and the portable Compose profile; `compose.gpu.yaml` builds `Dockerfile.cuda` and selects the real CUDA backends on the GN100.

The GN100's GB10 needs the ARM64 cu130 nightly pinned in `pyproject.toml`; stable Torch 2.6 imports there but has no compute-capability-12.1 kernels. This exception is limited to Linux ARM64, was proven with a real BF16 CUDA operation, and is recorded in `standards-exception.toml`. Linux x86_64 remains on the measured stable cu126 build.

**Stage D (done):** brought the service to the repository's Definition of Done. Structured, redacted JSON logging (`logging.py`, identical to `media_gateway`'s and `application_memory`'s); `config.py`'s `log_level` setting; a working Linux ARM64 Docker build (the Dockerfile's build context had to move to the repository root — see below); and a full pass of every required gate. None of this changed any endpoint's behavior.

**Still outside this service:** changing the relay or Memory contracts. Return audio and query orchestration are owned by the Gateway and Agent integration layers respectively; Speech keeps exposing its existing HTTP/WebSocket contract unchanged.

## Integration surface

What Alex (or anyone else) can call today, and what's deliberately not here yet:

- **`POST /v1/synthesize`** — `{"text": str, "voice"?: str, "lang_code"?: str}` → a self-describing `audio/wav` response (real RIFF/WAVE header, sample rate and channel count readable straight off the file, no side-channel config needed to interpret it). `voice`/`lang_code` are optional overrides of `config.py`'s defaults.
- **`WS /v1/stt/{session_id}`** — connect once the glasses' audio session is live on the gateway; receive one `Transcript` JSON message per contiguous audio segment as this service produces them (`{"text", "session_id", "epoch_id", "pts_samples_start", "samples", "sample_rate"}`). `session_id` scopes it to one session on the gateway's single shared relay stream — a mismatched or not-yet-started session_id just waits, it does not error.
- **Not here, and not this service's job to add:** sending synthesized audio back to the wearer through the gateway (blocked on `MediaClient` gaining a send method), anything that calls Memory or an Agent layer with a transcript, and any change to `compose.yaml` or how this service gets deployed alongside the others. All three are integration work for whoever picks them up next, not something Stage D does on their behalf.

## No-hardware start path

Everything above is testable with no gateway, no LiveKit, and no model running, using the recorded `audio_session_basic` fixture and `packages/media-contract`'s `replay_server`:

```python
from visual_memory_media_contract.testing import replay_server

from speech.ingest import ingest_segments

async with replay_server("audio_session_basic") as url:
    segments = [segment async for segment in ingest_segments(url, reconnect=False)]
```

That fixture carries three seconds of 48 kHz mono PCM with one deliberate ~500ms gap partway through — enough to exercise the full ingestion path (epoch start, chunk accumulation, gap-triggered segment split, epoch end) without any hardware. `tests/test_ingest.py` and `tests/test_audio_continuity.py` are this exact pattern; running the suite (see Checks below) is the fastest way to see it work.

## Tests

| Test file | Test name | Purpose (what it checks & why) | Pass criteria (the actual assertions) |
|---|---|---|---|
| `tests/test_audio_continuity.py` | `test_speech_detects_the_deliberate_gap_in_audio_session_basic` | Verifies gap detection reads `pts_samples` arithmetic rather than `sequence` or message count, since the latter would silently corrupt a transcript by hiding a real loss of audio. | `sequence` values run `0..29` with no gaps (proving a sequence-only check would miss the loss); `ContinuityTracker` finds exactly 1 gap; that gap's `lost_seconds == 0.5` (within `±0.01`). |
| `tests/test_ingest.py` | `test_ingest_splits_into_two_contiguous_segments_at_the_gap` | Verifies the full `ingest_segments` pipeline correctly splits the fixture into two contiguous `AudioSegment`s at the detected gap, with accurate sample counts and PCM sizes. | Exactly 2 segments; first has `pts_samples_start == 0` and `samples == 48_000`; second has `pts_samples_start == 72_000` and `samples == 96_000`; each segment's `len(pcm) == samples * 2 * channels`. |
| `tests/test_ingest.py` | `test_ingest_resets_cleanly_across_a_reconnect` | Verifies a genuine `MediaClient` reconnect (forced via `flaky_replay_server`, dropped after 5 wire frames) discards the interrupted fragment instead of leaking it into the result — this test caught a real bug where `epoch_started` handling finished a leftover fragment instead of discarding it. | Exactly 2 segments in the final result; `segments[0].samples == 48_000`; `segments[1].samples == 96_000` — identical to the clean run, proving the interrupted attempt contributed nothing. |
| `tests/test_resample.py` | `test_resample_pcm_produces_exact_expected_sample_count_and_length` | Verifies `resample_pcm`'s own arithmetic in isolation (synthetic sine wave, no fixture, no async). | 48 kHz → 16 kHz on 1 second of audio yields exactly 16,000 samples (a clean 3:1 downsample). |
| `tests/test_resample.py` | `test_resample_pcm_is_a_noop_when_rates_already_match` | Verifies resampling is skipped, not run as a no-op pass, when source and target rates are already equal. | Output bytes are identical to the input bytes. |
| `tests/test_resample.py` | `test_resample_pcm_rejects_an_unsupported_sample_format` | Verifies an unsupported sample format fails loudly instead of silently mis-decoding. | Raises `ValueError` mentioning `"s16le"`. |
| `tests/test_resample.py` | `test_resample_segment_on_real_ingested_segments_from_the_fixture` | Verifies `resample_segment` end to end on the two real segments `audio_session_basic` produces, including that identity fields survive unchanged. | Resampled sample counts are exactly 16,000 and 32,000; `session_id`/`epoch_id`/`pts_samples_start` match the pre-resample originals. |
| `tests/test_stt.py` | `test_stub_speech_to_text_returns_a_populated_transcript` | Verifies `StubSpeechToText` returns a well-formed `Transcript` whose source-location fields match the segment it transcribed. | Non-empty `text`; `session_id`/`epoch_id`/`pts_samples_start`/`samples`/`sample_rate` all match the source segment. |
| `tests/test_stt.py` | `test_offline_pipeline_ingest_resample_transcribe` | Verifies the full `ingest → resample → transcribe` pipeline end to end with the stub — the shape real Parakeet wiring replaces `StubSpeechToText` inside without changing anything upstream. | One `Transcript` per ingested segment (2 total); each has `sample_rate` equal to the configured target rate; source-location fields survive resampling. |
| `tests/test_tts.py` | `test_stub_text_to_speech_returns_well_formed_speech_audio` | Verifies `StubTextToSpeech` returns well-formed, real (if meaningless) `SpeechAudio` — mono s16le PCM at the configured rate. | `audio.text == "hello"`; `channels == 1`; `sample_format == "s16le"`; non-empty `pcm` with an even byte length (whole s16le samples). |
| `tests/test_health.py` | `test_liveness` | Confirms the FastAPI app boots and the required `/health/live` liveness route responds correctly. | Response status `200`; JSON body's `status` field equals `"ok"`. |
| `tests/test_health.py` | `test_readiness` | Confirms the required `/health/ready` readiness route responds correctly. | Response status `200`; JSON body's `status` field equals `"ready"`. |
| `tests/test_logging.py` | 16 tests (redaction guarantees) | Verifies the structured JSON logging/redaction filter (`logging.py`) — the enforcement behind "never logs raw audio or transcript text," not just a one-time audit. Identical suite to `media_gateway`'s and `application_memory`'s, plus one PCM-shaped case. | Secrets/tokens/JWTs redacted anywhere in the message or `extra` fields (including nested dicts/lists); raw `bytes` collapse to a byte count, never content; long strings truncate except the message itself; uvicorn's ANSI `color_message` is dropped. |
| `tests/test_api_synthesize.py` | `test_synthesize_returns_a_valid_wav` | Verifies `POST /v1/synthesize` returns a real, well-formed WAV file (forces the stub backend, so this is about the endpoint's plumbing, not which backend produced the bytes). | Status `200`; `content-type: audio/wav`; the WAV's frame rate/channels/sample width are readable and correct; non-zero frame count. |
| `tests/test_api_synthesize.py` | `test_synthesize_rejects_empty_text` | Verifies empty `text` is rejected as a request-validation error, not silently synthesized as nothing. | Status `422`. |
| `tests/test_api_synthesize.py` | `test_synthesize_accepts_voice_and_lang_code_overrides` | Verifies the optional `voice`/`lang_code` overrides are accepted by the request shape (the stub ignores their values, but the surface must exist). | Status `200`. |
| `tests/test_kokoro_backend.py` | `test_kokoro_mlx_synthesizes_a_short_phrase` | Real end-to-end inference through the MLX Kokoro backend. **Skips by default** unless `mlx`/`mlx-audio`/`misaki` are installed, so the default suite stays fully offline. Writes a WAV to `tmp_path` only (never into the service directory) purely for human spot-checking. | Non-empty PCM with an even byte length; `sample_rate` matches configured output rate; mono s16le. |
| `tests/test_api_synthesize_kokoro.py` | `test_synthesize_with_real_kokoro_backend` | Real end-to-end inference through `POST /v1/synthesize` itself, not just the backend in isolation. Lives in its own file, separate from `test_api_synthesize.py`'s stub-forcing `autouse` fixture, which would otherwise silently override it back to the stub. **Skips by default** without `mlx`/`mlx-audio`/`misaki`. | Status `200`; returns a real, non-trivial WAV. |
| `tests/test_parakeet_backend.py` | `test_parakeet_mlx_transcribes_a_known_clip` | Real end-to-end inference through the MLX Parakeet backend on a known clip generated with macOS `say`. **Skips by default** unless `mlx`/`parakeet-mlx` and `say` are all available. | A non-empty transcript whose text contains a real word from the known phrase (`"test"` or `"parakeet"`, case-insensitive). |
| `tests/test_api_stt.py` | `test_stt_streams_one_transcript_per_ingested_segment` | Verifies `WS /v1/stt/{session_id}` streams one `Transcript` JSON message per segment, in order, with correct boundaries (forces the stub backend + a `replay_server` fixture, so this is fully offline). | Exactly 2 messages received; `session_id` matches on both; `pts_samples_start` is `0` then `72_000`; both have non-empty `text`. |
| `tests/test_api_stt.py` | `test_stt_ignores_other_sessions_on_the_same_relay` | Verifies the `session_id` filter actually filters, and that connecting for a session_id absent from the fixture still tears `MediaClient` down cleanly on disconnect rather than hanging — a session that never matches leaves `ingest_segments` waiting indefinitely by design, so this also exercises the endpoint's concurrent disconnect-watch path. | Connecting and immediately closing completes promptly (does not hang). |
| `tests/test_api_stt_parakeet.py` | `test_stt_streams_real_transcripts_via_parakeet` | Real end-to-end inference through `WS /v1/stt/{session_id}` itself. Assertions are structural only — `audio_session_basic` isn't guaranteed real speech, so word content isn't asserted; `test_parakeet_backend.py` already covers transcription accuracy. **Skips by default** without `mlx`/`parakeet-mlx`. | 2 messages; correct `session_id` and `pts_samples_start` boundaries; `text` is a string on both. |

**Intentionally not tested:** the `1011 audio_backpressure` close behavior described above. Neither the fixture/replay tooling in `packages/media-contract` nor `MediaClient`'s public API can produce or expose that specific close code to a consumer, so there is nothing real to test against — see `ingest.py`'s module docstring for the full explanation. A test asserting behavior here would necessarily be faking coverage of something the current stack cannot exercise.

## Backend selection

`VMA_TTS_BACKEND=auto` is the default and selects MLX, CUDA, or the import-fallback stub as described above. `VMA_TTS_BACKEND=stub` is an explicit diagnostic mode: STT selection remains unchanged, but synthesis uses the deterministic silent WAV backend and `/v1/status` reports `tts.real=false`. The measured 8 GB WSL laptop runs Parakeet and Kokoro together; an earlier shutdown attributed to the reply-audio handoff was traced to a Console reload bug rather than insufficient Kokoro headroom.

`VMA_WARM_MODELS_ON_STARTUP=true` makes readiness wait for selected adapters that expose an initializer. The CUDA profile loads Parakeet first, then loads Kokoro and synthesizes one discarded warmup phrase so the first wearer query pays neither model-load nor default-voice load latency. Use this on a pre-seeded deployment together with Hugging Face/Transformers offline mode; a missing artifact then fails startup honestly instead of dropping the first STT listener while attempting network metadata requests.

## Development

```text
uv sync --frozen --all-groups
uv run uvicorn speech.main:app --reload
```

## Checks

```text
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest
```

`pytest` runs quietly by default. For verbose output that lists every test node id individually with its result, plus captured log output per test, override the project's default options for one run:

```text
uv run pytest -o addopts= -v -rA
```

## Docker

Build **from the repository root**, not this directory — the Dockerfile needs the sibling `packages/media-contract` package, which sits outside this service's own directory:

```text
docker build --platform linux/arm64 -f services/speech/Dockerfile -t speech .
```

The portable image installs base dependencies only and runs `StubSpeechToText`/`StubTextToSpeech`. Build the CUDA image for the physical Linux ARM64 gate:

```text
docker build --platform linux/arm64 -f services/speech/Dockerfile.cuda -t vma/speech:gn100-cuda .
docker compose -f compose.yaml -f compose.gpu.yaml up -d speech
```

`Dockerfile.cuda` installs the locked `cuda` dependency group, reserves one warm model instance per backend, and uses `speech-model-cache` for Hugging Face artifacts. The deployment network has no egress, so seed that volume before startup and keep `HF_HUB_OFFLINE=1`. A successful build is only the architecture gate; backend names from `/v1/status` plus real STT/TTS inference on the GN100 are the physical CUDA gate.

## Related

- [Role brief](../../role-prompts/Speech.md) — scope, interfaces, hard rules, and the open questions this README's "Known limits" section summarizes
- [Media Relay Contract](../../docs/12-Media-Relay-Contract.md) — the wire protocol this service consumes; the single source of truth for message shapes, not restated here
- [Dev Onboarding](../../docs/13-Dev-Onboarding.md) — repository-wide invariants, commands, and failure modes for the media stack this service depends on
- [Media Gateway](../media-gateway/README.md) — the service on the other end of `MediaClient`; read this for how to run a real gateway locally once this service needs one
- [Privacy and Security](../../docs/07-Privacy-and-Security.md) — the logging/retention rules `logging.py` and this README's privacy notes enforce
- [Team Split — Definition of Done](../../docs/05-Team-Split.md) — the checklist Stage D verified this service against
