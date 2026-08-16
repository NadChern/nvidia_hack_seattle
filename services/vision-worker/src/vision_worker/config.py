"""Service configuration.

All settings come from the environment with a `VMA_` prefix. Settings are
frozen and validated at startup so a misconfiguration fails the process
rather than surfacing as odd behaviour under load -- matching
`media_gateway.config` and `application_memory.config`.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

Environment = Literal["dev", "ci", "deploy"]
#: `fixture` needs no GPU and no model -- the ci and dev-macos path, and a
#: real deployment's honest state before a model is configured: it proves
#: every other piece of plumbing works while finding nothing. `yoloe` is the
#: real detector (`detect/yoloe.py`), requiring the `models` extra
#: (`uv sync --extra models`) and a CUDA-capable device.
DetectorKind = Literal["fixture", "yoloe"]
#: Personal-object identity is optional and annotates tracks without vetoing
#: detections or events. `fixture` is deterministic CPU CI; `radio` loads the
#: pinned C-RADIOv4 adapter from the models extra.
IdentityKind = Literal["none", "fixture", "radio"]
#: `none` (the default) runs the pipeline with no depth adapter at all --
#: the same image-space-only shape the service has always had, not a
#: degraded state. `fixture` scripts a constant range for testing the
#: wiring with no GPU.
#:
#: Two real adapters, and which one runs is a GPU-budget decision rather than a
#: quality preference:
#:
#: * `moge` (`depth/moge.py`) is the better geometry and the default wherever
#:   there is headroom -- the GN100 and any Linux+NVIDIA machine above the
#:   constrained-VRAM threshold in `scripts/dev_stack.sh`.
#: * `yolo` (`depth/yolo.py`) reuses the Ultralytics runtime YOLOE already
#:   loads, so it costs no second checkpoint. It exists for the 8 GB-class
#:   development GPU, where MoGe cannot coexist with the detector and Speech.
#:   `dev_stack.sh` selects it only under `VMA_ENABLE_CONSTRAINED_VISION=true`.
#:
#: Do not promote `yolo` to the deployment default because it is what the
#: laptop runs. See `model-manifest.toml`, whose `[yolo_depth]` license is
#: still `review-required`.
DepthKind = Literal["none", "fixture", "moge", "yolo"]
#: `rules` alone is deterministic and needs no model. `world_motion` adds the
#: DA3-backed world-trajectory veto on top of it -- see `Settings.verifier_kind`.
VerifierKind = Literal["rules", "world_motion", "vlm"]


def _env_file() -> str | None:
    """Load a local .env outside deploy only.

    A .env baked into a production image is exactly the secret-handling
    anti-pattern docs/07-Privacy-and-Security.md forbids.
    """
    return None if os.getenv("VMA_ENVIRONMENT") == "deploy" else ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VMA_",
        env_file=_env_file(),
        extra="ignore",
        frozen=True,
    )

    service_name: str = "vision-worker"
    environment: Environment = "dev"
    log_level: str = "INFO"

    # --- Upstream: the media relay -----------------------------------------
    #: The gateway's video relay endpoint. This service is a WebSocket
    #: consumer of it, never a publisher -- see
    #: `visual_memory_media_contract.client.MediaClient`.
    gateway_video_url: str = "ws://127.0.0.1:8080/v1/stream/video"
    internal_api_token: SecretStr | None = None
    #: The rate the relay actually delivers frames at, which is the gateway's
    #: `VMA_SAMPLE_FPS` -- **not** the 24fps the glasses capture at. The
    #: gateway relays a *sampled* stream; this default tracks its default.
    #: Every duration below is converted into a frame count at this rate, so
    #: setting it wrong does not fail loudly: it silently scales every
    #: threshold the state machine uses. `Pipeline` measures the rate it is
    #: really being fed and reports it at `/v1/status`, which is how a
    #: mismatch is caught.
    source_fps: float = Field(default=8.0, gt=0)

    # --- Downstream: the Memory Service -------------------------------------
    memory_base_url: str = "http://127.0.0.1:8081"
    memory_request_timeout_s: float = Field(default=10.0, gt=0)

    # --- Detection -----------------------------------------------------------
    detector_kind: DetectorKind = "fixture"
    #: Text prompts the detector targets, e.g. "keys,wallet". Empty means
    #: prompt-free / open-vocabulary -- see `detect.base.Detector.detect`.
    #: `NoDecode` for the same reason `media_gateway.config.
    #: device_id_allowlist` needs it: without it pydantic-settings JSON-decodes
    #: complex types before any validator runs, and a documented
    #: comma-separated value fails to parse.
    detection_labels: Annotated[tuple[str, ...], NoDecode] = ()
    #: The two checkpoints `detect/yoloe.py` keeps warm -- text-prompt for
    #: known targets, prompt-free for open vocabulary.
    #:
    #: The YOLO26-generation *large* variant, measured against the `11s` the
    #: prior-art project used and the `11m`/`26m` middle variants on the
    #: recorded clips in `media/clips`. At the confidence threshold the
    #: verifier actually applies, `26l` roughly doubles the usable detection
    #: rate over `11m` (51% vs 27% of frames on clip 1, 41% vs 26% on clip 3)
    #: for 2ms more per frame. Note `26m` is *worse* than `11m` -- the gain is
    #: specific to the large variant, not to the generation.
    yoloe_text_model: str = "yoloe-26l-seg.pt"
    yoloe_prompt_free_model: str = "yoloe-26l-seg-pf.pt"
    yoloe_score_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    #: Bound detector output before tracking, overlay serialization, and depth
    #: sampling. Open-vocabulary prompts can overlap semantically (for example
    #: laptop/monitor); keeping only the best boxes makes the HUD readable and
    #: puts a hard ceiling on per-frame downstream work.
    max_detections_per_frame: int = Field(default=20, ge=1, le=100)
    #: Force a torch device instead of detecting one. `None` (the default)
    #: picks CUDA, then Apple's MPS, then CPU -- see `detect/yoloe.py`'s
    #: `_select_device`. The escape hatch is for Metal specifically: MPS
    #: silently falls back to CPU for operations it has no kernel for, and
    #: YOLOE's text-prompt path runs CLIP, so `cpu` needs to be reachable
    #: without a code edit on the platform this project cannot test.
    yoloe_device: str | None = None

    # --- Personal-object identity -------------------------------------------
    identity_kind: IdentityKind = "none"
    identity_device: str | None = None
    identity_gallery_ttl_s: float = Field(default=30.0, gt=0)
    identity_track_frames: int = Field(default=3, ge=1, le=8)
    identity_min_detection_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    identity_min_scale: float = Field(default=0.01, ge=0.0, le=1.0)
    #: Keys-only Phase-0 probe starting values. Configurable and re-tuned once
    #: masked enrollment crops replace the conservative full-image probe.
    identity_min_cosine: float = Field(default=0.8334, ge=0.0, le=1.0)
    identity_min_margin: float = Field(default=0.0440, ge=0.0, le=1.0)
    identity_escalation_low: float = Field(default=0.8216, ge=0.0, le=1.0)
    identity_summary_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    identity_vlm_escalation: bool = True
    #: Promotion confidence is not raw cosine. Resolved matches are mapped to
    #: this Memory policy floor or above; unresolved objects retain detection
    #: confidence so identity cannot make the pre-existing world disappear.
    memory_min_identity_confidence: float = Field(default=0.7, ge=0.0, le=1.0)

    # --- Registration capture ------------------------------------------------
    registration_capture_seconds: float = Field(default=6.0, gt=0)
    registration_max_capture_seconds: float = Field(default=15.0, gt=0)
    #: Coarse temporal search scans evenly spaced frames in four-frame Cosmos
    #: batches before the image pass. Original relay frames near a temporal hit
    #: are then dynamically batched by vLLM for tight grounding.
    registration_temporal_max_frames: int = Field(default=16, ge=4, le=64)
    registration_temporal_batch_frames: int = Field(default=4, ge=2, le=8)
    registration_candidate_interval_seconds: float = Field(default=0.75, gt=0.0, le=3.0)
    registration_max_frames: int = Field(default=24, ge=2, le=240)
    registration_target_views: int = Field(default=6, ge=2, le=8)
    registration_min_views: int = Field(default=2, ge=2, le=8)
    registration_dedup_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    registration_min_mask_box_ratio: float = Field(default=0.4, ge=0.0, le=1.0)
    registration_max_mask_box_ratio: float = Field(default=1.0, ge=0.0, le=1.2)
    registration_relative_sharpness_floor: float = Field(default=0.5, ge=0.0)
    registration_max_angular_velocity_rad_s: float = Field(default=2.5, gt=0)

    # --- Window reasoner (see reason/cosmos.py, pipeline.py) -----------------
    #: The VLM that localizes objects and classifies events over a short video
    #: window -- this replaces the old detect/track/stability/verify chain.
    #: OpenAI-compatible (vLLM). The model runs wherever it runs; this service
    #: holds no weights. On the GN100 this is Cosmos 3 Nano at :8001/v1.
    #: `cosmos` is the real reasoner (`reason/cosmos.py`), needing a reachable
    #: vLLM endpoint. `fixture` returns a scripted empty result -- the ci/no-GPU
    #: shape that proves every other piece of plumbing works without a model.
    reason_kind: Literal["cosmos", "fixture"] = "cosmos"
    reason_base_url: str = "http://127.0.0.1:8001/v1"
    reason_model: str = "nvidia/Cosmos3-Nano"
    #: The rolling window handed to the reasoner, and how often a new one fires.
    #: Cosmos is ~5s+/call warm, so windows are short (few frames) and roughly
    #: non-overlapping -- the pipeline's event dedup tolerates a slow cadence.
    #: One window is analyzed at a time (one GPU, one model server).
    reason_window_seconds: float = Field(default=6.0, gt=0)
    reason_interval_seconds: float = Field(default=7.0, gt=0)
    reason_max_frames: int = Field(default=4, ge=1, le=16)
    #: Cosmos gives no numeric score; the memory contract needs one for
    #: `confidence.event`. Identity confidence is computed from the cosine.
    reason_event_confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    reason_max_tokens: int = Field(default=320, ge=64)
    reason_timeout_s: float = Field(default=120.0, gt=0)
    #: How long the same (object, action) is suppressed after a write, so
    #: overlapping windows do not re-report one placement many times.
    event_cooldown_seconds: float = Field(default=20.0, gt=0)
    #: Whether picked_up/carried are written to memory alongside placed. Off by
    #: default: at ~1fps Cosmos hallucinates handling on a resting object, and a
    #: single false pickup flips a confirmed placement to "moved afterward" -- so
    #: only `placed` is promoted, and "where is it" always reflects the last
    #: confirmed location. Turn on to record the full movement timeline once the
    #: frame rate is high enough for motion classification to be reliable.
    promote_motion_events: bool = False
    #: Cosmos boxes run tight/noisy on small objects; the identity crop pads the
    #: box by this fraction on each side before cropping (no segmenter -- SAM3
    #: is gated), keeping the object inside the frame the embedder sees.
    identity_box_padding: float = Field(default=0.12, ge=0.0, le=1.0)

    # --- Depth ---------------------------------------------------------------
    depth_kind: DepthKind = "none"
    moge_model_id: str = "Ruicheng/moge-2-vitl-normal"
    #: Lightweight metric-depth model using the already-installed Ultralytics
    #: runtime. This is the laptop overlay profile; MoGe remains the higher
    #: quality candidate/geometry adapter when memory permits.
    yolo_depth_model: str = "yolo26m-depth.pt"
    yolo_depth_device: str | None = None
    #: Off by default -- an extra PCA fit per candidate for a field
    #: (`Detection.box3d`) nothing in this service consumes yet. See
    #: `depth/moge.py`'s module docstring.
    moge_emit_box3d: bool = False
    #: Only used by `VMA_DEPTH_KIND=fixture`, for exercising the pipeline's
    #: depth wiring with no GPU.
    fixture_depth_range_m: float = Field(default=1.5, gt=0)

    # --- Stability thresholds (see domain/stability.py) ----------------------
    #: Expressed as durations because the machine compares them directly against
    #: the samples' own timestamps -- a frame count would silently mean a
    #: different real duration at every source rate (the exact bug this avoids).
    #: `/v1/status` reports the observed rate alongside, so an evaluation run can
    #: see how densely samples fell inside each window.
    #:
    #: Defaults are the plan's values: 0.5s of held position confirms a
    #: placement *after* observed motion, because motion-then-settle is strong
    #: evidence a placement just happened.
    dwell_seconds: float = Field(default=0.5, gt=0)
    #: Deliberately much longer -- a track that was already still when first
    #: seen looks identical to one that was placed before Vision was watching.
    passive_confirmation_seconds: float = Field(default=3.75, gt=0)
    #: How long a track may go undetected (occlusion, a missed frame) before a
    #: reappearance counts as a new sighting rather than a continuation.
    reacquire_within_seconds: float = Field(default=1.875, gt=0)
    #: How often "carried" re-fires while a track keeps moving.
    carried_emit_interval_seconds: float = Field(default=2.5, gt=0)
    #: World-space metres of drift per frame that still counts as "at rest".
    world_motion_threshold_m: float = Field(default=0.05, gt=0)
    #: Normalized image-space residual (object screen motion minus background
    #: motion) that still counts as "at rest" on the no-depth path.
    image_residual_threshold: float = Field(default=0.02, gt=0)

    # --- Evidence --------------------------------------------------------------
    #: How long the evidence ring retains already-sampled frames. An explicit,
    #: documented retention value per docs/07 -- reported at `/v1/status`,
    #: not a constant buried in code.
    evidence_ring_seconds: float = Field(default=30.0, gt=0)
    #: The frame rate evidence clips are encoded at. `None` (the default)
    #: means "whatever the source rate is", so a clip plays back at real
    #: speed. Pinning it to a different value is a deliberate choice, not a
    #: default: encoding a 2fps window at 24 produces a 12x timelapse, which
    #: is a poor thing to hand a human as evidence of an object being set
    #: down.
    clip_fps: float | None = Field(default=None, gt=0)

    # --- Verifier -------------------------------------------------------------
    #: `rules` is the deterministic default (`verify/rules.py`).
    #: `world_motion` wraps it with `verify/world_motion.py`, which
    #: reconstructs the candidate's window through `pose/da3.py` and vetoes a
    #: verdict the object's *world* trajectory contradicts -- the fix for
    #: pickups that are really the camera panning. Needs the `pose` extra and
    #: a CUDA device; degrades to plain `rules` if the checkpoint will not
    #: load.
    verifier_kind: VerifierKind = "rules"
    rule_verifier_min_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    rule_verifier_min_frame_count: int = Field(default=1, ge=1)

    # --- Deferred verification (see verify/pending.py) ------------------------
    #: Verification never runs on the frame loop. These bound what that costs:
    #: queued work holds its evidence frames, so depth is a memory ceiling, and
    #: a full queue drops candidates -- `/v1/status` reports both.
    verification_queue_depth: int = Field(default=8, ge=1)
    #: One by default. There is one GPU and one model server behind the VLM
    #: verifier, so a larger pool moves the queue inside the model server where
    #: nothing here can see or count it.
    verification_concurrency: int = Field(default=1, ge=1)
    #: How long shutdown waits for in-flight verification, rather than throwing
    #: away an answer that was nearly ready.
    #:
    #: Must stay below the Dockerfile's `--timeout-graceful-shutdown 20`, or
    #: uvicorn kills the process mid-drain and the wait buys nothing: a value
    #: larger than that is not a longer grace period, it is a guaranteed hard
    #: kill. One VLM call is ~20s, so a candidate mid-flight at shutdown may
    #: still be abandoned -- that is logged, with the pending count.
    shutdown_drain_timeout_s: float = Field(default=15.0, gt=0)

    # --- Overlay stream (see overlay/hub.py, api/overlay.py) ------------------
    #: Whether `WS /v1/overlay` streams per-frame detections to viewers. On by
    #: default because its cost is proportional to viewers -- with none
    #: connected the pipeline assembles nothing at all.
    overlay_enabled: bool = True
    #: A viewer is a browser tab. This bounds how many can be attached at once;
    #: each holds one pending overlay, so the memory cost is trivial and the
    #: real reason for a limit is to keep an accidental reconnect loop from
    #: accumulating sockets.
    overlay_max_viewers: int = Field(default=8, ge=1)
    #: How often depth is sampled for the overlay, in seconds. Depth is a
    #: second heavy model; running it per frame costs far more than the
    #: measurement is worth for a quantity that changes slowly, and on a
    #: machine already at its frame budget it would simply cause more frames
    #: to be dropped. So it runs at this cadence and each track keeps its last
    #: reading, with `depth_age_s` telling a viewer how stale that is.
    #:
    #: Only ever runs when a depth adapter is configured *and* somebody is
    #: watching -- see `Pipeline._should_sample_depth`.
    overlay_depth_interval_s: float = Field(default=1.0, gt=0)

    # --- VLM verifier (see verify/vlm.py) -------------------------------------
    #: Any OpenAI/Ollama-compatible chat endpoint. The model runs wherever it
    #: runs; this service holds no weights, which is what keeps it deployable
    #: on a machine that could not host one.
    vlm_base_url: str = "http://127.0.0.1:11434"
    vlm_model: str = "qwen3-vl:4b"
    #: Frames per window. Each costs roughly 550 tokens, so this and
    #: `vlm_num_ctx` move together -- 17 frames overflowed a 4096-token
    #: window the first time this ran for real.
    vlm_max_frames: int = Field(default=16, ge=2)
    vlm_num_ctx: int = Field(default=16384, ge=2048)
    vlm_timeout_s: float = Field(default=180.0, gt=0)
    #: How far before its last sighting a vanish window reaches. What matters
    #: is the approach -- a hand arriving -- not the empty table afterwards.
    vanish_lookback_s: float = Field(default=3.0, gt=0)

    # --- Window pose (see pose/da3.py) ----------------------------------------
    da3_model_id: str = "depth-anything/DA3-LARGE-1.1"
    #: Hard cap on frames per reconstruction, not a hint. Peak VRAM scales
    #: with window length -- 6 views cost 3.9GB and 13 cost 6.0GB of this
    #: card's 8GB with a detector also resident -- so a longer window is
    #: subsampled rather than reconstructed whole.
    da3_max_views: int = Field(default=8, ge=2)
    #: Metres. Used when DA3's scale was anchored to a metric depth adapter
    #: (`VMA_DEPTH_KIND=moge`), which is what makes real distances available.
    #: Measured basis: keys that never moved drifted 0.3cm while the camera
    #: moved 10cm.
    world_motion_still_m: float = Field(default=0.03, gt=0)
    world_motion_settled_m: float = Field(default=0.08, gt=0)
    #: Fractions of scene scale, used when no metric anchor was available --
    #: DA3's pose-capable checkpoint is not metric on its own. See
    #: `verify.world_motion.WorldMotionConfig`.
    world_motion_still_ratio: float = Field(default=0.02, gt=0)
    world_motion_settled_ratio: float = Field(default=0.06, gt=0)

    @field_validator("log_level")
    @classmethod
    def _upper_log_level(cls, value: str) -> str:
        return value.upper()

    @field_validator("detection_labels", mode="before")
    @classmethod
    def _split_labels(cls, value: object) -> object:
        """Accept a comma-separated string so the env var stays readable,
        matching `media_gateway.config.Settings._split_allowlist`."""
        if isinstance(value, str):
            if value.lstrip().startswith("["):
                raise ValueError(
                    "detection_labels is comma-separated, not JSON: "
                    'use VMA_DETECTION_LABELS="keys,wallet"'
                )
            return tuple(item.strip() for item in value.split(",") if item.strip())
        return value

    @model_validator(mode="after")
    def _identity_band_is_ordered(self) -> Settings:
        if self.identity_escalation_low > self.identity_min_cosine:
            raise ValueError("identity_escalation_low cannot exceed identity_min_cosine")
        if self.registration_min_views > self.registration_target_views:
            raise ValueError("registration_min_views cannot exceed registration_target_views")
        if self.registration_temporal_batch_frames > self.registration_temporal_max_frames:
            raise ValueError(
                "registration_temporal_batch_frames cannot exceed registration_temporal_max_frames"
            )
        if self.registration_capture_seconds > self.registration_max_capture_seconds:
            raise ValueError(
                "registration_capture_seconds cannot exceed registration_max_capture_seconds"
            )
        if self.registration_min_mask_box_ratio > self.registration_max_mask_box_ratio:
            raise ValueError(
                "registration_min_mask_box_ratio cannot exceed registration_max_mask_box_ratio"
            )
        return self

    @property
    def memory_token(self) -> str | None:
        return self.internal_api_token.get_secret_value() if self.internal_api_token else None

    @property
    def resolved_clip_fps(self) -> float:
        """`clip_fps` if it was set explicitly, else the source rate."""
        return self.clip_fps if self.clip_fps is not None else self.source_fps


@lru_cache
def get_settings() -> Settings:
    return Settings()


__all__ = [
    "DepthKind",
    "DetectorKind",
    "Environment",
    "IdentityKind",
    "Settings",
    "VerifierKind",
    "get_settings",
]
