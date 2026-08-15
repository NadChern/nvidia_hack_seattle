# Registration pipeline Phase-3 results

## Decision

**Adopt the bounded registration pipeline with fixture verification.** Vision arms the existing
EvidenceRing, performs segmentation and two-pass quality filtering, embeds through the same
masked crop path as matching, selects 2–4 diverse views, and persists them through Memory. Weak
capture is a terminal, counted failure rather than a silently weak gallery.

## Automated gates

- Good textured multi-view fixture: succeeds, quality yield 5/5, persists 2–4 views, and forces an
  immediate gallery refresh.
- Deliberately textureless fixture: rejected 1/1 with `too_few_quality_frames`; stores 0 views.
- Crop parity: enrollment and matching produce byte-identical image/mask arrays and identical
  fixture vectors; measured cosine is 1.0 within float32 rounding.
- Diversity fixture: selects the farthest distinct view and drops the remaining near-duplicate;
  selected count is within `[2, k]` for accepted registration fixtures.
- Label gate: an unconfigured `wallet` is rejected before the service promises registration.
- Session gate: capture without an active video epoch returns explicit `503 unavailable`.
- IMU absent path: `angular_velocity=None` leaves quality acceptance unchanged; a present
  over-limit gyro sample is rejected as `gyro_blur_risk`.

## Pending measurements

Registration success/honest-failure rates, selected-set mean pairwise cosine, auto-versus-curated
identity F1, and p95 wall time require the labeled real capture clips. IMU lift requires the
future glasses data-channel transport. Real C-RADIO extraction, YOLOE segmentation, and complete
registration wall time remain GPU integration measurements; no actual-glasses dependency was
introduced into the fixture path.
