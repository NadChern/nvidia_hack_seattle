# Console enrollment Phase-5 results

## Decision

**Adopt the optional operator workflow for demo setup.** The Console now provides an Enroll tab
that uses Vision's configured detection labels, requires a published session, creates and arms a
registration, polls bounded capture/extraction progress, and previews durable selected crops.
Confirm keeps the object; discard uses Memory's checked delete path and preserves the review state
if deletion fails.

Registered per-track matches are visually distinguished in the live overlay and include the
identity confidence when available. This is diagnostic presentation only; it does not alter
candidate or Memory truth semantics.

## Automated gates

- Component fixture covers create → capture → terminal poll → authenticated crop preview → discard.
- Existing console tests pass.
- TypeScript production build passes against overlay schema `1.4`.
- Oxlint reports only pre-existing vendored UI warnings.

## Physical demo gate

Reference thumbnail quality, rotate-object instructions, and end-to-end confirmation timing still
require a live glasses video session and real C-RADIO model. No raw enrollment video is added to
the Console or persisted.
