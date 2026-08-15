# Privacy and Security

Local inference is a useful privacy property, but it does not make the system private by itself. This document defines the MVP trust boundary and the controls required before describing the assistant as privacy-friendly.

## MVP trust boundary

By default:

- camera, microphone, inference, memory, and evidence remain on the glasses and the local workstation;
- the workstation binds application APIs to the trusted local network, not the public internet;
- WebRTC media uses encrypted transport;
- no cloud speech, VLM, signaling, telemetry, or TURN relay is enabled;
- any optional external service is disabled until the user explicitly enables it and sees what data leaves the device.

Document the actual IP addresses, ports, processes, and external connections used in the demo. A local TURN or signaling service is inside the boundary; a hosted relay is not.

When using self-hosted LiveKit, restrict both signaling and RTC ports to the trusted LAN. Binding the signaling address alone may leave ICE/TCP listening on additional interfaces, so verify listeners and firewall policy on the GN100 rather than inferring exposure from the WebSocket URL.

### Optional external language-model evaluation

The Agent's production/default endpoint is loopback. A non-loopback model endpoint is an evaluation profile with a hard startup gate: it is rejected unless the operator explicitly sets `VMA_ALLOW_EXTERNAL_LLM=true`. The Agent status endpoint and console label the resulting backend as external; it is never an automatic fallback for a missing or failed local model.

When enabled, the external provider receives the transcribed question and the complete Memory `QueryResponse` returned to the `where_is` tool. That response may contain an object identifier, current or historical location, uncertainty, invalidation, candidate identifiers, timestamps, and evidence identifiers. Audio, images, video, evidence bytes, and filesystem paths are not placed in the Agent prompt. This narrower payload is still sensitive personal-location data and must be disclosed before evaluation.

The temporary laptop profiles use either ModelBest's MiniCPM API or an explicitly selected OpenRouter route to avoid loading an additional model into an 8 GB GPU. API keys are entered through protected runtime configuration, never committed or logged, and shared/free-tier credentials are not release credentials. OpenRouter free routes may involve provider routing, rate limits, and provider-specific retention. The hackathon GN100 target returns to the local trust boundary by hosting the selected verifier/Agent model locally. External-evaluation logs and retained provider data must be considered outside this repository's deletion guarantee. See [Agent Laptop Testing](14-Agent-Laptop-Testing.md) for the exact profiles and disclosures.

## User controls

The glasses experience must provide:

- an unambiguous recording indicator;
- pause and resume controls;
- a command to delete the current session;
- a way to view the evidence supporting an answer;
- an explicit message when a query cannot be answered safely.

The workstation interface should support object-memory export and deletion without requiring direct database access.

## Data minimization and retention

Recommended demo defaults:

| Data | Default retention |
|---|---|
| Rolling raw audio/video buffer | At most 60 seconds; discard unless needed for an event |
| Non-event frames | Do not persist |
| Evidence frames or short clips | Session-scoped; automatically delete after 24 hours |
| Structured event metadata | Session-scoped; automatically delete after 24 hours |
| Logs | Seven days, with transcripts, media, tokens, and precise evidence paths redacted |

Retention must be configurable. Deletion removes database records, evidence files, derived embeddings, caches, and backups within the documented scope. For a longer-lived prototype, obtain an explicit retention decision rather than silently changing these defaults.

## Bystanders and sensitive content

- Capture only what is necessary for the target-object task.
- Avoid storing faces, screens, documents, conversations, or precise room imagery when a cropped object-and-surface frame is sufficient.
- Prefer cropped evidence and optional on-device face blurring.
- Do not infer or store health conditions from medication-related objects.
- Provide a visible capture indicator and a clear explanation for participants in recorded test sessions.

These are product safeguards, not a claim of legal or regulatory compliance.

## Access and storage controls

- Authenticate the glasses, workstation UI, and service-to-service requests.
- Pair glasses through a short-TTL, single-use code issued by the Gateway's internal-bearer surface. The QR contains only `{gateway_url, pairing_code, expires_at}`—never the raw internal bearer.
- A successful claim returns an HMAC-signed, expiring device credential scoped to one `device_id`. It authorizes only session creation, refresh/deletion of that device's own session, its own HUD event socket, and a short-lived manual trigger for that session; it cannot list sessions, mint viewer grants, publish HUD events, read relays/status, or administer sessions.
- Device credentials are signed from a domain-separated key derived from the configured internal token, so they survive restart but rotate when that operator secret rotates. Pairing codes themselves are held only as hashes in bounded process memory and are consumed once.
- **There is no per-device revocation.** The credential is a stateless HMAC, so the only way to withdraw one is rotating `internal_api_token`, which unpairs every device at once. Expiry is therefore the sole bound on lost glasses: `device_credential_ttl_s` defaults to 24 hours, and re-pairing costs one QR scan. Accepted for the event; a revocation list is the fix if these credentials ever outlive a demo.
- The client stores one Gateway URL and credential at a time. Switching between laptop and GN100 obtains a fresh single-use pairing from the selected target; it never copies one target's device credential or operator token to the other. The ADB helper accepts the GN100 operator token only through an environment variable or a mode-600 file and does not place it in argv, QR payloads, or source control.
- **A device credential may only be presented in the `Authorization` header, never in a query string.** The operator token keeps that concession because a browser cannot set headers on a WebSocket; a week-long device secret in a URL would reach every request line anything records. Services that accept a query token run with `--no-access-log`, and the redaction filter strips `?token=` from messages and from log-record `args` as a second line of defence.
- **The glasses reach the gateway over cleartext HTTP/WS on the venue LAN**, declared in `apps/glasses-x3/.../network_security_config.xml` rather than a blanket manifest flag so the decision is reviewable in one place. The device credential, transcripts, and reply text are readable to anyone on that network. This is an accepted demo constraint bounded by the credential TTL, not a property of the design.
- Use least-privilege service accounts and per-session authorization.
- Encrypt persistent evidence and database backups.
- Keep secrets in protected runtime configuration; never put tokens in source, prompts, logs, or demo scripts.
- Generate evidence paths server-side and prevent path traversal.
- Verify evidence hashes before returning an image as support.
- Record access, export, and deletion operations without logging sensitive media.

## Container and supply-chain controls

- Pin release images and trusted base images by immutable digest.
- Build and publish Linux ARM64 images through the documented CI path.
- Run containers as non-root and with a read-only root filesystem where the service permits it.
- Never mount the Docker socket into an application or inference container.
- Inject secrets at runtime; do not store them in images, model caches, Compose files, logs, or CI artifacts.
- Give each service only the volumes it needs. Mount the verified model cache read-only in inference containers.
- Keep database, evidence, and model-cache volumes outside disposable image layers and include them in deletion and backup scope.
- Publish only documented trusted-LAN ports; keep private APIs on the internal Compose network.
- Require authenticated service identity, authorization, request timeouts, payload limits, and rate limits on internal evidence, verification, and query APIs; a trusted LAN is not an authentication mechanism.
- Use TLS for service traffic that leaves the protected Compose host or crosses an untrusted network boundary.
- Do not download unpinned models or executable code during normal service startup.
- Scan release images and record unresolved high-severity findings before deployment.
- Protect registry and NGC credentials with least privilege and rotate them after the event.
- Do not expose a remote Docker API on the GN100.

See [Development and Deployment](08-Development-and-Deployment.md) for image ownership, model manifests, release records, and rollback.

These controls intentionally exceed the trusted-isolated-network assumptions documented by the [NVIDIA VSS known limitations](https://docs.nvidia.com/vss/latest/Known-Limitations.html). The project borrows VSS pipeline boundaries, not its current security gaps.

## Failure behavior

If authentication, encrypted transport, storage, or the retention worker fails:

1. stop accepting new persistent memories;
2. continue only in a clearly labeled non-persistent mode when safe;
3. tell the user that memory is temporarily unavailable;
4. never fall back silently to an external cloud service.

## Demo privacy checklist

- [ ] Recording indicator visible
- [ ] Pause/resume works
- [ ] No unexpected external network connections
- [ ] Evidence is cropped or minimized
- [ ] Session deletion removes metadata and files
- [ ] Logs contain no raw media, transcripts, or tokens
- [ ] Authentication is enabled
- [ ] Retention job tested
- [ ] Participants understand what is recorded
