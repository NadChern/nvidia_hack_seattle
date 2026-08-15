# Development and Deployment

This document is the authority for local development environments, container ownership, continuous integration, GN100 deployment, and rollback.

Repository-wide Python, uv, FastAPI, testing, and service-layout conventions are defined in [Engineering Standards](11-Engineering-Standards.md) and enforced for agents through the repository `AGENTS.md` and local skill.

## Decisions

- Native development is supported on Apple-silicon macOS and Windows.
- Docker is not required for every local model workflow.
- Services deployed to the Acer GN100 must have a tested Linux ARM64/CUDA container path.
- Docker Compose is the deployment orchestrator for the MVP.
- Kubernetes is not part of the MVP.
- Pull-request CI is required; deployment to the shared GN100 is manually triggered.
- A successful cross-build is not proof that CUDA inference works on the GN100.
- The physical GN100 is the final release gate.

Docker standardizes the Linux userland and dependency graph. It does not make MLX, Windows CUDA, x86-64, Linux ARM64, and GB10 CUDA kernels interchangeable.

## Environment profiles

| Profile | Execution | Required validation |
|---|---|---|
| `dev-macos` | Native Apple-silicon tools and MLX-compatible models; optional Compose for ordinary services | Unit, contract, mock-integration, and shared English speech tests |
| `dev-windows-cuda` | Native Windows PyTorch/CUDA models; optional Compose for ordinary services | Unit, contract, mock-integration, and shared English speech tests |
| `dev-remote` | Local application connects to explicitly configured GN100 services | Contract, authentication, and failure-path tests |
| `ci` | CPU-friendly tests, mocked model adapters, and image builds | Formatting, unit, reducer, contract, configuration, and ARM64 build checks |
| `deploy-gn100` | Docker Compose on Linux ARM64 with NVIDIA CUDA runtime | Health, media, model, memory, privacy, soak, and rollback gates |

Developers may choose their compatible local model runtime. Code outside a model adapter must not depend on MLX, PyTorch, CUDA, operating-system paths, or a checkpoint layout. A remote GN100 profile is explicit; no local profile may silently select a cloud API.

## Service and image contract

Every GN100 service owner supplies:

- service code and a versioned API contract;
- a Dockerfile that builds the Linux ARM64 target;
- a locked dependency set;
- startup, liveness, and readiness behavior;
- deterministic configuration validation;
- unit and contract fixtures;
- a model manifest when the service loads models;
- redacted structured logs;
- documented persistent and temporary storage;
- graceful shutdown behavior.

An image that builds but cannot start, become ready, process its fixtures, or shut down cleanly on the GN100 is not deployable.

## GN100 Compose topology

The initial `compose.yaml` should define these logical services:

```text
livekit
media-worker
speech
vision
application-memory
database
```

The physical packaging may combine a pair of services when hackathon simplicity or latency justifies it, but their logical interfaces and ownership remain separate.

The VSS-inspired architecture patterns do not add deployment services. DeepStream, NVIDIA NIM, VIOS, Kafka, Redis, Elasticsearch, the VSS Agent, and Kubernetes are not required by the MVP Compose topology.

Use:

- an internal service network for private APIs;
- explicit trusted-LAN port publication for LiveKit and the user-facing API;
- NVIDIA GPU access only for services that perform inference;
- health checks and restart policies;
- named persistent volumes for database data and evidence;
- a pinned model-cache volume for model artifacts;
- bounded temporary storage for decoded media and transient inference data.

Docker Compose GPU access does not partition the GN100's unified memory. Measure the complete service set together and preserve the agreed operating-system and failure-recovery headroom.

## Model artifacts

Do not rely on an unversioned model download during service startup.

For each runtime profile, record:

- logical model and role;
- source repository and checkpoint;
- immutable revision or verified hash;
- converted artifact revision or hash when applicable;
- runtime and package versions;
- precision or quantization;
- expected license and access requirements;
- preprocessing, sample rate, prompt, and voice configuration;
- expected disk and unified-memory use.

Prefer a controlled, persistent model cache mounted read-only into inference services. Keep large model weights out of ordinary application image layers unless a specific distribution requirement makes a self-contained image necessary.

## CI policy

Every pull request must:

1. run formatting and static checks;
2. run unit, reducer, and API contract tests;
3. run service tests against mocked model adapters;
4. validate configuration and required health checks;
5. build every affected deployment image for `linux/arm64`;
6. report failures to the owning service rather than transferring them to the release owner.

CPU CI and architecture emulation are suitable for contracts, configuration, and packaging checks. They do not replace native CUDA execution.

Full model downloads and GPU benchmarks are not required on every pull request. Use pinned fixtures and lightweight adapters for ordinary CI.

## Release and CD policy

Deployment to the shared GN100 is a manual release operation.

For a release candidate:

1. build and publish images tagged with the source commit;
2. record immutable image digests and the model manifest;
3. preserve the previous known-good deployment manifest;
4. pull artifacts before the demo or test window;
5. validate GN100 driver, CUDA, free disk, model cache, volumes, and ports;
6. apply the candidate with Docker Compose;
7. wait for all readiness checks;
8. run the deployment acceptance suite;
9. record latency, peak unified memory, network observations, and failures;
10. promote the candidate only after the release gate passes.

Do not automatically deploy every merge to the shared GN100. A failed release must restore the previous Compose configuration and image digests without deleting database, evidence, or model-cache volumes.

## Kubernetes decision

Kubernetes is deferred. A single GN100 does not need cluster scheduling, multi-node failover, or rolling placement, while Kubernetes would add GPU-device, storage, secret, and LiveKit networking complexity.

Revisit Kubernetes only if the project gains at least one of these requirements:

- multiple inference nodes;
- multiple isolated deployment environments;
- high availability;
- rolling deployments without a maintenance window;
- multi-tenant scheduling or quotas;
- an external operations team that already standardizes on Kubernetes.

## Team boundary

Each service owner owns the service's code, Dockerfile, dependency lock, health behavior, model manifest, tests, and ARM64 build. The service owner fixes a failing service image.

The release owner owns:

- the top-level Compose definition;
- shared CI workflows and registry conventions;
- environment configuration and secret injection;
- deployment manifests and release records;
- GN100 deployment coordination;
- readiness aggregation, rollback, and demo fallback selection.

The release owner coordinates integration but does not inherit subsystem defects. Contract changes require approval and fixtures from both the provider and consumer.

## Release gate

A candidate is releasable only when:

- all required Linux ARM64 images are pinned by digest;
- Compose starts on the GN100 without an unexpected external dependency;
- every required readiness check passes;
- the actual glasses, running the pure-Kotlin client with pinned Android/Gradle and LiveKit Android SDK dependencies, pass camera, microphone, HUD, return-audio, codec, and reconnect tests against the GN100;
- Parakeet and Kokoro pass the shared English golden set;
- representative vision and memory scenarios pass end to end;
- the complete workload preserves the agreed unified-memory headroom;
- persistent data survives a service restart;
- deletion and retention controls pass;
- network exposure matches the trusted-LAN policy;
- rollback to the previous known-good release is demonstrated.

See [Recommended Architecture](01-Recommended-Architecture.md), [Hackathon Stack](03-Hackathon-Stack.md), [Evaluation Plan](04-Evaluation-Plan.md), [Team Split](05-Team-Split.md), [Privacy and Security](07-Privacy-and-Security.md), [Spike Plan](09-Spike-Plan.md), and [Engineering Standards](11-Engineering-Standards.md).
