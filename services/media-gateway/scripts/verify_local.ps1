#Requires -Version 5.1
<#
.SYNOPSIS
Verify the media gateway on a dev machine, with no glasses and no GN100.

.DESCRIPTION
The Windows twin of verify_local.sh. The assertions live in pytest, not here:
tests/integration/ already ports the S01 spike's ten checks and runs them
against the real service. This script is the operator-facing sequence around
them, so that "does my machine work?" is one command.

Run it from services/media-gateway.

.PARAMETER Quick
Skip the live LiveKit round trip.

.PARAMETER Docker
Also build the image and run the emulated arm64 packaging gate.

.EXAMPLE
.\scripts\verify_local.ps1

.EXAMPLE
.\scripts\verify_local.ps1 -Quick
#>
[CmdletBinding()]
param(
    [switch]$Quick,
    [switch]$Docker
)

$ErrorActionPreference = "Stop"

$serviceDir = Split-Path -Parent $PSScriptRoot
$repoRoot = Split-Path -Parent (Split-Path -Parent $serviceDir)
$livekitUrl = if ($env:VMA_TEST_LIVEKIT_URL) { $env:VMA_TEST_LIVEKIT_URL } else { "ws://127.0.0.1:7880" }
$startedLiveKit = $false

function Write-Section([string]$Text) {
    Write-Host ""
    Write-Host "== $Text" -ForegroundColor White
}

function Invoke-Step([string]$What, [scriptblock]$Action) {
    & $Action
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAIL $What" -ForegroundColor Red
        exit 1
    }
}

function Test-LiveKitUp {
    try {
        Invoke-WebRequest -Uri "http://127.0.0.1:7880" -TimeoutSec 2 -UseBasicParsing | Out-Null
        return $true
    } catch {
        return $false
    }
}

try {
    # --- Structural --------------------------------------------------------

    Write-Section "Repository standards"
    Invoke-Step "repository structure" {
        python "$repoRoot\.agents\skills\visual-memory-repo-standards\scripts\validate_repo.py"
    }

    Write-Section "Shared contract package"
    Push-Location "$repoRoot\packages\media-contract"
    Invoke-Step "media-contract sync"       { uv sync --frozen --all-groups }
    Invoke-Step "media-contract formatting" { uv run ruff format --check . }
    Invoke-Step "media-contract lint"       { uv run ruff check . }
    Invoke-Step "media-contract types"      { uv run pyright }
    Invoke-Step "media-contract tests"      { uv run pytest }
    Pop-Location

    Write-Section "Media gateway"
    Push-Location $serviceDir
    Invoke-Step "gateway sync"       { uv sync --frozen --all-groups }
    Invoke-Step "gateway formatting" { uv run ruff format --check . }
    Invoke-Step "gateway lint"       { uv run ruff check . }
    Invoke-Step "gateway types"      { uv run pyright }
    Invoke-Step "gateway tests"      { uv run pytest }
    Pop-Location

    # --- Container ---------------------------------------------------------

    if ($Docker) {
        Write-Section "Image build (context is the repository root)"
        Invoke-Step "docker build" {
            docker build -f "$serviceDir\Dockerfile" -t vma/media-gateway:dev $repoRoot
        }
        Write-Section "ARM64 packaging gate (emulated; layering only, never CUDA or GN100)"
        Invoke-Step "arm64 build" {
            docker buildx build --platform linux/arm64 -f "$serviceDir\Dockerfile" $repoRoot
        }
    }

    # --- Live round trip ---------------------------------------------------

    if ($Quick) {
        Write-Section "Skipping the live round trip (-Quick)"
        Write-Host "`nOK  structural and unit checks passed." -ForegroundColor Green
        exit 0
    }

    Write-Section "LiveKit server"
    if (Test-LiveKitUp) {
        Write-Host "already running at $livekitUrl"
        if (-not $env:VMA_LIVEKIT_API_KEY -or -not $env:VMA_LIVEKIT_API_SECRET) {
            Write-Host @"
FAIL a LiveKit server is already running, but VMA_LIVEKIT_API_KEY and
     VMA_LIVEKIT_API_SECRET are not set. They must match the pair that server
     was started with, or every token it issues will be rejected.
"@ -ForegroundColor Red
            exit 1
        }
    } else {
        if (-not $env:VMA_LIVEKIT_API_KEY -or -not $env:VMA_LIVEKIT_API_SECRET) {
            # Ephemeral and never written to disk. The gateway's validator
            # refuses the well-known dev values, so a generated pair is the only
            # thing that works.
            $bytes = New-Object byte[] 24
            [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
            $env:VMA_LIVEKIT_API_SECRET = -join ($bytes | ForEach-Object { $_.ToString("x2") })
            $env:VMA_LIVEKIT_API_KEY = "vma-verify-$($env:VMA_LIVEKIT_API_SECRET.Substring(0, 6))"
            Write-Host "generated an ephemeral credential pair for this run"
        }
        Write-Host "starting one from compose.dev.yaml"
        Invoke-Step "start LiveKit" {
            docker compose -f "$repoRoot\compose.dev.yaml" up -d livekit
        }
        $startedLiveKit = $true
        for ($i = 0; $i -lt 30; $i++) {
            if (Test-LiveKitUp) { break }
            Start-Sleep -Seconds 1
        }
        if (-not (Test-LiveKitUp)) {
            Write-Host "FAIL LiveKit did not become reachable" -ForegroundColor Red
            exit 1
        }
    }

    Write-Section "Listener exposure"
    # docs/07: verify listeners, never infer exposure from the WebSocket URL.
    Invoke-Step "listener exposure" {
        python "$repoRoot\tools\dev-livekit\check_listeners.py"
    }

    Write-Section "Round trip through a real server (the spike's ten assertions)"
    Push-Location $serviceDir
    $env:VMA_TEST_LIVEKIT_URL = $livekitUrl
    Invoke-Step "integration round trip" { uv run pytest tests/integration -m livekit }
    Pop-Location

    Write-Host "`nOK  everything passed, including the privacy sweep for" -ForegroundColor Green
    Write-Host "    off-machine connections." -ForegroundColor Green
    Write-Host @"

To drive it by hand with your own camera and microphone, run
.\scripts\dev_stack.sh from the repository root and open http://localhost:5173
-- see services\media-gateway\README.md.
"@
}
finally {
    if ($startedLiveKit) {
        Write-Section "Stopping the LiveKit server this script started"
        docker compose -f "$repoRoot\compose.dev.yaml" down 2>&1 | Out-Null
    }
}
