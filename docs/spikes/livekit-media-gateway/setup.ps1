$ErrorActionPreference = "Stop"

$spikeRoot = $PSScriptRoot
$venvPython = Join-Path $spikeRoot ".venv\Scripts\python.exe"
$toolsRoot = Join-Path $spikeRoot ".tools"
$archivePath = Join-Path $toolsRoot "livekit_1.13.4_windows_amd64.zip"
$serverRoot = Join-Path $toolsRoot "livekit_1.13.4"
$serverPath = Join-Path $serverRoot "livekit-server.exe"
$downloadUrl = "https://github.com/livekit/livekit/releases/download/v1.13.4/livekit_1.13.4_windows_amd64.zip"
$expectedSha256 = "A326E025DE516E93DFB3719BCD28E5A4AC16F21BCF1EF562499403CA98CC65FE"

if (-not (Test-Path -LiteralPath $venvPython)) {
    py -3.11 -m venv (Join-Path $spikeRoot ".venv")
}

& $venvPython -m pip install --disable-pip-version-check -r (Join-Path $spikeRoot "requirements.txt")

New-Item -ItemType Directory -Force -Path $toolsRoot | Out-Null
if (-not (Test-Path -LiteralPath $archivePath)) {
    Invoke-WebRequest -Uri $downloadUrl -OutFile $archivePath
}

$actualSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $archivePath).Hash
if ($actualSha256 -ne $expectedSha256) {
    throw "LiveKit archive checksum mismatch. Expected $expectedSha256, got $actualSha256."
}

if (-not (Test-Path -LiteralPath $serverPath)) {
    Expand-Archive -LiteralPath $archivePath -DestinationPath $serverRoot
}

& $serverPath --version
Write-Output "Spike setup complete."
