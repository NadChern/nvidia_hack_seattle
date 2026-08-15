$ErrorActionPreference = "Stop"

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$script = Join-Path $PSScriptRoot "run_spike.py"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Run setup.ps1 first."
}

& $python $script
exit $LASTEXITCODE
