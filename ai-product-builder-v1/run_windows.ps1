param(
    [switch]$Demo,
    [int]$Port = 8767
)
$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    $python = Join-Path $HOME 'anaconda3\python.exe'
}
if (-not (Test-Path $python)) {
    throw 'No working Python found. Install Python 3.10+, or create a .venv in this folder.'
}
# API key and live/demo mode are configured from the browser Settings UI.
Push-Location $PSScriptRoot
try {
    & $python --version
    if ($Demo) { & $python server.py --demo --port $Port }
    else { & $python server.py --port $Port }
}
finally { Pop-Location }