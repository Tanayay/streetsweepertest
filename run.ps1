$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    Write-Host 'Python launcher not found. Install Python 3.11 first:' -ForegroundColor Red
    Write-Host 'winget install Python.Python.3.11' -ForegroundColor Yellow
    exit 1
}

if (-not (Test-Path '.venv\Scripts\python.exe')) {
    Write-Host 'Creating Python 3.11 environment...'
    py -3.11 -m venv .venv
}

$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$stamp = Join-Path $PSScriptRoot '.venv\.dependencies-installed'

if (-not (Test-Path $stamp)) {
    Write-Host 'Installing dependencies. This only happens the first time...'
    & $python -m pip install --upgrade pip setuptools wheel
    & $python -m pip install -r requirements.txt
    New-Item -ItemType File -Path $stamp -Force | Out-Null
}

Write-Host 'Starting ParkLink Street Sweeper Counter...' -ForegroundColor Cyan
Start-Process 'http://127.0.0.1:5000'
& $python app.py
