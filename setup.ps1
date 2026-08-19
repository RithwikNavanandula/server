#Requires -Version 5.1
<#
.SYNOPSIS
  One-shot setup for AI CCTV on the Windows ProLiant (server folder).

.DESCRIPTION
  Right-click setup.ps1 → Run with PowerShell (Admin preferred for firewall).
#>

$ErrorActionPreference = 'Stop'
$ServerDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ModelsDir = Join-Path $ServerDir 'models'
$VenvDir = Join-Path $ServerDir 'venv'
$ReqFile = Join-Path $ServerDir 'requirements.txt'
$Python = $null

function Test-PortAvailable($port) {
    try {
        $conn = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
        return ($null -eq $conn)
    } catch {
        return $false
    }
}

function Find-FreePort($startPort = 5000, $maxPort = 5050) {
    for ($p = $startPort; $p -le $maxPort; $p++) {
        if (Test-PortAvailable $p) {
            return $p
        }
    }
    throw "No free TCP port found between $startPort and $maxPort."
}


function Write-Step($n, $msg) {
    Write-Host ""
    Write-Host "[$n] $msg" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "======================================================" -ForegroundColor Green
Write-Host "  AI CCTV Server — Windows Setup" -ForegroundColor Green
Write-Host "  Folder: $ServerDir" -ForegroundColor Green
Write-Host "======================================================" -ForegroundColor Green

if (-not (Test-Path $ReqFile)) {
    Write-Host "  Missing requirements.txt in $ServerDir" -ForegroundColor Red
    Pause
    exit 1
}

Write-Step 1 "Checking Python 3.10+ ..."
$candidates = @(
    (Get-Command python -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source),
    (Get-Command py -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source),
    "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
    "C:\Python311\python.exe",
    "C:\Python312\python.exe"
) | Where-Object { $_ -and (Test-Path $_) }

foreach ($c in $candidates) {
    try {
        $ver = & $c -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        if ($ver -match '^3\.(1[0-9]|[2-9][0-9])') {
            $Python = $c
            if ((Split-Path -Leaf $c) -eq 'py.exe') { $Python = 'py' }
            Write-Host "  Found Python $ver -> $c" -ForegroundColor Green
            break
        }
    } catch {}
}

if (-not $Python) {
    Write-Host "  Python 3.10+ not found. Install from python.org (Add to PATH), then re-run." -ForegroundColor Red
    Pause
    exit 1
}

Write-Step 2 "Checking YOLO models in $ModelsDir ..."
if (-not (Test-Path (Join-Path $ModelsDir 'sugar_bag_final.pt'))) {
    Write-Host "  WARNING: sugar_bag_final.pt missing — copy .pt files into server\models\" -ForegroundColor Yellow
} else {
    Write-Host "  sugar_bag_final.pt OK" -ForegroundColor Green
}

Write-Step 3 "Creating virtualenv + installing packages ..."
if (-not (Test-Path $VenvDir)) {
    if ($Python -eq 'py') { & py -3 -m venv $VenvDir } else { & $Python -m venv $VenvDir }
}
$Pip = Join-Path $VenvDir 'Scripts\pip.exe'
$PyVenv = Join-Path $VenvDir 'Scripts\python.exe'
if (-not (Test-Path $PyVenv)) { Write-Host "  venv failed" -ForegroundColor Red; exit 1 }

& $Pip install --upgrade pip
Write-Host "  Installing CPU PyTorch first (several minutes)..." -ForegroundColor Yellow
& $Pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
Write-Host "  Installing remaining requirements (torch already installed — will not replace)..." -ForegroundColor Yellow
# --upgrade-strategy only-if-needed avoids pulling a second CUDA torch from PyPI
& $Pip install --upgrade-strategy only-if-needed -r $ReqFile

Write-Step 4 "Configuring .env and selecting server port ..."

$EnvFile = Join-Path $ServerDir '.env'
$defaultCloud = 'https://rishi01.pythonanywhere.com'

if (Test-Path $EnvFile) {
    Write-Host "  .env exists — keeping existing configuration" -ForegroundColor Yellow

    $envLines = Get-Content $EnvFile
    $existingPort = $null

    foreach ($line in $envLines) {
        if ($line -match '^\s*PORT\s*=\s*(\d+)\s*$') {
            $existingPort = [int]$Matches[1]
            break
        }
    }

    if ($existingPort) {
    $SelectedPort = $existingPort

    if (Test-PortAvailable $SelectedPort) {
        Write-Host "  Keeping existing PORT=$SelectedPort (currently available)" -ForegroundColor Green
    } else {
        Write-Host "  Keeping existing PORT=$SelectedPort (currently in use)" -ForegroundColor Yellow
        Write-Host "  The setup will NOT change the configured port." -ForegroundColor Yellow
    }
}
else {
    $SelectedPort = Find-FreePort 5000 5050
    Write-Host "  No PORT configured. Selected free port: $SelectedPort" -ForegroundColor Green

    Add-Content -Path $EnvFile -Value "PORT=$SelectedPort"
}
} else {
    $cloud = Read-Host "  PythonAnywhere / cloud URL [$defaultCloud]"
    if ([string]::IsNullOrWhiteSpace($cloud)) {
        $cloud = $defaultCloud
    }

    $edgeSecret = Read-Host "  EDGE_SYNC_SECRET (must match cloud)"
    if ([string]::IsNullOrWhiteSpace($edgeSecret)) {
        throw "EDGE_SYNC_SECRET is required."
    }

    $jwtSecret = Read-Host "  JWT_SECRET (must match cloud)"
    if ([string]::IsNullOrWhiteSpace($jwtSecret)) {
        throw "JWT_SECRET is required."
    }

    $SelectedPort = Find-FreePort 5000 5050

@"
CLOUD_URL=$cloud
EDGE_SYNC_SECRET=$edgeSecret
JWT_SECRET=$jwtSecret
EDGE_FRAME_FPS=3
ML_DEVICE=cpu
MODEL_DIR=$ModelsDir
PORT=$SelectedPort
"@ | Set-Content -Path $EnvFile -Encoding UTF8

    Write-Host "  Created .env with PORT=$SelectedPort" -ForegroundColor Green
}

Write-Host "  Selected server port: $SelectedPort" -ForegroundColor Green

Write-Step 5 "Firewall port $SelectedPort ..."
try {
    $isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    if ($isAdmin) {
        New-NetFirewallRule -DisplayName "AI CCTV Server $SelectedPort" -Direction Inbound -Protocol TCP -LocalPort $SelectedPort -Action Allow -ErrorAction SilentlyContinue | Out-Null
        Write-Host "  TCP $SelectedPort allowed" -ForegroundColor Green
    } else {
        Write-Host "  Not admin — open TCP $SelectedPort manually if needed" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  Skipped: $_" -ForegroundColor Yellow
}

if (-not (Test-Path (Join-Path $ServerDir 'start.bat'))) {
    Write-Host "  WARNING: start.bat missing — create it or pull from git" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "======================================================" -ForegroundColor Green
Write-Host "  SETUP COMPLETE — double-click start.bat" -ForegroundColor Green
Write-Host "======================================================" -ForegroundColor Green
Write-Host "  Server port: $SelectedPort"
Write-Host "  Cloud EDGE_API_URL must ultimately point to a public HTTPS endpoint for this port"
Write-Host "  Cloud login: demo@aicctv.com / demo123"
Write-Host ""
Pause
