#Requires -Version 5.1
<#
.SYNOPSIS
    Start a detached training run and optionally follow its output.

.EXAMPLE
    # Resume latest run in background
    .\scripts\launch_training.ps1

    # Start a fresh run and stream to Twitch via OBS
    .\scripts\launch_training.ps1 -Fresh -RunName v3_test -Stream

    # Start and follow output live
    .\scripts\launch_training.ps1 -Fresh -RunName v3_test -Follow

    # Check if training is running
    .\scripts\launch_training.ps1 -Status

    # Tail the most recent log
    .\scripts\launch_training.ps1 -Tail
#>
param(
    [switch]$Fresh,
    [string]$RunName        = "",
    [switch]$Follow,
    [switch]$Status,
    [switch]$Tail,
    [switch]$Stream,
    [int]   $OBSPort        = 4455,
    [string]$BeamNGProcess  = "BeamNG.tech.x64"
)

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$LogDir   = Join-Path $RepoRoot "logs\remote"
$PidFile  = Join-Path $LogDir "training.pid"
$LogLink  = Join-Path $LogDir "latest.log"
$SessionConfigFile = Join-Path $RepoRoot "config\launcher_session.json"

function Get-LatestLog {
    if (Test-Path $LogLink) {
        $item = Get-Item $LogLink -ErrorAction SilentlyContinue
        # Real symlink — resolve to target
        if ($item -and $item.LinkType) { return $item.Target }
        # Plain-text fallback written when symlink creation failed
        $content = (Get-Content $LogLink -Raw -ErrorAction SilentlyContinue).Trim()
        if ($content -and (Test-Path $content)) { return $content }
    }
    $logs = Get-ChildItem $LogDir -Filter "training_*.log" -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending
    if ($logs) { return $logs[0].FullName }
    return $null
}

function Get-RunningProcess {
    if (-not (Test-Path $PidFile)) { return $null }
    $savedPid = (Get-Content $PidFile -Raw).Trim()
    return Get-Process -Id $savedPid -ErrorAction SilentlyContinue
}

# ── Status ────────────────────────────────────────────────────────────────────
if ($Status) {
    $proc = Get-RunningProcess
    if ($proc) {
        Write-Host "Training is running  (PID $($proc.Id), CPU $([math]::Round($proc.CPU,1))s)"
    } else {
        Write-Host "No training process running."
    }
    $log = Get-LatestLog
    if ($log) { Write-Host "Latest log : $log" }
    exit
}

# ── Tail only ─────────────────────────────────────────────────────────────────
if ($Tail) {
    $log = Get-LatestLog
    if (-not $log) { Write-Host "No log file found."; exit 1 }
    Write-Host "Tailing $log  (Ctrl+C to stop)`n"
    Get-Content $log -Wait
    exit
}

# ── Guard: don't double-launch ────────────────────────────────────────────────
$existing = Get-RunningProcess
if ($existing) {
    Write-Host "Training already running (PID $($existing.Id))."
    Write-Host "Use -Status or -Tail to monitor it."
    exit 1
}

# ── Resolve python ────────────────────────────────────────────────────────────
$PythonExe = Join-Path $RepoRoot "venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"
}
if (-not (Test-Path $PythonExe)) { $PythonExe = "python" }

# ── Build argument string ─────────────────────────────────────────────────────
$TrainArgs = "scripts\train\train_live_ppo_run.py"
if ($Fresh)   { $TrainArgs += " --fresh" }
if ($RunName) { $TrainArgs += " --run-name $RunName" }

if (Test-Path $SessionConfigFile) {
    try {
        $SessionConfig = Get-Content $SessionConfigFile -Raw | ConvertFrom-Json
        if ($SessionConfig.stream -eq $true) { $Stream = $true }
    } catch {
        Write-Host "Warning: could not read launcher session config - continuing with CLI flags."
    }
}

# ── Prepare log file ──────────────────────────────────────────────────────────
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile   = Join-Path $LogDir "training_$Timestamp.log"

# Keep a stable "latest" pointer
if (Test-Path $LogLink) { Remove-Item $LogLink -Force }
New-Item -ItemType SymbolicLink -Path $LogLink -Target $LogFile -ErrorAction SilentlyContinue | Out-Null
# Fallback if symlinks aren't allowed: just copy path into a text file
if (-not (Test-Path $LogLink)) {
    $LogFile | Out-File $LogLink -Encoding utf8
}

# ── OBS: start stream ────────────────────────────────────────────────────────
$OBSArgs = "scripts\env\obs_control.py --port $OBSPort --process $BeamNGProcess"
if ($env:OBS_WEBSOCKET_PASSWORD) { $OBSArgs += " --password $env:OBS_WEBSOCKET_PASSWORD" }

if ($Stream) {
    Write-Host "Starting OBS stream..."
    & $PythonExe $OBSArgs.Split(" ") start
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Warning: could not start OBS stream - training will still launch."
    }
}

# ── Launch detached powershell that runs python and captures both streams ─────
$watchOBS = if ($Stream) { "Start-Process '$PythonExe' -ArgumentList '$OBSArgs watch'.Split(' ') -WindowStyle Hidden; " } else { "" }
$stopOBS  = if ($Stream) { "; & '$PythonExe' $OBSArgs stop" } else { "" }
$inner = "Set-Location '$RepoRoot'; ${watchOBS}& '$PythonExe' $TrainArgs 2>&1 | Tee-Object -FilePath '$LogFile'${stopOBS}"
$proc  = Start-Process powershell `
    -ArgumentList "-NonInteractive", "-Command", $inner `
    -WindowStyle Hidden `
    -PassThru

$proc.Id | Out-File $PidFile -Encoding utf8

Write-Host "Training started."
Write-Host "  PID : $($proc.Id)"
Write-Host "  Log : $LogFile"
if ($Stream) { Write-Host "  OBS stream will stop automatically when training finishes." }
Write-Host ""
Write-Host "Commands:"
Write-Host "  .\scripts\launch_training.ps1 -Status   # check if running"
Write-Host "  .\scripts\launch_training.ps1 -Tail     # follow output"

if ($Follow) {
    Write-Host "`nFollowing output (Ctrl+C to stop watching - training keeps running)`n"
    Start-Sleep -Seconds 1
    Get-Content $LogFile -Wait
}
