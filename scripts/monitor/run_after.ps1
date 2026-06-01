# run_after.ps1 — waits for slot 1 training to finish, then launches obs2
# Usage: run in background after respawn run is started on slot 1

$repo = "C:\Users\Jango\workspace\BeamNG"
$pidFile = "$repo\logs\remote\training.pid"
$logFile = "$repo\logs\remote\run_after.log"

function Log($msg) {
    $ts = Get-Date -Format "HH:mm:ss"
    "$ts $msg" | Tee-Object -FilePath $logFile -Append
}

Log "run_after.ps1 started — waiting for slot 1 respawn run to finish"

# Wait for the PID file to appear first (in case training just started)
$waited = 0
while (-not (Test-Path $pidFile)) {
    Start-Sleep -Seconds 30
    $waited += 30
    if ($waited -gt 300) { Log "WARNING: no PID file after 5 min — training may not have started"; break }
}

# Now wait for the training process to die
while ($true) {
    if (-not (Test-Path $pidFile)) {
        Log "PID file gone — training finished"
        break
    }
    $pid1 = Get-Content $pidFile -ErrorAction SilentlyContinue
    if ($pid1) {
        $proc = Get-Process -Id $pid1 -ErrorAction SilentlyContinue
        if (-not $proc) {
            Log "PID $pid1 no longer running — training finished"
            break
        }
        $steps = Get-Content "$repo\logs\remote\current_steps.txt" -ErrorAction SilentlyContinue
        Log "Still running — PID=$pid1 steps=$steps"
    }
    Start-Sleep -Seconds 300
}

Log "Launching obs2 run on port 25252..."

# Write slot 1 session config for obs2
$session = @{
    car             = "sbr"
    total_timesteps = 400000
    speed_factor    = 16
    timing_profile  = "det50ms"
    reward_config   = "v21b"
    obs_config      = "v2"
    fresh           = $false
    run_name        = "v21b_subaru_16x_obs2_454steps"
    max_damage      = 500
    debug_mode      = $false
    port            = 25252
    spawn_mode      = "bootstrap"
} | ConvertTo-Json
$session | Out-File -Encoding utf8 "$repo\config\launcher_session.json"

$python = "$repo\venv\Scripts\python.exe"
$script = "$repo\scripts\train\train_live_ppo_run.py"
$args = @(
    $script,
    "--session-file", "$repo\config\launcher_session.json",
    "--run-name", "v21b_subaru_16x_obs2_454steps",
    "--car", "sbr",
    "--reward-config", "v21b",
    "--obs-config", "v2",
    "--port", "25252"
)

Log "Command: $python $args"
& $python @args 2>&1 | Tee-Object -FilePath "$repo\logs\remote\obs2_training.log" -Append

Log "obs2 run finished."
