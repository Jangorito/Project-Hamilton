# Remote Access & Training Guide

## Network Setup

The training PC (JANGO-DESKTOP) is accessible over Tailscale VPN.

| Detail | Value |
|--------|-------|
| Tailscale IP | `100.120.21.69` |
| SSH port | `22` |
| Windows user | `Jango` |
| Repo path | `C:\Users\Jango\workspace\BeamNG` |

---

## Setting Up a New Remote Device

1. Install [Tailscale](https://tailscale.com/download) on the new device
2. Sign in with the **same account** used on JANGO-DESKTOP
3. Both devices will appear in the Tailscale admin panel automatically
4. Install an SSH client:
   - **iPhone/iPad** — Termius (free tier is sufficient)
   - **Android** — Termius or JuiceSSH
   - **Another PC/Mac** — built-in `ssh` in terminal

5. Connect:
   ```
   ssh Jango@100.120.21.69
   ```
   Use the Windows login password when prompted.

---

## Web Launcher GUI

The preferred way to control training. Run the Flask server on Jango, then open a browser.

### Start the launcher (on Jango via SSH)

```powershell
cd C:\Users\Jango\workspace\BeamNG
.\venv\Scripts\python.exe scripts\launcher\app.py
```

### Access from Student machine

**Same network (Tailscale):**
Open `http://100.120.21.69:5000` in your browser.

**SSH tunnel (if direct access is blocked):**
```powershell
# Run in a Student machine terminal — keep this session open
ssh -L 5000:127.0.0.1:5000 jango@100.120.21.69
```
Then open `http://localhost:5000`.

### Reload Flask after app changes

Use this after changing `scripts/launcher/app.py`, `scripts/launcher/templates/index.html`, or launcher-related code that Flask imports.

1. Stop any old local SSH tunnel with `Ctrl+C`, or close that terminal.
2. Stop/replace the current Flask server by running the VS Code task:
   `Terminal` -> `Run Task` -> `Jango: pull + restart Flask`
3. Keep the task terminal open. It SSHes to Jango, pulls the repo, kills whatever is on port `5000`, then starts Flask.
4. Open a new local PowerShell terminal. In this workspace it should default into the venv.
5. Start a fresh SSH tunnel:

```powershell
ssh -L 5000:127.0.0.1:5000 jango@100.120.21.69
```

6. Open `http://localhost:5000`.

---

## Launching Training (CLI fallback)

All training is launched via the PowerShell launcher script. Always `cd` to the repo root first.

```powershell
cd C:\Users\Jango\workspace\BeamNG
```

### Common commands

| Intent | Command |
|--------|---------|
| Resume latest run | `powershell -ExecutionPolicy Bypass -File .\scripts\train\launch_training.ps1` |
| Resume + stream to Twitch | `powershell -ExecutionPolicy Bypass -File .\scripts\train\launch_training.ps1 -Stream` |
| Fresh named run | `powershell -ExecutionPolicy Bypass -File .\scripts\train\launch_training.ps1 -Fresh -RunName v4_test` |
| Fresh run + stream | `powershell -ExecutionPolicy Bypass -File .\scripts\train\launch_training.ps1 -Fresh -RunName v4_test -Stream` |
| Check if training is running | `powershell -ExecutionPolicy Bypass -File .\scripts\train\launch_training.ps1 -Status` |
| Follow live log output | `powershell -ExecutionPolicy Bypass -File .\scripts\train\launch_training.ps1 -Tail` |
| Launch via scheduled task | `schtasks /run /tn "HamiltonTraining"` |

Training runs **detached** — closing the SSH session will not kill it.

`-Tail` and `-Follow` let you watch live output; `Ctrl+C` stops watching but leaves training running.

### Training script flags (advanced)

The launcher passes these through to `train_live_ppo_run.py`:

| Flag | Effect |
|------|--------|
| `--fresh` | Start a new run from scratch (fails if run name already exists) |
| `--run-name NAME` | Specify run name (omit to auto-resume latest or be prompted) |

Approximate training times at default pace (~13.5 steps/sec on Jango):

| Steps | Approx time |
|-------|-------------|
| 27,000 | ~2 hours |
| 55,000 | ~4 hours |
| 100,000 | ~7.5 hours |

---

## OBS / Twitch Streaming

The launcher can start and stop an OBS Twitch stream automatically around a training run.

### Prerequisites
- OBS is running on JANGO-DESKTOP with a scene capturing the BeamNG window
- Twitch account is connected in OBS settings
- OBS WebSocket Server is enabled: **Tools → WebSocket Server Settings → Enable WebSocket server**

### OBS WebSocket password (one-time setup)

If the WebSocket has a password set, store it as a persistent environment variable on JANGO-DESKTOP (run once in admin PowerShell locally):

```powershell
[System.Environment]::SetEnvironmentVariable("OBS_WEBSOCKET_PASSWORD", "your-password", "User")
```
Or:

PS$p = Join-Path $env:APPDATA 'obs-studio\plugin_config\obs-websocket\config.json'; $c = Get-Content $p -Raw | ConvertFrom-Json; $c | Select-Object server_enabled,server_port,auth_required,server_password

server_enabled server_port auth_required server_pas   
                                         sword        
-------------- ----------- ------------- ----------   
          True        4455          True WrCEAQd...   


PS C:\Users\Jango\workspace\BeamNG> $p = Join-Path $env:APPDATA 'obs-studio\plugin_config\obs-websocket\config.json'; $c = Get-Content $p -Raw | ConvertFrom-Json; [System.Environment]::SetEnvironmentVariable('OBS_WEBSOCKET_PASSWORD', $c.server_password, 'User'); $env:OBS_WEBSOCKET_PASSWORD = $c.server_password

then run commands like this:

PS C:\Users\Jango\workspace\BeamNG> .\venv\Scripts\python.exe .\scripts\env\obs_control.py start --password

If no password is set in OBS, skip this step.

### Control OBS stream manually

```powershell
# From inside the repo, with venv active
python scripts\env\obs_control.py start
python scripts\env\obs_control.py stop
python scripts\env\obs_control.py status
python scripts\env\obs_control.py screenshot
```

With a password:
```powershell
python scripts\env\obs_control.py start --password your-password
```

---

## Scheduled Tasks (Jango)

These Windows scheduled tasks exist on JANGO-DESKTOP and can be triggered over SSH without needing an interactive session.

| Task name | What it does |
|-----------|-------------|
| `HamiltonTraining` | Runs `launch_training.ps1 -Stream` in an interactive (Session 1) desktop process — required for BeamNG GPU rendering |
| `BeamNGDirect` | Launches BeamNG standalone for diagnostics (no training) |
| `HamiltonWatch` | Runs `scripts/launcher/watch_task_runner.py`, which launches `watch_model.py` from the interactive desktop session |
| `WakeScreen` | Sends a mouse input event to wake a sleeping monitor |
| `WakeScreen2` | Alternative wake task using `SendInput` — use this one if `WakeScreen` doesn't work |

### Run a task remotely

```powershell
schtasks /run /tn "HamiltonTraining"
schtasks /run /tn "HamiltonWatch"
schtasks /run /tn "WakeScreen2"
schtasks /run /tn "BeamNGDirect"
```

> **Why scheduled tasks?** SSH sessions land in Windows Session 0 (no GPU rendering, no visible window). Scheduled tasks with the `/it` flag run in Session 1 (the interactive desktop), which BeamNG and OBS require.
>
> `watch_model.py` is especially sensitive to this because it opens BeamNG in a graphical mode. If Flask is started over SSH, use `HamiltonWatch` or start the Flask server locally on Jango so the BeamNG window has an interactive desktop.

---

## Git — Sync Between Machines

```powershell
# Pull latest changes onto Jango (run via SSH)
cd C:\Users\Jango\workspace\BeamNG
git pull

# Push changes from Student machine
cd C:\Users\Student\Workspace\Project-Hamilton
git add <files>
git commit -m "message"
git push
```

---

## Log Files

Logs from remote-triggered runs are saved to `logs/remote/`.

| File | Purpose |
|------|---------|
| `logs/remote/training_YYYYMMDD_HHMMSS.log` | Full stdout+stderr for each run |
| `logs/remote/latest.log` | Symlink pointing to the most recent log |
| `logs/remote/training.pid` | PID of the running training process |

---

## Troubleshooting

**"The argument to -File does not exist"**
You are not in the repo directory, or using the old path. Run `cd C:\Users\Jango\workspace\BeamNG` first. Script is now at `scripts\train\launch_training.ps1`.

**"Execution policy" error**
Use `powershell -ExecutionPolicy Bypass -File .\scripts\...` as shown above, or run this once on the PC:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

**"Training already running"**
Check with `-Status`. If the process is stale (crashed), delete `logs\remote\training.pid` and try again.

**OBS "Could not connect to OBS WebSocket"**
OBS must be open and WebSocket server enabled. Default port is `4455`.

**SSH connection refused**
Verify the `sshd` service is running on JANGO-DESKTOP:
```powershell
Get-Service sshd
```
If stopped: `Start-Service sshd`
