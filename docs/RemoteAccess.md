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

## Launching Training

All training is launched via the PowerShell launcher script. Always `cd` to the repo root first.

```powershell
cd C:\Users\Jango\workspace\BeamNG
```

### Common commands

| Intent | Command |
|--------|---------|
| Resume latest run | `powershell -ExecutionPolicy Bypass -File .\scripts\launch_training.ps1` |
| Resume + stream to Twitch | `powershell -ExecutionPolicy Bypass -File .\scripts\launch_training.ps1 -Stream` |
| Fresh named run | `powershell -ExecutionPolicy Bypass -File .\scripts\launch_training.ps1 -Fresh -RunName v4_test` |
| Fresh run + stream | `powershell -ExecutionPolicy Bypass -File .\scripts\launch_training.ps1 -Fresh -RunName v4_test -Stream` |
| Check if training is running | `powershell -ExecutionPolicy Bypass -File .\scripts\launch_training.ps1 -Status` |
| Follow live log output | `powershell -ExecutionPolicy Bypass -File .\scripts\launch_training.ps1 -Tail` |

Training runs **detached** — closing the SSH session will not kill it.

`-Tail` and `-Follow` let you watch live output; `Ctrl+C` stops watching but leaves training running.

### Training script flags (advanced)

The launcher passes these through to `train_live_ppo_run.py`:

| Flag | Effect |
|------|--------|
| `--fresh` | Start a new run from scratch (fails if run name already exists) |
| `--run-name NAME` | Specify run name (omit to auto-resume latest or be prompted) |

Tunable constants (edit `scripts/train_live_ppo_run.py` directly):

| Constant | Default | Approx time |
|----------|---------|-------------|
| `TOTAL_TIMESTEPS = 27_000` | — | ~2 hours |
| `TOTAL_TIMESTEPS = 55_000` | — | ~4 hours |
| `TOTAL_TIMESTEPS = 100_000` | — | ~7.5 hours |

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

If no password is set in OBS, skip this step.

### Control OBS stream manually

```powershell
# From inside the repo, with venv active
python scripts\obs_control.py start
python scripts\obs_control.py stop
python scripts\obs_control.py status
```

With a password:
```powershell
python scripts\obs_control.py start --password your-password
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
You are not in the repo directory. Run `cd C:\Users\Jango\workspace\BeamNG` first.

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
