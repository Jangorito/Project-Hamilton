# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project overview

**Project Hamilton** — a PPO-based autonomous racing agent trained inside BeamNG.tech (soft-body physics simulator) on Hirochi Raceway. The research question is a 2×2 factorial: how do vehicle dynamics (ETK K-Series vs SBR4) and reward function design (V1 vs V2.1) independently affect learning? V2.1 replaces V1's hand-tuned linear curvature penalty with a physics-informed reward grounded in traction-budget theory (Beckman's *Physics of Racing*). Additional research avenues (checkpoint respawning ✅ implemented, racing line reward, behavioural cloning) are in `TODO.md`.

---

## Environment setup

The venv lives at `venv/` (not `.venv/`). Always activate before running anything:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

The package uses a `src/` layout without an installable `pyproject.toml`. Set `PYTHONPATH` for any module-style invocation:

```powershell
$env:PYTHONPATH = "$PWD\src"
```

Scripts under `scripts/` add `src/` to `sys.path` themselves, so they don't need the env var — but module-style runs (`python -m beamng_rl.*`) do.

BeamNG install path is auto-detected from two hardcoded candidates in `beamng_setup.py`:
- Jango machine: `C:\Users\Jango\BeamNG.tech.v0.38.3.0`
- Student machine: `C:\Users\Student\BeamNG`

Override with `$env:BEAMNG_HOME = "C:\Path\To\BeamNG.tech"` or `--beamng-home` if the path differs.

---

## Common commands

**Smoke tests (no BeamNG required):**
```powershell
python src\beamng_rl\envs\beamng_racing_env.py
python src\beamng_rl\envs\observation_builder.py
python src\beamng_rl\track\query_utils.py
```

**Training (requires BeamNG running or launchable):**
```powershell
python scripts\train\train_live_ppo_run.py --fresh --run-name v1_etk_v2
python scripts\train\train_live_ppo_run.py                                   # resume latest
python scripts\train\train_live_ppo_run.py --fresh --car etkc --reward-config v2
python scripts\train\train_live_ppo_run.py --fresh --car etkc --reward-config v2 --spawn-mode random_checkpoint  # spawn at random centreline point each reset; adds _respawn token to run name

# Parallel second instance (different port + user folder)
python scripts\train\train_live_ppo_run.py --fresh --run-name cond_C --car etkc --reward-config v2 --port 25253 --beamng-user "C:\Users\Jango\AppData\Local\BeamNG_w2" --session-file config\launcher_session_2.json --progress-file logs\remote\current_steps_2.txt --stop-signal-file logs\remote\stop_signal_2.txt
```

**Flask launcher UI (preferred for multi-run management):**
```powershell
python scripts\launcher\app.py
# Open http://localhost:5000
```

**Evaluation:**
```powershell
python scripts\eval\watch_model.py --run v1_etk_2x
python scripts\eval\eval_checkpoints.py --run v1_etk_2x     # → checkpoints_eval.csv
python scripts\eval\log_live_rollout.py --run v1_etk_2x
```

**AI raceline collection (no BeamNG required for physics raceline; BeamNG required for AI collection):**
```powershell
# Regenerate physics racing line (pure Python, ~60s)
python src\beamng_rl\track\physics_raceline.py

# Collect BeamNG CPU AI reference laps
python scripts\eval\collect_ai_raceline.py --vehicles both --aggressions 0.9 1.0
```

**In-sim visualisation:**
```powershell
# Draws centreline (cyan), track limits (red/green), racing line (yellow)
python -m beamng_rl.bootstrap.beamng_bootstrap --draw-path

# Skip individual overlays
python -m beamng_rl.bootstrap.beamng_bootstrap --draw-path --no-raceline
python -m beamng_rl.bootstrap.beamng_bootstrap --draw-path --no-limits
```

**TensorBoard:**
```powershell
tensorboard --logdir logs\runs
```

---

## Architecture

### Data flow per RL step

```
BeamNG.tech (physics)
    ↓ vehicle state dict {pos, vel, heading_rad, damage}
beamng_racing_env.py → _get_vehicle_state()
    ↓
track/query_utils.py → TrackCentreline.query()   [nearest segment, progress, lateral error]
    ↓
envs/observation_builder.py → ObservationBuilder.build()   [12-feature float32 vector]
    ↓
beamng_racing_env.py → _compute_reward()   [scalar reward + reward_info dict]
    ↓
SB3 PPO (MlpPolicy)
```

### Key source files

| File | Owns |
|------|------|
| `src/beamng_rl/envs/beamng_racing_env.py` | Gymnasium env, `REWARD_CONFIGS` registry, reward computation, termination logic, BeamNGpy connection lifecycle |
| `src/beamng_rl/envs/observation_builder.py` | Converts raw BeamNG state dict → normalised 12-feature obs vector |
| `src/beamng_rl/track/query_utils.py` | `TrackCentreline` — nearest-point projection (3D disambiguation), arc-length progress, heading, curvature, lookahead |
| `src/beamng_rl/track/raceline_loader.py` | Parses `items.level.json` DecalRoad nodes; `build_sequential_chain()` stitches road segments into a single loop using hardcoded `HIGHLIGHT_IDS = [30, 12, 10, 17]` |
| `src/beamng_rl/track/geometry.py` | `resample_polyline()` — uniform arc-length resampling with carry-over across segment boundaries |
| `src/beamng_rl/track/physics_raceline.py` | Minimum-curvature racing line via L-BFGS-B; `compute_track_boundaries()`, `find_racing_line()`, `compute_speed_profile()` |
| `src/beamng_rl/track/ai_raceline.py` | Converts BeamNG AI telemetry into raceline JSON exports; lap segmentation + quality gates |
| `src/beamng_rl/bootstrap/beamng_setup.py` | BeamNG home resolution, scenario builders for etkc and sbr, spawn pose constants, `PORT = 25252` default |
| `src/beamng_rl/bootstrap/beamng_bootstrap.py` | Manual debug/visualisation entry point; draws centreline, track limits, and physics racing line in-sim |
| `src/beamng_rl/training/run_manager.py` | Run directory layout, config snapshot, checkpoint discovery, resume logic |
| `src/beamng_rl/training/callbacks.py` | `RewardComponentLogger` (per-component TensorBoard curves), `StopSignalCallback`, `StepProgressWriter` |
| `scripts/train/train_live_ppo_run.py` | Main training entry point — reads session config, creates/resumes runs, calls `model.learn()` |
| `scripts/eval/collect_ai_raceline.py` | Drives BeamNG CPU AI, collects telemetry at 20 Hz, exports static and driven raceline JSONs |
| `scripts/launcher/app.py` | Flask UI backend — slot 1 routes + slot 2 (`/api/2/*`) for parallel run |
| `scripts/launcher/templates/index.html` | Single-page UI — run nav tabs, all controls and monitoring for both slots |

### Run directory layout

```
models/runs/<run_name>/
    run_config.json        # source of truth — vehicle, reward key, timing, PPO config
    run_info.md            # human-readable summary + results appended at completion
    checkpoints/
        ppo_<N>_steps.zip
    model_final.zip
logs/runs/<run_name>/      # TensorBoard event files
logs/runs/<run_name>/rollout_final.csv
logs/runs/<run_name>/checkpoints_eval.csv  (written by eval_checkpoints.py)
```

`run_config.json` is the authoritative record of what a run used. Always check it before resuming or evaluating a run.

### Track data layout

All track data lives in `data/hirochi_track/`:

| File | Contents |
|------|----------|
| `items.level.json` | BeamNG level objects (newline-delimited JSON). `DecalRoad` nodes in `AIDecalWaypointsGroup` with `material="road_invisible"` are the centreline source. Node format: `[x, y, z, width_m]` — 4th element is road width. |
| `race.race.json` | Timing gate / checkpoint geometry (`pathnodes` with `pos`, `radius`) |
| `centreline_resampled_2_0m.json` | 1077 points, 2 m spacing, closed loop. Regenerated after kink fix — do not use pre-fix copies. |
| `physics_raceline.json` | Minimum-curvature racing line, 1068 points, ~69.3 s estimated lap at μ=1.1 |
| `ai_static_raceline_*.json` | BeamNG AI speed profile attached to centreline points (~81 s laps) |
| `ai_driven_raceline_*.json` | Actual AI trajectory resampled at 2 m spacing |

### Reward system

Named presets live in `REWARD_CONFIGS` dict in `beamng_racing_env.py`. Select with `--reward-config v1` (or `v2`). Every component is returned in `reward_info` inside `step()` info dict, and `RewardComponentLogger` writes them all to TensorBoard under `reward/*`.

V1 vs V2 structural difference: V1 has a flat `speed_weight` bonus that partially cancels the curvature overspeed penalty. V2 removes the speed bonus entirely — progress reward implicitly incentivises speed.

V2.1 configs (`v21a`, `v21b`) switch to a **physics curvature target** when `lateral_accel_budget_mps2` is not `None`: `v_target = sqrt(a_lat / max(κ, ε))` replaces the linear formula. The risk penalty is `(max(0, v/v_target − aggression))^power * weight`. V2.1-B adds a braking shortfall term: `max(0, (v²−v_target_80m²)/(2·a_brake) − 80 m) * weight`. All new penalty components appear in TensorBoard under `reward/*` automatically.

### Observation space (12 features)

| Index | Feature | Normalisation |
|-------|---------|---------------|
| 0–2 | forward / lateral / vertical speed | ÷ 60, 20, 10 m/s, clipped |
| 3 | heading error | ÷ π rad |
| 4 | lateral error | ÷ 10 m (+ = left of centreline) |
| 5–7 | lookahead heading errors at 20/40/80 m | ÷ π |
| 8–10 | track curvature at 20/40/80 m ahead | ÷ 0.05 m⁻¹ |
| 11 | progress ratio | [0, 1] |

### Timing profiles

Always use `det50ms` + `speed_factor=16` for training. `legacy_60hz` crashes BeamNG after ~40 episodes.

| Profile | Physics step | Steps per action | Sim time per action |
|---------|-------------|-----------------|-------------------|
| `legacy_60hz` | 1/60 s | 15 | 0.250 s |
| `det50ms` 2× | 0.050 s | 5 | 0.250 s |

### Vehicles

- `"etkc"` — ETK K-Series Trackday A. RWD, high polar moment, progressive oversteer. More forgiving for RL exploration.
- `"sbr"` — SBR4 Track. RWD, low CoG, snap oversteer. Punishes imprecise actions severely; random exploration is destructive.

---

## Experimental conditions

| Condition | Vehicle | Reward | Run name | Steps | Best progress | Status |
|-----------|---------|--------|----------|-------|---------------|--------|
| A | sbr | v1 | `v1_subaru_16x_278353steps` | 278,353 | 1869.6 m | ✅ done |
| B | etkc | v1 | `v1_etk_16x_278670steps` | 278,670 | 860.6 m | ✅ done |
| C | etkc | v21b | — | ~278k | — | ⬜ needs training |
| D | sbr | v21b | — | ~278k | — | ⬜ needs training |

All conditions use `det50ms`, `speed_factor=16`. Condition C must match A/B step budget (~278k) for valid comparison. Full lap length is 2158 m (post kink-fix centreline).

Both A and B were trained with the correct 3D nearest-point projection and post-kink-fix centreline.

---

## Track geometry notes

### Overpass (3D disambiguation)
`TrackCentreline.query()` uses **3D distance for segment disambiguation** but keeps arc-length progress, heading, and lateral error in XY. The Hirochi Raceway overpass sits at ~264 m and ~1262 m progress — segments are <1 m apart in XY but 6.75 m apart in Z.

### Centreline generation and kink removal
`raceline_loader.py`'s `derive_race_path_from_files()` calls `remove_kinks(max_angle_deg=90)` after concatenating DecalRoad segments. This removes direction-reversal artifacts at segment joins — previously a 150° kink at ~1407 m (raw points 72–73) created a phantom r ≈ 0.5 m corner that corrupted curvature features and the speed profile. The fix is applied at source; if you ever regenerate the centreline JSON (e.g. via `beamng_bootstrap --draw-path`), the output will be clean automatically.

### Physics racing line
`src/beamng_rl/track/physics_raceline.py` implements a three-stage pipeline:
1. **`compute_track_boundaries()`** — parses DecalRoad node widths (4th element), interpolates onto centreline arc-lengths, computes per-point left/right boundary XY positions and unit normals.
2. **`find_racing_line()`** — L-BFGS-B minimisation of Σκ² over offset variables α[i] ∈ [−hw, +hw]. Curvature is computed via fully vectorised circumcircle formula. ~500 iterations recommended; `max_iter=1000` is the default.
3. **`compute_speed_profile()`** — forward-backward traction-circle integration: braking limit `a = √((μg)² − a_lat²)`, acceleration limit `min(engine_accel, a_traction)`.

Result at default params (μ=1.1, margin=0.5 m, engine_accel=8 m/s²): **~69.3 s estimated lap**, 119.9 km/h avg, 65.1 km/h min — vs ~81 s for the BeamNG CPU AI reference.

---

## BeamNG connection

- **Slot 1** (default): port `25252`, user folder `%LOCALAPPDATA%\BeamNG`
- **Slot 2** (parallel): port `25253`, user folder `%LOCALAPPDATA%\BeamNG_w2`
- `use_mock=True` (default) runs a toy physics model — safe for import, reward debugging, and CI-style checks without a BeamNG installation.

**One-time slot 2 setup:**
```powershell
New-Item -ItemType Directory -Force "$env:LOCALAPPDATA\BeamNG_w2\BeamNG.tech\current"
Copy-Item -Recurse "$env:LOCALAPPDATA\BeamNG\BeamNG.tech\current\settings" "$env:LOCALAPPDATA\BeamNG_w2\BeamNG.tech\current\settings"
Copy-Item "$env:LOCALAPPDATA\BeamNG\BeamNG.tech.ini" "$env:LOCALAPPDATA\BeamNG_w2\"
```

Override the slot 2 user folder with `$env:BEAMNG_USER_SLOT2` before starting Flask.

---

## Config / session handoff

The Flask launcher writes `config/launcher_session.json` (slot 1) or `config/launcher_session_2.json` (slot 2) before spawning the training script. Keys: `car`, `total_timesteps`, `speed_factor`, `timing_profile`, `reward_config`, `fresh`, `run_name`, `max_damage`, `debug_mode`, `port`, `beamng_user`, `spawn_mode` (`"bootstrap"` | `"random_checkpoint"`).

The training script enforces a vehicle-metadata guard on resume: if `run_config.json` records a different vehicle than the session config requests, training aborts rather than silently cross-contaminating weights.

### Live monitoring files (under `logs/remote/`)

| File | Slot | Purpose |
|------|------|---------|
| `training.pid` | 1 | PID of active training subprocess |
| `current_steps.txt` | 1 | Live step count (written each rollout) |
| `stop_signal.txt` | 1 | Touch to trigger graceful stop |
| `training_2.pid` | 2 | PID of slot 2 subprocess |
| `current_steps_2.txt` | 2 | Live step count for slot 2 |
| `stop_signal_2.txt` | 2 | Stop signal for slot 2 |
| `training_2.log` | 2 | Slot 2 stdout |
