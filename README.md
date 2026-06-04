# Project Hamilton

Project Hamilton is a BeamNG.tech reinforcement-learning project for training
continuous-control racing agents on Hirochi Raceway. The codebase wraps BeamNG
with a Gymnasium-style environment, trains PPO policies with Stable-Baselines3,
and includes utilities for track processing, training, evaluation, and
visualisation.

## Project Layout

```text
src/beamng_rl/
  bootstrap/       BeamNG launch and setup helpers
  envs/            Gymnasium environment and observation builder
  track/           Centreline, racing-line, and geometry utilities
  training/        Run management and training callbacks
  visualisation/   Debug drawing and HUD helpers

scripts/
  env/             Environment checks and timing benchmarks
  eval/            Rollout, checkpoint evaluation, and baseline collection
  launcher/        Local Flask launcher for training runs
  monitor/         Metric export helpers
  train/           PPO training entry points
  viz/             Plotting and raceline visualisation

data/hirochi_track/
  Track centreline and raceline reference data used by the environment.
```

## Requirements

- Windows with BeamNG.tech installed
- Python 3.10 or newer
- A BeamNG.tech version compatible with `beamngpy==1.35.1`

BeamNG.tech itself is not included in this repository. Set `BEAMNG_HOME` if the
bootstrap cannot find your installation automatically.

## Setup

Run these commands from the repository root in PowerShell:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If you prefer a virtual environment named `venv`, use:

```powershell
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

This project uses a `src` layout without an installable package file, so set
`PYTHONPATH` before module-style commands:

```powershell
$env:PYTHONPATH = "$PWD\src"
```

Optional path overrides:

```powershell
$env:BEAMNG_HOME = "C:\Path\To\BeamNG.tech"
$env:BEAMNG_TRACK_DATA_DIR = "$PWD\data\hirochi_track"
```

## Bootstrap And Visualisation

Launch BeamNG with the project bootstrap:

```powershell
python -m beamng_rl.bootstrap.beamng_bootstrap
```

Draw the derived Hirochi path and optional overlays in BeamNG:

```powershell
python -m beamng_rl.bootstrap.beamng_bootstrap --draw-path
```

Apply the editable low-graphics preset:

```powershell
python -m beamng_rl.bootstrap.beamng_bootstrap --low-graphics
```

The preset lives in `config/beamng_low_graphics.ini` and can be replaced at
runtime:

```powershell
python -m beamng_rl.bootstrap.beamng_bootstrap --low-graphics --low-graphics-settings-file "config\beamng_low_graphics.ini"
```

## Training

Start a fresh PPO run:

```powershell
python scripts\train\train_live_ppo_run.py --fresh --run-name example_run
```

Resume a named run:

```powershell
python scripts\train\train_live_ppo_run.py --run-name example_run
```

Useful options include:

```powershell
python scripts\train\train_live_ppo_run.py --fresh --run-name example_etk --car etkc --reward-config v1
python scripts\train\train_live_ppo_run.py --fresh --run-name example_sbr --car sbr --reward-config v21b --obs-config v2
python scripts\train\train_live_ppo_run.py --fresh --run-name example_respawn --spawn-mode random_checkpoint
```

The Flask launcher provides a local UI for starting and monitoring runs:

```powershell
python scripts\launcher\app.py
```

Then open `http://localhost:5000`.

## Evaluation

Watch a trained model:

```powershell
python scripts\eval\watch_model.py --run example_run
```

Evaluate all checkpoints for a run:

```powershell
python scripts\eval\eval_checkpoints.py --run example_run
```

Log one live rollout to CSV:

```powershell
python scripts\eval\log_live_rollout.py --run example_run
```

Export TensorBoard metrics:

```powershell
python scripts\monitor\export_metrics.py --run example_run
```

## Reward And Observation Documentation

- [FeaturesDescriptions.md](FeaturesDescriptions.md) describes the observation
  vectors used by the policy.
- [docs/RewardFunction.md](docs/RewardFunction.md) describes the reward
  components and named reward configurations.

## Generated Artefacts

Training runs produce logs, TensorBoard events, checkpoints, rollout CSVs, and
model archives. These artefacts can be large and machine-specific, so they are
not intended to be committed wholesale. For public reporting, prefer selected
figures, summary CSVs, and the exact run configuration needed to reproduce a
result.
