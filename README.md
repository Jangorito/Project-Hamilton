# Project Hamilton

BeamNG reinforcement-learning experiments and bootstrap utilities.

## Setup

Run these commands from the repository root in PowerShell.

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If the virtual environment was created as `venv` instead of `.venv`, activate it with:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

The execution-policy command only affects the current PowerShell session.

## Launching The Bootstrap

Because this project currently uses a `src` layout without an installable package file, set `PYTHONPATH` before launching:

```powershell
$env:PYTHONPATH = "$PWD\src"
```

Launch the current BeamNG bootstrap:

```powershell
python -m beamng_rl.bootstrap.beamng_bootstrap
```

Launch the Student-machine shadow-disabling smoke test:

```powershell
python -m beamng_rl.bootstrap.beamng_bootstrap --disable-shadows
```

Launch the Student-machine editable low-graphics preset:

```powershell
python -m beamng_rl.bootstrap.beamng_bootstrap --low-graphics
```

Launch it with the derived Hirochi race path drawn in BeamNG:

```powershell
python -m beamng_rl.bootstrap.beamng_bootstrap --draw-path
```

Useful variants:

```powershell
python -m beamng_rl.bootstrap.beamng_bootstrap --draw-path --resample-spacing 0
python -m beamng_rl.bootstrap.beamng_bootstrap --draw-path --sphere-every 10
```

## 'Student' Machine

The bootstrap automatically checks this BeamNG install path:

```text
C:\Users\Student\BeamNG
```

Expected launch flow:

```powershell
cd C:\Users\Student\Workspace\Project-Hamilton
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "$PWD\src"
python -m beamng_rl.bootstrap.beamng_bootstrap --draw-path
```

To test BeamNGpy graphics preferences without starting an RL training run, launch
the bootstrap with shadows disabled:

```powershell
python -m beamng_rl.bootstrap.beamng_bootstrap --disable-shadows
```

This currently only attempts `$pref::Shadows::disable = 2` through BeamNGpy on
the Student install. The bootstrap uses BeamNG's graphics settings API key
`GraphicDisableShadows`, which the local BeamNG Lua maps to
`$pref::Shadows::disable`; if BeamNGpy rejects it, the command prints a warning
and continues.

To apply a broader editable low-graphics preset:

```powershell
python -m beamng_rl.bootstrap.beamng_bootstrap --low-graphics
```

The preset is stored in:

```text
config\beamng_low_graphics.ini
```

That file is intentionally commented and ordered by expected hardware impact.
Edit values there, then relaunch with `--low-graphics`. To test a different
file without replacing the default:

```powershell
python -m beamng_rl.bootstrap.beamng_bootstrap --low-graphics --low-graphics-settings-file "config\beamng_low_graphics.ini"
```

The default track-data directory on this machine is:

```text
C:\Users\Student\Workspace\Project-Hamilton\data\hirochi_track
```

## Jango Machine

The bootstrap automatically checks this BeamNG.tech install path:

```text
C:\Users\Jango\BeamNG.tech.v0.38.3.0
```

Expected launch flow:

```powershell
cd C:\Users\Jango\workspace\BeamNG
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "$PWD\src"
python -m beamng_rl.bootstrap.beamng_bootstrap --draw-path
```

The default track-data directory on this machine is:

```text
C:\Users\Jango\workspace\BeamNG\data\hirochi_track
```

## Path Overrides

If BeamNG or the track data live somewhere else, set environment variables for the current PowerShell session:

```powershell
$env:BEAMNG_HOME = "C:\Path\To\BeamNG.tech"
$env:BEAMNG_TRACK_DATA_DIR = "$PWD\data\hirochi_track"
python -m beamng_rl.bootstrap.beamng_bootstrap --draw-path
```

You can also pass the paths directly:

```powershell
python -m beamng_rl.bootstrap.beamng_bootstrap --beamng-home "C:\Path\To\BeamNG.tech" --track-data-dir "$PWD\data\hirochi_track" --draw-path
```
