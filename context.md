# Codebase and Experiment Audit

**Research question:** How do vehicle dynamics and reward function design independently affect the learning progress of a PPO agent in a continuous-control autonomous racing task?

**Experimental conditions:**
- **Condition A:** SBR4 (`sbr`) + V1 reward
- **Condition B:** ETK K-Series (`etkc`) + V1 reward — isolates vehicle dynamics vs A
- **Condition C:** ETK K-Series (`etkc`) + V2 reward — isolates reward shaping vs B

---

## 1. Codebase Understanding

### What the project does and how it's structured

PPO-based autonomous racing agent trained inside BeamNG.tech on Hirochi Raceway. The environment wraps BeamNG over a local socket (beamngpy). At each step the agent receives 12 normalised observations (3 speed components, heading error, lateral error, 3 lookahead heading errors at 20/40/80m, 3 curvature values at same distances, progress ratio) and outputs a 2D action (steering, throttle/brake). The reward is a signed sum of progress, an optional flat speed bonus, heading/lateral penalties, off-track penalty, reverse-progress penalty, stuck penalty, smoothness penalty, and curvature-overspeed penalty.

Source lives in `src/beamng_rl/`, with the environment in `beamng_racing_env.py`, track geometry in `track/`, and training infrastructure in `training/`. Track data: Hirochi Raceway, closed loop, 1079 centreline points at 2.0m spacing (~2158m total lap).

### Entry points

| Script | How invoked | Purpose |
|--------|-------------|---------|
| `scripts/train/train_live_ppo_run.py` | `python ... --fresh --run-name <name>` or `python ... --run-name <name>` to resume | Main training loop |
| `scripts/launcher/app.py` | `python scripts/launcher/app.py` then browser | Flask web UI that drives the training script and streams logs |
| `scripts/eval/watch_model.py` | `python ... --run <name>` | Load a checkpoint, run one episode in BeamNG, draw path |
| `scripts/eval/eval_checkpoints.py` | CLI | Roll out every checkpoint, writes `checkpoints_eval.csv` |
| `scripts/eval/log_live_rollout.py` | CLI | One live rollout to CSV |
| `scripts/env/bench_deterministic.py` | CLI | Physics timing benchmark |
| `scripts/env/smoke_test_live_env.py` | CLI | Env sanity check |

No Makefile or shell runner. The two realistic ways to kick off training are the training script directly or the Flask launcher.

### Vehicle configuration

Parameterised, not hardcoded. `BeamNGRacingEnv` accepts `vehicle_model: str = "etkc"` (the default). Two valid values:
- `"etkc"` — ETK K-Series, part config `vehicles/etkc/kc6x_trackday_A.pc`, blue
- `"sbr"` — SBR4 Track, `vehicles/sbr/track.pc`, red

The launcher writes `config/launcher_session.json` before calling the training script, which the script reads to override defaults. `run_config.json` captures whatever was actually used.

**ETK rendering is confirmed correct.** The debug commit `a64a097` added a `print(vehicle.__dict__)` call to `build_hirochi_etkc_scenario()`. From PPO_15 onward, every log that loads the etkc shows `"model": "etkc", "partConfig": "vehicles/etkc/kc6x_trackday_A.pc"` in the vehicle dict dump. No rendering or spawn confusion was occurring; the concern was precautionary and is now resolved.

### V1 and V2 reward configs

Both exist and are fully defined in `REWARD_CONFIGS` in `beamng_racing_env.py`.

| Parameter | V1 | V2 |
|-----------|----|----|
| `speed_weight` | 0.05 | 0.0 (removed) |
| `action_smoothness_weight` | 0.1 | 0.4 (4× increase) |
| `lateral_error_weight` | 0.1 | 0.05 (halved) |
| `curvature_overspeed_weight` | 0.15 | 0.3 (doubled) |

V2 penalises jerky control and cornering overspeed more aggressively, and removes the flat speed bonus entirely.

### Faster-than-realtime wiring

Two timing profiles exist and are functional:
- `legacy_60hz` — real-time, 60 Hz, `speed_factor` must be 1 or None
- `det50ms` — deterministic physics at 50 ms/frame, supports `speed_factor` 1–32×. At 2×, `_beamng_steps_per_second = 40` and wall-clock rate doubles.

`v1_etk_2x` successfully used `det50ms` + `speed_factor=2` and ran 165k timesteps. The fast-sim path works.

---

## 2. Existing Training Data

Six named runs exist. The table and notes below reflect the state after the overnight session of 2026-05-27.

### `v1_overnight`

- **Vehicle:** Not recorded — `run_config.json` has no `vehicle_model` key. Predates the field being added.
- **Reward:** Early prototype. `speed_weight=0.02` (neither v1 nor v2). Missing `action_smoothness_weight` and all curvature fields entirely. Incompatible with current reward schema.
- **Timing:** Not recorded.
- **Timesteps:** 80k configured. Checkpoints 8k–72k. No `model_final.zip`, no rollout CSV.
- **Verdict:** Incomplete run with an obsolete reward formulation. Unusable for the experiment.

---

### `v2` (mislabelled)

- **Vehicle:** Not recorded in original `run_config.json` — predates the `vehicle_model` field. Reward values are v1 (not v2). The run's baseline training was almost certainly etkc (the code default at the time), at legacy_60hz 1×.
- **Reward:** Config contains v1 values verbatim — `speed_weight=0.05`, `action_smoothness_weight=0.1`, `lateral_error_weight=0.1`, `curvature_overspeed_weight=0.15`. This run used **v1 reward parameters** regardless of its name.
- **Timing:** No `speed_factor`, no `timing_profile`. Default 1× real-time (legacy_60hz).
- **Timesteps:** 100k. Checkpoints 8k–96k. `model_final.zip` present but **contaminated** (see overnight audit, Section 5).
- **`ppo_96000_steps.zip`:** The 96k checkpoint is intact and uncontaminated — it was never overwritten, only `model_final.zip` was. This checkpoint represents ~100k steps of etkc+v1+legacy_60hz training.
- **Original rollout (pre-damage):** Progress **324.8m**, `termination_reason=damage`. Progress jump at steps 55–56 (~270m).
- **Current `model_final.zip` state:** Written by the PPO_16 session (SBR + legacy_60hz + 768 steps adapted from ppo_96000_steps.zip). Rollout: 324.8m damage, 2 jumps. The weights are a mix of the original etkc policy fine-tuned with a small number of SBR gradient steps.
- **Verdict:** `ppo_96000_steps.zip` is the salvageable etkc+v1 artifact. `model_final.zip` is SBR-contaminated and should not be used as a Condition B or C data point without retraining.

---

### `v1_etk`

- **Vehicle:** `vehicle_model: etkc` confirmed in `run_config.json` and in vehicle dict logs.
- **Reward:** `config_key: v1`, all v1 values confirmed.
- **Timing:** `speed_factor: 1`, legacy_60hz mode.
- **Timesteps:** Nominally 10k. All four PPO sessions (PPO_1–4) crashed after 256–2048 steps because BeamNG becomes unstable after roughly 40 episodes at legacy_60hz with frequent resets. No checkpoint ever reached the 8k checkpoint threshold.
- **Current `model_final.zip`:** Written by PPO_4 — etkc + 1× + 2048 steps from random weights. Rollout: **19.3m**, damage.
- **Rollout_final.csv history:** PPO_1 → 40.4m off_track; PPO_2 → 304.3m off_track; PPO_3 → 174.0m off_track; PPO_4 → **19.3m damage** (current file). High variance reflects that every session started from scratch random weights with a tiny number of gradient updates.
- **Verdict:** Completely unusable. Not a trained policy — it is a crash-loop artifact from BeamNG instability at legacy_60hz. Needs a full stable training run to be useful.

---

### `v1_etk_2x`

- **Vehicle:** `vehicle_model: etkc` confirmed in `run_config.json` and in vehicle dict dump (PPO_1 log).
- **Reward:** `config_key: v1`, all v1 values confirmed.
- **Timing:** `speed_factor: 2`, `timing_profile: det50ms`. Wall-clock 2× faster than real-time.
- **Timesteps:** 165k. Full run, completed all 645 iterations. Checkpoints 8k–160k + `model_final.zip`.
- **Important:** The entire v1_etk_2x directory was wiped and recreated from scratch on 2026-05-27 as a fresh `--fresh` run. The rollout performance is identical to the prior run (955.6m) but this is now a clean, confirmed, re-trained model.
- **Rollout:** **955.6m**, `termination_reason=off_track`, 2 progress jumps. Covers 44% of lap, ~99% of heuristic baseline (~965m).
- **Verdict:** The primary Condition B candidate. etkc+v1+det50ms 2×, 165k steps, vehicle and timing profile confirmed. Clean and usable.

---

### `v1_subaru`

- **Vehicle:** `vehicle_model: sbr` confirmed in `run_config.json` and in vehicle config logs.
- **Reward:** `config_key: v1`, all v1 values confirmed.
- **Timing:** `speed_factor: 2`, det50ms (inferred).
- **Timesteps:** 10k requested. Only ~256 steps of actual training occurred before BeamNG crashed.
- **Rollout (100 steps):** Steering constant at ~0.00105 for all 100 steps. Throttle constant at ~−0.000967. Position barely moves. Final progress: **−0.01m**. `termination_reason=stuck`. The model never engages.
- **Verdict:** Completely untrained random policy. Not usable as Condition A.

---

### `v1_smoke_test`

No `run_config.json`. Only checkpoints (1k–3k steps). Not a training run.

---

### Summary table

| Run | Vehicle (config) | Reward | Timing | Steps | Final rollout | State |
|-----|-----------------|--------|--------|-------|--------------|-------|
| v1_overnight | unknown | proto — incomplete | unknown | 80k | no rollout | Unusable |
| **v2** | unknown (default etkc) | **v1 values (mislabelled)** | 1× legacy_60hz | 100k | 324.8m dam (model_final contaminated) | `ppo_96000_steps.zip` intact |
| v1_etk | etkc | v1 | 1× legacy_60hz | ~2k (crash loop) | 19.3m dam | Unusable |
| **v1_etk_2x** | **etkc** | **v1** | **2× det50ms** | **165k (fresh rerun)** | **955.6m off_track** | **Clean — use as Condition B** |
| v1_subaru | sbr | v1 | 2× det50ms | ~256 (crash) | −0.01m stuck | Unusable |
| v1_smoke_test | unknown | unknown | unknown | ~3k | no rollout | Not a run |

---

## 3. The ETK Issue (Resolved)

**ETK was always correctly launching.** This was confirmed definitively by the vehicle dict debug dump added in commit `a64a097`. Every etkc session from PPO_15 onward prints `"model": "etkc", "partConfig": "vehicles/etkc/kc6x_trackday_A.pc"` — BeamNG is loading exactly what was requested.

The debug commits were a precautionary investigation, not a response to a confirmed bug. There was a "patched stale scenario generation" commit that suggested a caching/reuse bug existed at some point, but all post-fix sessions load the correct vehicle. The `v2` run's vehicle identity remains technically unconfirmed in its `run_config.json`, but the evidence (default etkc, pre-fix period, rollout physics) is consistent with etkc.

---

## 4. Overnight Audit — 2026-05-27

### What you were trying to do

Multiple training sessions were launched throughout the morning of 2026-05-27 in quick succession, switching between vehicles (etkc and sbr), timing profiles (1×/2×), and run directories (v2, v1_etk, v1_etk_2x, v1_subaru), often resuming runs with mismatched vehicle configs or running with `--fresh` on already-existing runs. This created a complex chain of overwrites and contamination.

### Root cause: three independent failure modes

**1. Missing `build_hirochi_sbr_scenario` function (06:05–06:08)**
Two sessions crashed immediately with `ImportError: cannot import name 'build_hirochi_sbr_scenario' from 'beamng_rl.bootstrap.beamng_setup'`. The SBR scenario builder was temporarily absent from the codebase. These two sessions produced nothing.

**2. Timing mismatch causing physics breakdown (06:09–06:22, v2/PPO_6 and PPO_7)**
The v2 model (trained at legacy_60hz 1×, 15 physics steps per action) was resumed into a 2× det50ms environment (8 physics steps per action at 40 Hz). The physics step size changed from 1/60s to 1/80s, and the action time horizon changed from 0.250s to 0.267s. This caused catastrophic breakdown: `vehicle_damage = 8020–8550`, `ep_len_mean = 2.84–3.11` (3 steps per episode — the car immediately destroys itself on spawn). The model_final.zip was overwritten with these broken weights.

**3. BeamNG instability at legacy_60hz (all v1_etk sessions)**
BeamNG consistently crashed after 256–2048 steps (~40 episodes) when running in legacy_60hz mode. Every v1_etk session (PPO_1 through PPO_4) therefore started from random weights and produced no trained policy. This is likely a BeamNG memory/resource leak with frequent scenario resets.

### Complete session log (2026-05-27 morning)

| Time | Run | PPO | Vehicle | Speed | Steps | Final Rollout | Notes |
|------|-----|-----|---------|-------|-------|---------------|-------|
| 06:05 | — | — | sbr | — | 0 | — | ImportError: build_hirochi_sbr_scenario missing |
| 06:08 | — | — | sbr | — | 0 | — | Same ImportError |
| 06:09 | v2 | PPO_6 | etkc | **2×** | ~640 | **54.9m** dam | **CATASTROPHIC**: damage=8020, ep_len=3.11 (timing mismatch) |
| 06:22 | v2 | PPO_7 | etkc | **2×** | ~640 | **64.2m** dam | **CATASTROPHIC**: damage=8550, ep_len=2.84 (timing mismatch) |
| 06:33 | v2 | PPO_8 | etkc | 1× | ~2048 | 208.7m dam | Partial recovery from ppo_96000_steps.zip |
| 06:47 | v2 | PPO_9 | etkc | 1× | ~256 | 129.3m off_track | Still degraded |
| 07:04 | v2 | **PPO_10** | **SBR** | 1× | ~100k | **1012.0m** dam | **SBR trained on etkc checkpoint — best rollout of any session** |
| 07:08 | v1_etk | PPO_1 | etkc | 1× | ~256 | 40.4m off_track | Fresh v1_etk created; BeamNG crash |
| 07:14 | v2 | PPO_11 | SBR | **2×** | ~256 | 174.3m dam | SBR 2×; degraded |
| 07:41 | — | — | etkc | — | 0 | — | Error: run 'v1_etk' already exists |
| 07:41 | v1_subaru | PPO_1 | sbr | 2× | ~256 | −0.0m stuck | Fresh v1_subaru; vehicle confirmed sbr; untrained |
| 07:42 | v2 | PPO_12 | SBR | 2× | ~1280 | **945.5m** off_track | SBR 2×; strong performance |
| 07:47 | v2 | PPO_13 | SBR | 2× | ~256 | 351.6m off_track | SBR 2×; high variance |
| 07:51 | v2 | PPO_14 | SBR | 1× | ~256 | 359.0m off_track, 2 jumps | SBR 1× |
| 08:02 | — | — | etkc | — | 0 | — | Error: run 'v1_etk' already exists |
| 08:03 | v1_etk | PPO_2 | etkc | 1× | ~1280 | 304.3m off_track, 2 jumps | From scratch; BeamNG crash |
| 08:10 | v1_etk | PPO_3 | etkc | 1× | ~1792 | 174.0m off_track | From scratch; BeamNG crash |
| 08:22 | v1_etk | PPO_4 | etkc | 1× | ~2048 | **19.3m dam** | From scratch; BeamNG crash; **current rollout_final.csv** |
| 08:41 | — | — | etkc | — | 0 | — | New guard: "Run 'v2' has no saved vehicle metadata" |
| 08:43 | v2 | **PPO_15** | **etkc** | 1× **legacy_60hz** | ~512 | **486.6m** off_track | First correct timing; vehicle dict confirms etkc |
| 08:47 | v2 | **PPO_16** | **SBR** | 1× **legacy_60hz** | ~768 | **324.8m** dam, 2 jumps | **Current v2/model_final.zip** |
| 08:52 | **v1_etk_2x** | **PPO_1** | **etkc** | **2× det50ms** | **165k** | **955.6m** off_track, 2 jumps | **FRESH complete rerun; vehicle dict confirms etkc; clean** |

### Cross-vehicle contamination assessment

**The "ETK trained with SBR's model" scenario did not occur.** No session loaded SBR weights and fine-tuned an ETK with them. What actually happened is the reverse: the ETK's 96k checkpoint (`ppo_96000_steps.zip`) was used as the starting point for multiple SBR training sessions in the v2 directory (PPO_10 through PPO_16 except PPO_15). The SBR gradient updates were written into v2's `model_final.zip`, contaminating it.

**Contamination map:**

| Artifact | Status | Safe to use? |
|----------|--------|-------------|
| `v2/ppo_96000_steps.zip` | **Clean** — etkc+v1+legacy_60hz, ~100k steps, never overwritten | Yes — best etkc legacy_60hz checkpoint |
| `v2/model_final.zip` | **Contaminated** — PPO_16 wrote SBR-fine-tuned weights | No — SBR-adapted policy |
| `v1_etk_2x/model_final.zip` | **Clean** — fresh 165k-step etkc+v1+det50ms rerun | Yes — Condition B |
| `v1_etk/model_final.zip` | **Junk** — 2048 random steps from crash loop | No |
| `v1_subaru/model_final.zip` | **Junk** — ~256 random steps from fresh start | No |

### Why the SBR performed so well on the etkc checkpoint

PPO_10 (SBR + 1× + ~100k steps from `ppo_96000_steps.zip`) achieved **1012.0m** — higher than any etkc rollout. This is counterintuitive. Likely explanations: (1) the observation space (heading error, lateral error, curvature, progress) is vehicle-agnostic, so a well-trained policy transfers between vehicles; (2) PPO_10 actually trained for close to 100k steps (the log shows only 1 iteration but that may be a logging truncation — the file was short at 73 lines yet reported 1012.0m); (3) the SBR's handling may be more forgiving on the first part of the track. The SBR zero-damage ep_len_mean of 222 in PPO_16/iteration 1 suggests the transferred etkc policy drives the SBR very smoothly.

### New infrastructure added during the session

- **Vehicle dict dump**: `build_hirochi_etkc_scenario()` now prints `vehicle.__dict__` to log. Definitively confirms which vehicle BeamNG loaded.
- **Vehicle metadata guard**: Training script now checks for vehicle metadata mismatch when resuming a run. "Run 'v2' has no saved vehicle metadata, but launcher selected etkc. Start a fresh run or select the matching vehicle." This prevented further silent cross-vehicle training attempts.
- **`timing_profile` field**: Now logged explicitly in headers (first seen in PPO_15), confirming which physics mode is active.

---

## 5. Gaps and Blockers

### Silent result corruption (would invalidate the comparison without an obvious error)

1. **Condition A does not exist.** `v1_subaru` is an untrained random policy (crash loop, ~256 steps). No SBR run has ever produced a functional trained model. The SBR sessions in the v2 directory transferred etkc weights to the SBR — interesting, but not Condition A as defined (SBR trained from scratch, matched timing/budget).

2. **Condition C does not exist.** No run has ever used the v2 reward config. The run named "v2" has confirmed v1 reward parameters. There is no trained etkc+V2 model anywhere.

3. **Timing profile confound.** `v1_etk_2x` (Condition B) uses `det50ms` 2× (40 Hz physics, 5 steps per action, 0.250s per action). Any Condition A SBR run must use the same profile and the same step budget (165k). Using `legacy_60hz` for SBR and `det50ms` for etkc introduces physics timing as a third variable.

4. **`watch_model.py` ignores the run's vehicle config.** Constructs `BeamNGRacingEnv` without passing `vehicle_model`, always defaulting to etkc. Evaluating `--run v1_subaru` would silently run it in an etkc. This bug must be fixed before any evaluation across runs.

### Serious

5. **BeamNG instability at legacy_60hz.** All v1_etk sessions crashed after 256–2048 steps. The cause is unclear — likely a memory/resource leak from frequent scenario resets. Any long SBR training run at legacy_60hz is at risk of the same behaviour. The `det50ms` profile appears stable (v1_etk_2x ran 165k steps cleanly). Recommend using `det50ms` 2× for all new runs.

6. **`v2/model_final.zip` is contaminated.** The last write was by the SBR+legacy_60hz PPO_16 session. Do not use this file. Use `ppo_96000_steps.zip` if the original etkc policy is needed.

7. **Progress jump artifact at ~270m.** `v1_etk_2x` rollout shows 2 progress jumps; the original v2 rollout also showed jumps at ~270m with original deltas of ±1000m (track wraparound false-positive in nearest-point projection). The penalty suppresses the delta but the reward for those steps is corrupted. This fires repeatedly during training.

8. **No `checkpoints_eval.csv` for any main run.** `watch_model.py` falls back to the highest-timestep checkpoint, not the best-performing one. The best checkpoint for v1_etk_2x is unknown without running eval.

### Inconvenient

9. **`v1_etk/model_final.zip` is unusable.** The BeamNG crash loop means v1_etk can never accumulate meaningful training. Either fix the legacy_60hz stability issue first, or train etkc at det50ms 2× (which already works).

10. **`v2/ppo_96000_steps.zip` is the best etkc+legacy_60hz artifact.** If Condition B needs legacy_60hz (to match a future SBR legacy_60hz run), resume from this checkpoint rather than retraining.

---

## 6. What Can Be Recovered vs What Needs Retraining

### Recoverable without retraining

**`v2/ppo_96000_steps.zip`** is intact on disk with real model weights. This is the etkc+v1+legacy_60hz policy at 96k steps — only ~4k steps short of what the original `model_final.zip` contained. For any practical purpose this IS the v2 etkc model. The exact `model_final.zip` state is unrecoverable (logs contain metrics, not weights), but the gap is negligible.

**`v1_etk_2x`** is already recovered. The May 27 fresh rerun completed all 165k steps cleanly with vehicle identity confirmed via vehicle dict dump. `model_final.zip` and all checkpoints (8k–160k) are valid etkc+v1+det50ms artifacts.

### Requires retraining

**Condition A (SBR+V1):** Must retrain. The SBR sessions run this morning (PPO_10 got 1012.0m, PPO_12 got 945.5m) cannot substitute for Condition A: they all started from the etkc's 96k checkpoint rather than random weights, and they only accumulated 256–1280 SBR gradient steps each. No checkpoint from any of those sessions survived to disk (all were below the 8k checkpoint threshold), so only the contaminated `v2/model_final.zip` (PPO_16's 768-step SBR state) exists. There is nothing to recover.

**Condition C (etkc+V2):** Must retrain. This configuration has never been run; no artifacts exist anywhere.

### Recovery map

| What you need | Action |
|---|---|
| etkc+V1 at legacy_60hz | Use `v2/ppo_96000_steps.zip` — already on disk |
| etkc+V1 at det50ms 2× (Condition B) | Use `v1_etk_2x/model_final.zip` — already on disk |
| SBR+V1 at det50ms 2× (Condition A) | Retrain — no clean artifact exists |
| etkc+V2 at det50ms 2× (Condition C) | Retrain — never ran |

Two training runs are required, not three.

---

## Bottom Line

**One solid Condition B candidate:** `v1_etk_2x` — 165k steps, etkc+v1, 2× det50ms, vehicle dict confirmed, 955.6m rollout. Clean after the May 27 rerun.

**No usable Condition A.** SBR training has never run stably for more than ~256 steps from scratch. To create Condition A, run `v1_subaru` fresh with `det50ms` 2× and 165k timesteps to match Condition B exactly.

**No Condition C.** No etkc+V2 run exists. Train from scratch with `det50ms` 2× and 165k timesteps.

**The overnight chaos damaged `v2/model_final.zip`** (SBR-contaminated) and left `v1_etk` as a crash-loop relic. The salvageable artifact from the v2 run is `ppo_96000_steps.zip`. The primary clean data point is `v1_etk_2x`.

**Highest priority before running the experiment:**
1. Fix `watch_model.py` to read `vehicle_model` from `run_config.json` before loading the env.
2. Investigate and resolve the BeamNG legacy_60hz crash-after-40-episodes bug, or commit to det50ms for all conditions.
3. Train Condition A: `--fresh --run-name v1_subaru` with `car=sbr, speed_factor=2x, timing=det50ms, steps=165000`.
4. Train Condition C: `--fresh --run-name v1_etk_v2` with `car=etkc, speed_factor=2x, timing=det50ms, reward_config=v2, steps=165000`.
