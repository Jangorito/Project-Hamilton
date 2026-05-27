# Project Hamilton — Context Brief for Research

## What This Is

A reinforcement learning agent learning to race on **Hirochi Raceway** in **BeamNG.tech** (a soft-body physics driving simulator). The agent drives a Subaru BRZ (SBR4) using PPO. The goal is eventually to complete a clean, fast lap.

---

## The Stack

| Layer | Technology |
|---|---|
| Simulator | BeamNG.tech v0.38.3 |
| Sim API | BeamNGpy (Python) |
| RL library | Stable Baselines3 (PPO) |
| Env interface | Gymnasium (continuous control) |
| Training machine | Windows 11, dedicated GPU, SSH access |

---

## Environment

### Action Space
2D continuous, both axes in `[-1, 1]`:
- `action[0]` — steering (left/right)
- `action[1]` — throttle/brake combined: positive = throttle, negative = brake

Throttle and brake are mutually exclusive (one axis). This reduces exploration complexity for the first baseline. Future work could split them into separate dimensions.

### Observation Space
12-feature vector, all `float32`. Features 0–10 are in `[-1, 1]`, feature 11 (progress) is in `[0, 1]`.

| Index | Feature | Normalisation |
|---|---|---|
| 0 | `forward_speed_norm` | ÷ 60 m/s, clipped |
| 1 | `lateral_speed_norm` | ÷ 20 m/s, clipped |
| 2 | `vertical_speed_norm` | ÷ 10 m/s, clipped |
| 3 | `heading_error_norm` | ÷ π rad, clipped |
| 4 | `lateral_error_norm` | ÷ 10 m, clipped (+ = left of centreline) |
| 5 | `lookahead_heading_error_20m_norm` | angle to centreline point 20 m ahead, ÷ π |
| 6 | `lookahead_heading_error_40m_norm` | same at 40 m |
| 7 | `lookahead_heading_error_80m_norm` | same at 80 m |
| 8 | `curvature_20m_norm` | track curvature 20 m ahead, ÷ 0.05 m⁻¹ |
| 9 | `curvature_40m_norm` | same at 40 m |
| 10 | `curvature_80m_norm` | same at 80 m |
| 11 | `progress_ratio` | fraction of lap completed [0, 1] |

All speeds are **vehicle-frame** (forward = along heading, lateral = perpendicular). Lookahead heading errors give the agent near/mid/far track curvature context for planning corner entry.

### Episode Termination
| Condition | Type |
|---|---|
| Lateral error > 10 m | `terminated` |
| No forward progress for 100 consecutive steps | `terminated` (stuck) |
| Car near wall + no progress for 30 steps | `terminated` (wall bash) |
| Damage > 500 units (live mode only) | `terminated` |
| Steps > 500 (`max_episode_steps`) | `truncated` |
| Lap complete | `terminated` — **not yet implemented** |

---

## Reward Function

Every component is logged per-step so reward attribution is fully inspectable. The reward is:

```
reward = progress_reward
       + speed_reward
       - heading_penalty
       - lateral_penalty
       - off_track_penalty
       - reverse_progress_penalty
       - stuck_step_penalty
       - progress_jump_penalty
       - smoothness_penalty
       - curvature_overspeed_penalty
```

### Component Details

| Component | Formula | Default weight |
|---|---|---|
| **progress_reward** | `progress_delta_m × 1.0` | Main signal — metres advanced along centreline |
| **speed_reward** | `forward_speed_mps × 0.05` | Small bonus for going fast |
| **heading_penalty** | `\|heading_error_rad\| × 0.2` | Penalises pointing away from track direction |
| **lateral_penalty** | `\|lateral_error_m\| × 0.1` | Penalises drifting off centreline |
| **off_track_penalty** | `10.0` flat when `\|lateral_error\| > 10 m` | Hard hit for leaving track |
| **reverse_progress_penalty** | `2.0` flat when `progress_delta < 0` | Discourages reversing |
| **stuck_step_penalty** | `0.05` per step when `progress_delta < 0.05 m` | Discourages stalling |
| **progress_jump_penalty** | `5.0` flat when delta > 50 m (teleport artifact) | Filters bogus progress |
| **smoothness_penalty** | `\|action_t - action_{t-1}\|₁ × 0.1` | Penalises jerky inputs per dimension |
| **curvature_overspeed_penalty** | `max(0, speed - target_speed) × 0.15` | Penalises carrying too much speed into corners |

### Curvature Speed Target
```
target_speed = clamp(35.0 - 700.0 × max_curvature_ahead, 12.0, 35.0)
```
`max_curvature_ahead` is the maximum of curvatures at 20 m, 40 m, 80 m ahead. On a straight (curvature ≈ 0) the target is 35 m/s (~126 km/h). On a tight hairpin (curvature ≈ 0.033 m⁻¹) it drops toward 12 m/s (~43 km/h).

---

## PPO Configuration

```python
n_steps       = 256       # steps per environment before each update
batch_size    = 64
gamma         = 0.99
learning_rate = 3e-4
policy        = MlpPolicy (SB3 default MLP)
```

Simulation is run at **60 Hz deterministic** with `steps_per_action = 15`, so each policy step advances the sim by 15 physics frames (~0.25 s of real car time).

---

## Track

**Hirochi Raceway** — a medium-complexity circuit in BeamNG.  
Lap length: ~2158 m. Centreline is a resampled 2 m–spaced polyline derived from the BeamNG `.race.json` and `items.level.json` files. The centreline JSON lives at `data/hirochi_track/centreline_resampled_2_0m.json`.

---

## Current Training State

- Active run: **v2** (or similar — check `models/runs/` for the latest)
- Heuristic baseline (random/simple controller): **~965 m** per episode before termination
- Full lap target: **2158 m**
- Approximate training pace: ~27k timesteps / 2 hours on the Jango machine

TensorBoard logs are under `logs/runs/<run_name>/`.

---

## Known Reward Issues / Research Questions

Things worth investigating for reward function improvements:

1. **Lap completion is not yet rewarded.** There is a `TODO` in the code — completing the loop is currently not detected or rewarded. The agent has no incentive to finish laps over maximising per-step progress.

2. **Speed reward vs. curvature penalty balance.** The flat `speed_weight = 0.05` always incentivises going faster, which conflicts with the curvature overspeed penalty. Is there a cleaner way to shape speed without the two terms partially cancelling?

3. **Progress as the main signal has no concept of lap time.** The agent learns to go forward, not to go fast-forward. A time-based or speed-weighted progress signal might produce faster lap times.

4. **Lookahead features exist but aren't exploited in the reward.** The observation includes 20/40/80 m curvature and heading errors, giving the agent the information to brake early for corners. Whether the current reward structure exploits this well is an open question.

5. **Action smoothness weight.** Currently `0.1` applied as L1 across both dimensions. Is per-dimension weighting better (e.g. heavier on steering than throttle)?

6. **Dense vs. sparse reward tradeoffs.** All current reward components are dense. Literature on racing RL sometimes uses sparse lap-time rewards with curriculum (starting from easy segments) — worth researching.

---

## File Map (relevant to reward/obs research)

```
src/beamng_rl/
  envs/
    beamng_racing_env.py      ← RewardConfig, _compute_reward, _check_termination
    observation_builder.py    ← 12-feature obs, ObservationConfig, lookahead logic
  track/
    query_utils.py            ← TrackCentreline, centreline projection, curvature_at
scripts/
  train_live_ppo_run.py       ← PPO_CONFIG, ENV_CONFIG, training loop
data/hirochi_track/
  centreline_resampled_2_0m.json  ← the track geometry
```
