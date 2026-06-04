# Reward Function Design

The reward is a dense proxy for racing performance. The long-term objective is
to complete fast, clean laps, but lap time is too sparse to optimise directly at
every policy step. The environment therefore rewards valid centreline progress
and uses shaping terms to discourage unstable, off-track, or physically
implausible behaviour.

## Base Formula

The V1 reward family is built from these components:

```text
reward =
    progress_delta_m * progress_weight
    + forward_speed_mps * speed_weight
    - abs(heading_error_rad) * heading_error_weight
    - abs(lateral_error_m) * lateral_error_weight
    - off_track_penalty_if_applicable
    - reverse_progress_penalty_if_applicable
    - stuck_penalty_if_applicable
    - progress_jump_penalty_if_applicable
    - abs(action_t - action_t_minus_1).sum() * action_smoothness_weight
    - max(0, forward_speed_mps - target_speed_mps) * curvature_overspeed_weight
```

Later reward variants keep progress as the main objective while replacing or
extending the speed-shaping terms.

## Core Components

| Component | Purpose |
| --- | --- |
| Progress reward | Main dense learning signal; rewards metres advanced around the lap. |
| Speed reward | Small V1-only bonus for useful forward velocity. |
| Heading penalty | Discourages spinning, sideways motion, and driving against the track direction. |
| Lateral penalty | Encourages stable track-following near the centreline. |
| Off-track penalty | Penalises excessive lateral error and likely track exits. |
| Reverse-progress penalty | Penalises movement backwards along the lap. |
| Stuck penalty | Discourages stationary or barely moving behaviour before termination. |
| Progress-jump guard | Filters impossible centreline projection jumps from live simulator data. |
| Smoothness penalty | Penalises rapid steering/throttle changes. |
| Curvature overspeed penalty | Penalises carrying too much speed into upcoming curvature. |

## Named Reward Configurations

The implementation exposes named presets through `REWARD_CONFIGS` in
`src/beamng_rl/envs/beamng_racing_env.py`.

| Config | Main idea |
| --- | --- |
| `v1` | Baseline progress, speed, heading, lateral, smoothness, and linear curvature-overspeed shaping. |
| `v2` | Removes the flat speed bonus, increases smoothness, relaxes centreline pressure, and strengthens curvature overspeed. |
| `v21a` | Replaces the linear curvature target with a physics-shaped traction-budget target. |
| `v21b` | Adds braking-distance anticipation on top of `v21a`. |
| `v22` | Uses a precomputed physics racing line as the lateral reference and speed-profile target. |
| `v22b` | Treats the physics racing line as a soft prior while keeping progress and physics constraints dominant. |

## V1 Baseline

V1 is the original hand-shaped dense reward:

| Parameter | Value |
| --- | ---: |
| `progress_weight` | `1.0` |
| `speed_weight` | `0.05` |
| `heading_error_weight` | `0.2` |
| `lateral_error_weight` | `0.1` |
| `off_track_penalty` | `10.0` |
| `reverse_progress_penalty` | `2.0` |
| `stuck_step_penalty` | `0.05` |
| `action_smoothness_weight` | `0.1` |
| `curvature_overspeed_weight` | `0.15` |

The curvature target is linear:

```text
target_speed_mps = clamp(35.0 - 700.0 * max_curvature_ahead, 12.0, 35.0)
```

V1 is intentionally simple and inspectable, but the speed bonus can conflict
with the overspeed penalty near corner entry.

## V2

V2 keeps the V1 structure but changes four terms:

| Parameter | V1 | V2 |
| --- | ---: | ---: |
| `speed_weight` | `0.05` | `0.0` |
| `action_smoothness_weight` | `0.1` | `0.4` |
| `lateral_error_weight` | `0.1` | `0.05` |
| `curvature_overspeed_weight` | `0.15` | `0.3` |

The design intent is to make speed valuable only when it produces progress,
reduce steering oscillation, and allow more road-width usage than strict
centreline following.

## V2.1-A: Physics-Shaped Curvature Target

`v21a` replaces the hand-tuned linear curvature target with a target derived
from lateral acceleration:

```text
target_speed_mps = sqrt(lateral_accel_budget_mps2 / max(curvature, epsilon))
target_speed_mps = clamp(target_speed_mps, min_corner_speed_mps, max_straight_speed_mps)
risk_ratio = forward_speed_mps / target_speed_mps
traction_penalty = max(0, risk_ratio - allowed_aggression) ** traction_budget_power
                   * traction_budget_weight
```

Default values:

| Parameter | Value |
| --- | ---: |
| `lateral_accel_budget_mps2` | `8.0` |
| `min_corner_speed_mps` | `8.0` |
| `max_straight_speed_mps` | `35.0` |
| `curvature_epsilon` | `0.001` |
| `allowed_aggression` | `1.0` |
| `traction_budget_weight` | `3.0` |
| `traction_budget_power` | `2.0` |

This reframes overspeed as a traction-budget problem rather than a purely
empirical speed penalty.

## V2.1-B: Braking Anticipation

`v21b` adds a braking-distance shortfall term. The policy is penalised when its
current speed is too high to slow down for upcoming curvature within the
lookahead distance:

```text
required_braking_distance_m =
    max(0, current_speed_mps ** 2 - target_speed_mps ** 2)
    / (2 * braking_accel_budget_mps2)

braking_shortfall_m = max(0, required_braking_distance_m - lookahead_distance_m)
braking_penalty = braking_shortfall_m * braking_shortfall_weight
```

Default values:

| Parameter | Value |
| --- | ---: |
| `braking_accel_budget_mps2` | `9.0` |
| `braking_shortfall_weight` | `0.1` |

## V2.2: Racing-Line Reward

`v22` introduces `data/hirochi_track/physics_raceline.json`. It removes the
centreline lateral penalty and instead applies:

| Parameter | Value |
| --- | ---: |
| `raceline_lateral_weight` | `0.10` |
| `raceline_overspeed_weight` | `0.18` |
| `raceline_speed_lookahead_m` | `5.0` |

This version is useful for testing a stronger racing-line prior, but it can
overconstrain behaviour if the policy optimises for matching the line rather
than making fast, stable progress.

## V2.2-B: Soft Racing-Line Prior

`v22b` keeps the physics racing line available but treats it as guidance rather
than a hard target. It combines the V2.1-B traction and braking terms with a
weaker, gated, and decaying racing-line term.

Key settings:

| Parameter | Value |
| --- | ---: |
| `lateral_error_weight` | `0.03` |
| `raceline_lateral_weight` | `0.06` |
| `raceline_overspeed_weight` | `0.03` |
| `raceline_lateral_corridor_m` | `1.25` |
| `raceline_overspeed_margin_mps` | `4.0` |
| `raceline_speed_gate_min_mps` | `8.0` |
| `raceline_speed_gate_full_mps` | `22.0` |
| `raceline_max_heading_error_rad` | `0.45` |
| `raceline_weight_decay_steps` | `120000` |
| `raceline_weight_final_scale` | `0.25` |
| `raceline_slow_speed_penalty_weight` | `0.08` |
| `raceline_baseline_speed_bonus_weight` | `0.06` |
| `raceline_baseline_speed_margin_mps` | `1.0` |

The line-distance term is ignored inside the corridor, fades in only at useful
speeds and headings, and decays over training. This keeps the optimisation
target focused on quick, stable progress rather than imitation.

## Progress Origin

The live BeamNG spawn is near the painted start/finish marker, but the raw
closed-loop centreline progress at that pose may lie near the wrap boundary
rather than exactly zero. Training therefore uses episode-relative progress and
progress deltas instead of assuming that raw `progress_ratio` starts at zero.

## Known Limitations

- The centreline is not necessarily the optimal racing line.
- Lateral error penalties can discourage wider out-in-out racing lines if set
  too high.
- Collision and wall-contact signals depend on available BeamNG telemetry.
- Dense reward shaping is easier to learn from than sparse lap time, but it is
  still a proxy for the final racing objective.
- Best-checkpoint evaluation is usually more informative than final-checkpoint
  evaluation because PPO performance can be non-monotonic.
