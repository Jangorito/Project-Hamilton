# V2.1 Proposal: Physics-Informed Racing Reward Design

Status: aspirational research proposal  
Date: 2026-05-28  
Project: PPO autonomous racing in BeamNG.tech on Hirochi Raceway

## Executive Summary

This proposal discards the previous code-level "V2" as the conceptual next
step. The old V2 was useful as an interim hand-tuned reward variant, but it is
not the theory this project should now build around.

The proposed V2.1 starts from a stronger premise:

> A racing policy should maximise lap progress while managing a finite traction
> budget, choosing a line that increases effective corner radius, braking early
> enough for upcoming curvature, and moving smoothly between acceleration,
> braking, and turning.

This reframes reward design from "encourage speed and punish bad states" into:

```text
racing performance
    = progress maximisation
    + traction-budget management
    + curvature-aware speed choice
    + braking-distance anticipation
    + road-width and racing-line usage
    + smooth control transitions
    + robustness under controlled variation
```

The proposal combines two sources of insight:

1. BeamNG's AI parameters, especially `racerAggression`, `racerSkill`, and
   `racerRandomnessScale`.
2. Brian Beckman's *The Physics of Racing*, especially the ideas of traction
   budget, racing line, braking distance, and smoothness.

The core research question becomes:

> Can a PPO racing agent learn more interpretable and transferable racing
> behaviour when reward design is grounded in traction-budget physics rather
> than hand-shaped speed and centreline penalties?

## What Is Being Rejected

The current repository contains a reward config named `v2`. That config makes
several reasonable edits to V1:

- removes the flat speed reward,
- increases action smoothness penalty,
- reduces centreline lateral penalty,
- increases curvature overspeed penalty.

Those changes point in a good direction, but this proposal does not treat that
config as the official V2. It should be considered superseded.

The reason is not that the old config is bad. The reason is that it is still
mostly a weight-tuning intervention. V2.1 should instead be a physics-informed
reward design whose terms have clear behavioural meanings.

In plain terms:

```text
Old V2:
    "Tweak the existing dense reward so it is less reckless."

Proposed V2.1:
    "Model racing as progress under a traction budget, with line choice,
    braking anticipation, and smooth budget transitions."
```

The project may still borrow lessons from the old config, especially the
instinct to remove a global speed subsidy. But the old config should not define
the proposal's structure.

## Literature And Simulator Anchors

### BeamNG AI Anchor

BeamNG's AI exposes several knobs that are unusually useful for theory-building:

- `racerAggression` acts like a traction or speed-risk tolerance.
- `racerSkill` decomposes into road-margin and line-shaping behaviour.
- `racerRandomnessScale` acts like a built-in randomisation control.

These are not RL rewards, but they are a simulator-native decomposition of
driving behaviour. They suggest that "driving better" is not one scalar. It is a
combination of risk tolerance, road-width usage, line quality, and robustness.

### Beckman Anchor

Brian Beckman's *The Physics of Racing* series provides the physics vocabulary
for the new reward proposal.

Useful references:

- [Physics of Racing series index](https://www.miata.net/sport/Physics/)
- [Part 5: Introduction to the Racing Line](https://www.miata.net/sport/Physics/05-Cornering.html)
- [Part 7: The Traction Budget](https://www.miata.net/sport/Physics/07-Circle.html)
- [Readable compiled mirror](https://studylib.net/doc/25193633/the-physics-of-car-racing---brian-beckam)

This proposal uses Beckman in five specific ways:

1. Traction budget / traction circle.
2. Physics-shaped curvature speed.
3. Road width and racing line as effective-radius improvement.
4. Braking distance scaling with speed squared.
5. Smoothness as physically meaningful movement through the grip envelope.

## The Five Beckman Findings For V2.1

### 1. Traction Budget Is The Core Abstraction

Beckman's traction-budget framing says that tyres have a finite ability to
generate acceleration. That budget can be spent on:

- forward acceleration,
- braking,
- lateral cornering,
- or combinations of these.

This is exactly the missing theoretical object in the current reward. The old
reward separately reasons about speed, heading, lateral error, and smoothness.
V2.1 should reason about how the car is using its grip budget.

The key quantity is not just speed. It is combined demand:

```text
traction_demand = sqrt(lateral_demand^2 + longitudinal_demand^2)
```

The reward should penalise demand that exceeds an allowed budget:

```text
traction_excess = max(0, traction_utilisation - allowed_aggression)
```

This makes `allowed_aggression` a direct conceptual analogue of BeamNG's
`racerAggression`.

### 2. Curvature Speed Should Be Physics-Shaped

The current reward uses a hand-shaped linear target:

```text
target_speed = base - scale * curvature
```

That is useful but physically crude.

For a path with curvature `kappa`, lateral acceleration demand is approximately:

```text
a_lat = v^2 * kappa
```

Solving for speed gives:

```text
v_target = sqrt(a_lat_budget / max(kappa, epsilon))
```

This is much more satisfying than a linear curvature penalty because it encodes
the actual relationship between speed, curvature, and lateral acceleration.

V2.1 should therefore move toward:

```text
curvature_target_speed_mps =
    sqrt(lateral_accel_budget_mps2 / max(max_curvature_ahead, epsilon))
```

with practical caps:

```text
curvature_target_speed_mps =
    clamp(raw_physics_target, min_corner_speed, max_straight_speed)
```

This keeps the reward numerically stable while making the theory honest.

### 3. Road Width Matters Because It Increases Effective Radius

Beckman's racing-line discussion supports the move away from centreline worship.
A racing line is not valuable merely because an expert drove it. It is valuable
because it often increases the effective radius of a corner.

Larger radius means lower curvature:

```text
kappa = 1 / radius
```

Lower curvature means higher feasible speed for the same lateral acceleration
budget:

```text
v_target = sqrt(a_lat_budget / kappa)
```

This turns the racing-line idea into physics:

```text
better line -> larger effective radius -> lower curvature demand -> higher
safe speed -> faster progress
```

That is a much stronger justification than:

```text
stay near the AI line because the AI line is good
```

V2.1 should treat road-width usage as a way to reshape the path, not merely as
lateral error management.

### 4. Braking Distance Scales With Speed Squared

Beckman's braking discussion highlights a crucial fact: braking distance grows
with the square of speed.

For a simple constant-deceleration approximation:

```text
d_brake = (v_current^2 - v_target^2) / (2 * a_brake_budget)
```

This means corner-entry reward should not only ask:

```text
Am I too fast for the curvature I am currently on?
```

It should also ask:

```text
Can I slow down enough before the upcoming high-curvature segment?
```

This is especially relevant because the current observation already contains
lookahead heading and curvature features at 20 m, 40 m, and 80 m. V2.1 can use
those same ideas in the reward:

```text
required_brake_distance_m =
    (current_speed^2 - upcoming_target_speed^2) / (2 * brake_budget)

braking_shortfall_m =
    max(0, required_brake_distance_m - distance_to_upcoming_corner_m)
```

This gives the agent a dense signal for braking too late, not merely for being
too fast after it has already arrived at the corner.

### 5. Smoothness Is Physical, Not Cosmetic

The existing reward treats smoothness as an action-delta penalty:

```text
smoothness_penalty = abs(action_t - action_t_minus_1)
```

That is useful, but Beckman's framing suggests a deeper interpretation.
Smoothness matters because the driver is moving around inside the traction
budget. Abruptly jumping from braking to turning to acceleration can exceed the
available grip even when each individual command is locally plausible.

V2.1 should eventually measure smoothness in traction-budget space:

```text
traction_vector_t = [longitudinal_demand_norm, lateral_demand_norm]

budget_motion_penalty =
    norm(traction_vector_t - traction_vector_t_minus_1)
```

This makes smoothness about controlled transitions through the grip envelope,
not merely about small joystick changes.

## BeamNG AI Findings For V2.1

### `racerAggression`: Risk Budget

BeamNG's `racerAggression` maps naturally to:

```text
allowed_aggression
```

In V2.1, aggression should mean:

```text
How close to the traction boundary is the policy allowed or encouraged to run?
```

Possible interpretation:

| Value | Meaning |
| ---: | --- |
| 0.3 | everyday driving, far inside the budget |
| 0.7 | spirited but conservative racing |
| 1.0 | near the available traction boundary |
| >1.0 | deliberately permissive, useful only if the model is conservative |

This can become a named experimental axis instead of being hidden inside reward
weights.

### `racerSkill`: Line And Margin Decomposition

BeamNG's skill parameter appears to affect separate values such as road-edge
margin and line-shaping force. The important lesson is:

```text
skill != speed
skill != centreline proximity
skill = margin choice + line quality + anticipation
```

V2.1 should therefore avoid treating `abs(lateral_error_m)` as the main measure
of spatial skill.

Instead, spatial behaviour should be decomposed:

```text
road_boundary_safety: stay within usable track
edge_margin_policy: choose how much margin to preserve
line_quality: choose a path with favourable curvature
heading_alignment: point along the intended path
```

Centreline error can remain as a temporary fallback, but it should not be the
theoretical foundation.

### `racerRandomnessScale`: Later Robustness

For the core experiments, randomness should be set to zero. The first goal is
clean comparison.

Later, `racerRandomnessScale` becomes useful as an analogue of domain
randomisation:

```text
deterministic expert baseline
-> stochastic expert baseline
-> robustness evaluation for learned policies
```

This belongs after the deterministic V2.1 experiments, not before them.

## V2.1 Core Thesis

V2.1 should be described as:

> A physics-informed reward family that trains an agent to maximise progress
> while respecting a configurable traction budget, planning speed from
> curvature, anticipating braking distance, using road width to improve
> effective radius, and transitioning smoothly through the grip envelope.

This gives the proposal a clear causal chain:

```text
track geometry defines curvature
curvature defines lateral acceleration demand
lateral and longitudinal demand consume traction budget
road-width usage can reduce effective curvature
braking distance determines whether future curvature is reachable safely
smooth control keeps transitions inside the budget
progress remains the objective
```

## Aspirational V2.1 Reward Model

The proposed reward is:

```text
reward =
    progress_reward
    + lap_completion_reward
    - traction_budget_penalty
    - braking_shortfall_penalty
    - line_quality_penalty
    - road_margin_penalty
    - heading_alignment_penalty
    - budget_smoothness_penalty
    - failure_penalties
```

Not every term must be implemented immediately. The point is that every term
has a theoretical role.

### Progress Reward

Progress remains the primary objective:

```text
progress_reward = progress_delta_m * progress_weight
```

Rationale:

- dense,
- interpretable,
- directly related to lap completion,
- avoids needing sparse lap-time reward at the beginning.

Important change:

```text
No global speed reward is needed.
```

Speed is rewarded indirectly because faster useful driving produces more
progress per step. Unsafe speed is constrained by the traction and braking
terms.

### Lap Completion Reward

Lap completion should eventually be explicit:

```text
lap_completed = episode_progress_m >= track.total_lap_length
```

Possible reward:

```text
lap_completion_reward = lap_completion_bonus if lap_completed else 0
```

This is not the main learning signal, but it marks the actual task objective.

### Curvature-Based Lateral Demand

The geometry-only estimate of lateral demand is:

```text
a_lat_required = forward_speed_mps^2 * path_curvature
```

For early versions, `path_curvature` can be:

```text
max_curvature_ahead = max(curvature_20m, curvature_40m, curvature_80m)
```

For later versions, it should come from the intended path:

```text
centreline curvature
AI raceline curvature
learned candidate-line curvature
road-width-aware effective curvature
```

### Physics-Shaped Target Speed

The raw target speed is:

```text
raw_target_speed_mps =
    sqrt(lateral_accel_budget_mps2 / max(max_curvature_ahead, epsilon))
```

The practical target speed is:

```text
target_speed_mps =
    clamp(raw_target_speed_mps, min_corner_speed_mps, max_straight_speed_mps)
```

This replaces the old linear hand-shaped target as the conceptual default.

### Traction Budget Penalty

The aspirational traction utilisation is:

```text
lateral_utilisation =
    abs(lateral_accel_mps2) / lateral_accel_budget_mps2

longitudinal_utilisation =
    abs(longitudinal_accel_mps2) / longitudinal_accel_budget_mps2

traction_utilisation =
    sqrt(lateral_utilisation^2 + longitudinal_utilisation^2)
```

Then:

```text
traction_excess =
    max(0, traction_utilisation - allowed_aggression)

traction_budget_penalty =
    traction_excess^2 * traction_budget_weight
```

If full acceleration telemetry is unavailable, V2.1 can begin with a proxy:

```text
lateral_accel_proxy = forward_speed_mps^2 * max_curvature_ahead
longitudinal_accel_proxy = (forward_speed_mps - prev_forward_speed_mps) / dt
```

If even longitudinal acceleration is too noisy, begin with a lateral-only
version:

```text
lateral_risk_ratio =
    forward_speed_mps / target_speed_mps
```

This is still more physics-grounded than the old linear overspeed penalty.

### Braking Shortfall Penalty

For upcoming curvature, compute the speed the agent should reach before the
corner:

```text
upcoming_target_speed_mps =
    sqrt(lateral_accel_budget_mps2 / max(upcoming_curvature, epsilon))
```

Then estimate required braking distance:

```text
required_braking_distance_m =
    max(0, current_speed^2 - upcoming_target_speed^2)
    / (2 * braking_accel_budget_mps2)
```

Compare it with distance to the upcoming curvature event:

```text
braking_shortfall_m =
    max(0, required_braking_distance_m - distance_to_corner_m)
```

Penalty:

```text
braking_shortfall_penalty =
    braking_shortfall_m * braking_shortfall_weight
```

This term is important because it rewards anticipation. It can teach the policy
that the correct braking point is before the corner, not at the point where the
overspeed penalty becomes obvious.

### Line Quality Penalty

The minimal version uses AI raceline proximity:

```text
raceline_distance_excess =
    max(0, abs(raceline_lateral_error_m) - raceline_tolerance_m)

line_quality_penalty =
    raceline_distance_excess * raceline_proximity_weight
```

The stronger version uses heading alignment to the intended path:

```text
line_heading_penalty =
    abs(raceline_heading_error_rad) * raceline_heading_weight
```

The aspirational version measures whether the chosen path improves effective
radius:

```text
line_quality = reduction_in_effective_curvature
```

This is harder to implement, but theoretically cleaner. The AI raceline should
be treated as a practical proxy for a path with better curvature properties, not
as a sacred rail.

### Road Margin Penalty

The road-margin term should eventually replace centreline lateral error as the
main spatial safety term.

```text
nearest_edge_margin_m =
    min(left_edge_margin_m, right_edge_margin_m)

margin_shortfall_m =
    max(0, desired_edge_margin_m - nearest_edge_margin_m)

road_margin_penalty =
    margin_shortfall_m^2 * road_margin_weight
```

This maps directly to the BeamNG `racerSkill` interpretation:

```text
more conservative skill setting -> larger desired margin
more racing-oriented setting -> smaller desired margin
```

It also allows the car to use track width without being punished for leaving
the centreline.

### Heading Alignment Penalty

Heading should align with the intended path, not necessarily the centreline:

```text
heading_alignment_penalty =
    abs(path_heading_error_rad) * heading_alignment_weight
```

Candidate path hierarchy:

1. AI raceline heading if raceline is available.
2. Centreline heading if not.
3. Road-width-aware generated path in a later version.

### Budget Smoothness Penalty

Action smoothness is the fallback:

```text
action_smoothness_penalty =
    abs(steer_t - steer_t_minus_1) * steering_smoothness_weight
    + abs(pedal_t - pedal_t_minus_1) * pedal_smoothness_weight
```

The aspirational version is budget smoothness:

```text
traction_vector_t =
    [longitudinal_utilisation_t, lateral_utilisation_t]

budget_smoothness_penalty =
    norm(traction_vector_t - traction_vector_t_minus_1)
    * budget_smoothness_weight
```

This reflects the physical reason smoothness matters.

### Failure Penalties

Failure penalties remain necessary:

```text
off_track_penalty
damage_penalty
reverse_progress_penalty
stuck_penalty
progress_jump_penalty
wall_contact_penalty
```

But these should be treated as guardrails, not the main theory.

## Minimum V2.1, Rich V2.1, Full V2.1

Because this is aspirational, V2.1 should be a family with implementation
levels. The proposal should start from the theory, then evaluate feasibility.

### V2.1-A: Geometry-Only Traction Proxy

This is the likely minimum viable implementation.

Uses:

- progress reward,
- physics-shaped curvature target,
- lateral risk ratio,
- no flat speed reward,
- action smoothness,
- existing centreline fallback,
- existing failure penalties.

Core formulas:

```text
target_speed = sqrt(a_lat_budget / max(max_curvature_ahead, epsilon))
risk_ratio = forward_speed / target_speed
risk_penalty = max(0, risk_ratio - allowed_aggression)^2 * weight
```

Why it is attractive:

- uses existing speed and curvature data,
- does not require tyre telemetry,
- directly improves the old curvature overspeed term,
- makes `allowed_aggression` explicit.

### V2.1-B: Braking-Aware Geometry Reward

Adds:

- braking-distance lookahead,
- upcoming corner target speed,
- dense penalty for braking too late.

Core formula:

```text
d_brake = (v_current^2 - v_target_next^2) / (2 * a_brake_budget)
shortfall = max(0, d_brake - distance_to_corner)
```

Why it matters:

- uses Beckman's speed-squared braking insight,
- teaches anticipation,
- may solve late-corner failures more directly than overspeed penalty alone.

### V2.1-C: Racing-Line And Road-Width Reward

Adds:

- AI raceline object,
- raceline distance,
- raceline heading,
- raceline curvature,
- eventual road-edge margin.

Purpose:

- stop treating centreline proximity as racing skill,
- encourage path choices that reduce effective curvature,
- compare learned behaviour to BeamNG AI line behaviour.

Important caution:

The racing line should be a soft prior, not a hard rail.

### V2.1-D: Full Traction-Budget Reward

Adds actual vehicle dynamics telemetry if accessible:

- longitudinal acceleration,
- lateral acceleration,
- yaw rate,
- slip angle,
- wheel slip,
- ABS or traction-control activity,
- damage or contact impulses.

Core formula:

```text
traction_utilisation =
    sqrt(
        (a_lat / a_lat_budget)^2
        + (a_long / a_long_budget)^2
    )
```

This is the closest version to Beckman's traction-circle framing.

Why it is aspirational:

- requires telemetry plumbing,
- needs careful normalisation,
- may become vehicle-specific,
- but could be more feasible in BeamNG than it would be in a simpler simulator.

## Proposed RewardConfig Fields

V2.1 should introduce named concepts rather than only weights.

```text
# Progress
progress_weight
lap_completion_bonus

# Aggression / traction budget
allowed_aggression
lateral_accel_budget_mps2
longitudinal_accel_budget_mps2
braking_accel_budget_mps2
traction_budget_weight
traction_budget_power

# Curvature speed
min_corner_speed_mps
max_straight_speed_mps
curvature_epsilon
curvature_lookahead_distances_m

# Braking lookahead
braking_shortfall_weight
braking_lookahead_distances_m
corner_curvature_threshold
braking_safety_margin_m

# Racing line
raceline_proximity_weight
raceline_heading_weight
raceline_tolerance_m
raceline_curvature_weight

# Road margin
road_margin_weight
desired_edge_margin_m
edge_margin_power

# Smoothness
steering_smoothness_weight
pedal_smoothness_weight
budget_smoothness_weight

# Failure
off_track_penalty
damage_penalty
reverse_progress_penalty
stuck_penalty
progress_jump_penalty
wall_contact_penalty
```

These fields are more numerous than the current reward needs, but they make the
reward interpretable. Feasibility can decide which subset becomes the first
implemented V2.1 run.

## Observation And Telemetry Roadmap

### Already Available

The current observation vector already gives a useful foundation:

- forward speed,
- lateral speed,
- vertical speed,
- heading error,
- lateral centreline error,
- lookahead heading errors,
- lookahead curvature,
- progress ratio.

This is enough for V2.1-A and probably V2.1-B.

### High-Value Additions

For V2.1-C:

```text
raceline_lateral_error_norm
raceline_heading_error_norm
raceline_curvature_20m_norm
raceline_curvature_40m_norm
raceline_curvature_80m_norm
```

For road-margin modelling:

```text
left_edge_margin_norm
right_edge_margin_norm
nearest_edge_margin_norm
road_width_norm
```

For V2.1-D:

```text
longitudinal_accel_norm
lateral_accel_norm
yaw_rate_norm
slip_angle_norm
wheel_slip_norm
```

### Important Design Principle

Reward terms and observation terms should be separated conceptually.

It is acceptable for the reward to use geometry that is not yet in the
observation, but if a term becomes central to behaviour, the agent should
eventually be able to observe the relevant state.

For example:

- if braking shortfall becomes important, the agent needs enough lookahead to
  anticipate it,
- if raceline reward becomes important, the agent should observe raceline
  error,
- if traction utilisation becomes important, the agent should observe some
  proxy for vehicle state near the grip limit.

## Evaluation Metrics

The V2.1 proposal should be evaluated behaviourally, not just by total reward.

### Primary Task Metrics

```text
max_episode_progress_m
lap_completed
lap_time_s
sector_times_s
termination_reason
damage
best_checkpoint_step
```

### Traction And Risk Metrics

```text
target_speed_mps
lateral_accel_required_mps2
lateral_risk_ratio
longitudinal_accel_mps2
traction_utilisation
traction_excess
steps_over_aggression_budget
mean_traction_budget_penalty
max_traction_utilisation
```

### Braking Metrics

```text
upcoming_target_speed_mps
required_braking_distance_m
distance_to_corner_m
braking_shortfall_m
steps_with_braking_shortfall
speed_at_corner_entry_mps
```

### Line And Road-Width Metrics

```text
mean_abs_centreline_error_m
mean_abs_raceline_error_m
mean_raceline_heading_error_rad
raceline_curvature_vs_centreline_curvature
nearest_edge_margin_min_m
mean_nearest_edge_margin_m
road_width_used_m
```

### Smoothness Metrics

```text
mean_abs_steering_delta
mean_abs_pedal_delta
steering_oscillation_index
budget_vector_delta
mean_budget_smoothness_penalty
```

### Learning Metrics

```text
area_under_progress_curve
time_to_300m
time_to_500m
time_to_900m
time_to_first_lap_completion
checkpoint_performance_variance
```

### Transfer Metrics

Because vehicle dynamics are part of the project, include:

```text
source_vehicle
target_vehicle
zero_shot_progress_m
fine_tuned_progress_m
steps_to_recover_progress
failure_mode_shift
traction_metric_shift
```

## BeamNG AI Baseline Plan

BeamNG CPU AI should be used as a behavioural baseline, not only a performance
ceiling.

Recommended deterministic baselines:

| Baseline | Aggression | Skill | Randomness | Purpose |
| --- | ---: | ---: | ---: | --- |
| Everyday AI | 0.3 | 1.0 | 0.0 | Conservative traction use |
| Limit AI | 1.0 | 1.0 | 0.0 | Near-limit racing reference |
| Lower-skill AI | 1.0 | lower | 0.0 | Margin and line-quality contrast |

Metrics to collect:

```text
lap_time_s
sector_times_s
mean_speed_mps
target_speed_mps
traction_utilisation_proxy
braking_shortfall_m
raceline_error_m
edge_margin_m
damage/contact events
```

This lets the learned policy be described in a richer way:

- high aggression but poor braking,
- safe but centreline-bound,
- smooth but slow,
- good road-width usage but unstable,
- raceline-like but too dependent on one vehicle,
- fast because it uses radius, not because it drives recklessly.

## What Condition C Becomes

The original backbone was:

| Condition | Vehicle | Reward | Purpose |
| --- | --- | --- | --- |
| A | SBR4 | V1 | Vehicle-dynamics comparison |
| B | ETK K-Series | V1 | Baseline |
| C | ETK K-Series | V2 | Reward-design comparison |

Under this proposal, Condition C should not use the old code-level `v2`
configuration.

Condition C becomes:

```text
ETK K-Series + V2.1 selected implementation + det50ms + matched step budget
```

The selected implementation may be V2.1-A, V2.1-B, V2.1-C, or V2.1-D depending
on feasibility. The thesis should state clearly which version was implemented.

Possible naming:

```text
v21_etk_geometry
v21_etk_braking
v21_etk_raceline
v21_etk_traction
```

Avoid naming it `v2` if that would confuse it with the old code config.

## Research Hypotheses

### H1: Physics-Shaped Curvature Speed Improves Interpretability

Replacing a linear curvature target with:

```text
v_target = sqrt(a_lat_budget / curvature)
```

should make the reward easier to justify and easier to diagnose.

Expected evidence:

- risk metrics explain failure points,
- target speeds vary plausibly by corner,
- overspeed penalties align with observed off-track events.

### H2: Braking Lookahead Improves Corner Entry

Braking-distance shortfall should penalise late braking before the car reaches
the corner.

Expected evidence:

- earlier braking before high-curvature segments,
- lower speed at corner entry,
- fewer off-track failures after long straights,
- improved progress past known failure zones.

### H3: Removing Global Speed Reward Does Not Remove Speed Incentive

If progress remains the primary reward, useful speed is still rewarded.

Expected evidence:

- similar or better progress per step,
- lower crash rate,
- lower traction excess,
- speed concentrated where geometry allows it.

### H4: Road-Width-Aware Line Reward Beats Centreline Reward

The agent should eventually learn better behaviour from road margin and
effective-radius cues than from centreline distance.

Expected evidence:

- lower raceline error,
- greater but controlled road-width usage,
- better speed through corners,
- fewer unnecessary centreline corrections.

### H5: Budget Smoothness Is More Meaningful Than Action Smoothness

Penalising abrupt changes in traction demand should produce smoother driving
than penalising raw action deltas alone.

Expected evidence:

- lower budget-vector delta,
- less steering oscillation,
- fewer grip-loss failures,
- smoother braking-to-turn-in transitions.

### H6: Aggression Becomes A Tunable Behavioural Axis

Varying `allowed_aggression` should produce predictable behavioural changes.

Expected evidence:

- lower aggression gives safer but slower policies,
- higher aggression gives faster but riskier policies,
- learned behaviour can be compared to BeamNG AI aggression levels.

## Risks And Caveats

### The Traction Circle Is An Approximation

Real tyres do not produce a perfect circular grip envelope. Load transfer,
tyre temperature, surface, suspension, and combined slip all matter.

But the traction circle is still useful as a first-order behavioural model. The
goal is not perfect vehicle dynamics. The goal is a better reward prior than a
flat speed bonus and centreline penalty.

### Curvature From Centreline May Misrepresent The Chosen Path

If the agent uses road width, centreline curvature may overestimate or
underestimate the actual path curvature.

Mitigation:

- use raceline curvature when available,
- log centreline and raceline risk separately,
- treat road-width modelling as a future improvement.

### Braking-Distance Targets Depend On Brake Budget

The chosen braking acceleration budget may be wrong for a vehicle, surface, or
damage threshold.

Mitigation:

- estimate from BeamNG AI telemetry,
- estimate from human or heuristic rollouts,
- sweep several conservative values,
- log shortfall instead of trusting it blindly.

### Racing-Line Reward Can Overconstrain

An AI line may not be optimal for the RL policy, vehicle, or reward objective.

Mitigation:

- use tolerance bands,
- keep line weights low,
- start with logging before reward,
- compare against no-line V2.1 variants.

### Too Many Terms Can Obscure Interpretation

V2.1 is aspirational, but experiments must remain legible.

Mitigation:

- implement one conceptual layer at a time,
- use version names such as V2.1-A and V2.1-B,
- log every reward component,
- compare against the same ETK + V1 baseline.

## Feasibility Questions For Implementation

These are the questions to answer when moving from proposal to code:

1. Can BeamNG provide reliable lateral and longitudinal acceleration directly?
2. If not, is finite-difference speed stable enough for longitudinal
   acceleration?
3. Can road edges or road width be extracted reliably from Hirochi data?
4. Is the AI raceline geometrically clean and correctly aligned to the track?
5. Which lookahead distance best predicts braking failures: 20 m, 40 m, 80 m,
   or a longer search for upcoming curvature?
6. What lateral acceleration budget corresponds to ETK and SBR driving near the
   limit?
7. Can BeamNG AI baselines provide empirical aggression and braking references?
8. Should the first V2.1 run use only geometry, or include raceline terms too?

## Suggested Implementation Order

This is not a commitment to feasibility. It is the cleanest order if the pieces
are available.

### Step 1: Add Diagnostics Without Changing Training

Log:

```text
physics_target_speed_mps
lateral_accel_required_mps2
lateral_risk_ratio
required_braking_distance_m
braking_shortfall_m
```

Use existing V1 and ETK rollouts first.

Purpose:

- sanity-check the physics quantities,
- see whether they explain current failure points,
- avoid training on broken metrics.

### Step 2: Implement V2.1-A

Train geometry-only risk reward:

```text
progress
- physics-shaped lateral risk
- action smoothness
- existing failure penalties
```

Purpose:

- replace old curvature overspeed with a physics-shaped target,
- keep implementation manageable,
- establish a clean reward-design comparison.

### Step 3: Implement V2.1-B

Add braking shortfall.

Purpose:

- test whether anticipation improves corner entry,
- use Beckman's braking-distance insight directly.

### Step 4: Add AI Raceline Metrics

Before rewarding the raceline:

- extract the AI line,
- validate visually,
- log raceline deviation for V1 and V2.1 rollouts.

Purpose:

- determine whether current policies already discover line-like behaviour,
- avoid overcommitting to a flawed reference.

### Step 5: Implement V2.1-C

Add soft raceline and road-width terms.

Purpose:

- move beyond centreline reward,
- test the effective-radius theory.

### Step 6: Explore Full Traction Telemetry

If BeamNG exposes useful vehicle dynamics:

- lateral acceleration,
- longitudinal acceleration,
- yaw rate,
- slip angle,
- wheel slip.

Purpose:

- move from geometry proxy to actual traction utilisation.

## Thesis Framing

The strongest thesis narrative is:

1. V1 establishes that dense progress-based PPO can learn partial racing
   behaviour.
2. BeamNG's AI suggests that driving behaviour decomposes into aggression,
   skill, and randomness rather than one vague "quality" dimension.
3. Beckman's racing physics provides a principled interpretation of those
   dimensions: aggression maps to traction-budget use; skill maps to line
   choice, road margin, and anticipation; smoothness maps to controlled motion
   through the grip envelope.
4. V2.1 therefore replaces hand-shaped speed reward with a physics-informed
   reward family based on traction budget, curvature speed, braking distance,
   road-width usage, and smooth transitions.
5. The experiment asks whether this reward produces more interpretable,
   stable, and transferable racing behaviour than V1 under matched vehicle and
   simulator conditions.

Possible dissertation paragraph:

```text
The proposed V2.1 reward was motivated by the observation that autonomous
racing performance is not adequately represented by a global speed bonus or a
centreline-following penalty. BeamNG's native AI exposes separate controls for
aggression, skill, and randomness, suggesting that racing behaviour can be
decomposed into risk tolerance, line choice, margin management, and robustness.
Beckman's Physics of Racing provides a physical basis for this decomposition:
the driver manages a finite traction budget, speed through a corner depends on
curvature and lateral acceleration, racing lines increase effective radius,
braking distance grows with speed squared, and smoothness reflects controlled
movement through the grip envelope. V2.1 therefore frames reward design as
progress maximisation under traction-budget, braking, line-quality, and
smoothness constraints.
```

## North Star

The aspirational V2.1 agent should be describable like a racing driver:

```text
It carries speed where curvature allows it.
It brakes early enough for the next corner.
It spends traction deliberately across braking, turning, and acceleration.
It uses road width to increase effective radius.
It follows or discovers racing-line structure without being locked to rails.
It preserves enough edge margin to survive.
It transitions smoothly through the grip envelope.
It can be made more conservative or aggressive through an explicit parameter.
It remains robust when randomness is introduced later.
```

That is the conceptual promise of V2.1: not just a better reward curve, but a
reward model that makes the learned driving behaviour explainable.
