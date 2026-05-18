# Reward Function Design

## Purpose

The reward is a dense proxy for racing performance. The final racing objective is low lap time, but lap time is too sparse to use directly during step-by-step learning. The current reward therefore uses centreline progress, forward speed, heading alignment, lateral control, and explicit failure penalties.

## Current Formula

```text
reward =
    progress_delta * progress_weight
    + forward_speed_mps * speed_weight
    - abs(heading_error_rad) * heading_error_weight
    - abs(lateral_error_m) * lateral_error_weight
    - off_track_penalty_if_applicable
    - reverse_progress_penalty_if_applicable
    - stuck_penalty_if_applicable
    - progress_jump_penalty_if_applicable
```

## Components

| Component | Initial value/weight | Purpose | Literature/design basis | Risk/TODO |
| --- | ---: | --- | --- | --- |
| Progress reward | `progress_weight = 1.0` | Make forward course progress the main dense learning signal. | Gran Turismo Sport RL uses course progress as a dense proxy because raw lap time is too sparse for step-by-step learning. | Centreline progress may later need replacing or comparing with racing-line progress. |
| Forward speed reward | `speed_weight = 0.02` | Encourage useful longitudinal speed without letting speed dominate progress. | Formula RL and TORCS-style rewards include longitudinal speed. | Too much speed reward can encourage crashes, wall-riding, or poor braking. |
| Heading error penalty | `heading_error_weight = 0.2` | Discourage sideways, spinning, or backwards driving. | Formula RL uses alignment between car direction and track or racing-line direction. | May need tuning for controlled oversteer or drift-like behaviour. |
| Lateral error penalty | `lateral_error_weight = 0.1` | Encourage stable track-following near the centreline. | Racing RL rewards commonly penalise distance from road centre or track axis. | The centreline is not always the fastest racing line, so this penalty may overconstrain the agent. |
| Excessive lateral error/off-track penalty | `max_lateral_error_m = 8.0`, `off_track_penalty = 10.0` | Make likely off-track states visibly bad in reward logs. | Formula RL discusses out-of-track termination; GT Sport adds penalties for wall exploitation. | Needs replacing or strengthening with true road-boundary and collision signals. |
| Reverse progress penalty | `reverse_progress_penalty = 2.0` | Penalise meaningful movement against lap direction. | Formula RL discusses terminating backwards movement. | Termination is conservative for now; thresholding may be needed later. |
| Stuck/no-progress penalty | `min_progress_delta_m = 0.05`, `stuck_step_penalty = 0.05` | Discourage stationary or barely moving behaviour before stuck termination. | Formula RL discusses slow-progress and max-step termination. | Needs tuning with the final simulator step rate and action repeat. |
| Progress projection guard | `max_progress_delta_m = 50.0`, `progress_jump_penalty = 5.0` | Ignore one-frame centreline projection jumps that are too large to be real vehicle movement. | Defensive live-simulator guard. | Threshold may need tuning for other tracks or action repeat settings. |

## Design Rationale

Progress is the main signal because it approximates lap-time minimisation while remaining dense enough for learning. Speed is included but deliberately low-weighted because speed alone can encourage crashes, wall-riding, or poor brake usage. Heading alignment discourages sideways and backwards driving. Lateral error encourages stable track-following, but it must not permanently overconstrain the agent to the centreline because an optimal racing line may legitimately use more width. Off-track, reverse-progress, and stuck penalties make common failure modes explicit in logs and easier to debug.

## Progress Origin Note

The live BeamNG spawn uses a hand-verified bootstrap pose near the painted start/finish marker. The centreline JSON raw progress at this pose may be near the wrap boundary rather than zero, so training should use episode-relative progress and progress deltas instead of assuming raw `progress_ratio` starts at zero. Lap completion should eventually be based on accumulated `episode_progress_m`.

## Progress projection guard

BeamNG live positions are projected onto a closed centreline. Near complex track geometry, nearest-point projection may occasionally jump to a distant part of the loop for one frame. Impossible progress deltas are detected after normal wrap handling, ignored for reward and episode progress, and given a small explicit penalty. This prevents one-frame projection artefacts from dominating training logs or PPO reward updates.

## Known Limitations

- The centreline is not necessarily the optimal racing line.
- Lateral error penalties may need reducing later if the agent should discover wider racing lines.
- Collision or wall-contact penalty is not yet implemented until BeamNG collision/damage data is connected.
- The reward currently uses simple hand-tuned weights.
- Future work should tune weights empirically and compare reward variants.

## Planned Iterations

- [ ] Add BeamNG collision/damage penalty.
- [ ] Add wall-contact or impact penalty if available.
- [ ] Tune lateral error penalty to avoid preventing out-in-out racing lines.
- [ ] Compare centreline progress reward against racing-line progress reward if a racing line is later generated.
- [ ] Log reward components during training.
- [ ] Plot reward components against episode performance.
- [ ] Add reward ablation experiments if time allows.

## Tunable Constants

| Constant | Initial value | Where used |
| --- | ---: | --- |
| `progress_weight` | `1.0` | Multiplies centreline progress delta in metres. |
| `speed_weight` | `0.02` | Multiplies forward speed in metres per second. |
| `heading_error_weight` | `0.2` | Multiplies absolute heading error in radians. |
| `lateral_error_weight` | `0.1` | Multiplies absolute signed lateral error in metres. |
| `max_lateral_error_m` | `8.0` | Off-track termination and off-track penalty threshold. |
| `off_track_penalty` | `10.0` | Applied when lateral error exceeds `max_lateral_error_m`. |
| `reverse_progress_penalty` | `2.0` | Applied when progress delta is negative after wrap-around handling. |
| `min_progress_delta_m` | `0.05` | Threshold for no-progress/stuck detection. |
| `stuck_step_penalty` | `0.05` | Applied on steps below `min_progress_delta_m`. |
| `stuck_steps_limit` | `100` | Terminates after repeated low-progress steps. |
| `max_progress_delta_m` | `50.0` | Maximum allowed per-step progress delta after wrap handling. |
| `progress_jump_penalty` | `5.0` | Applied when an impossible centreline projection jump is detected. |
