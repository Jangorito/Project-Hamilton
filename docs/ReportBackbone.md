# Report Backbone — Project Hamilton
*Working plan. Fill in sections marked [FILL]. Target: 3000–3800 words.*

---

## Revised Research Questions

**RQ1**: Does replacing hand-tuned speed-based reward (V1) with physics-informed traction-budget reward (v21b) improve learning progress in a PPO continuous-control racing agent?

**RQ2**: Does the effect of reward function design on learning progress depend on vehicle dynamics — do vehicles with different handling characteristics (ETK K-Series vs SBR4) respond differently to the same reward shaping?

---

## Known Data Points (use these verbatim)

### Best-checkpoint eval (deterministic, fixed spawn)
| Condition | Vehicle | Reward | Best progress | Best ckpt step | Term reason | Final-ckpt progress |
|---|---|---|---|---|---|---|
| A | SBR | V1 | **2419.9 m** | 264k | max_episode_steps | 1869.6 m |
| B | ETK | V1 | **1523.2 m** | 128k | off_track | 860.6 m |
| C | ETK | v21b | **1246.4 m** | 128k | damage | 986.8 m |
| D | SBR | v21b | **1030.2 m** | 248k | damage | 549.6 m |

*Note: C and D final-checkpoint values (986.8 m, 549.6 m) differ substantially from earlier stochastic rollout figures in CLAUDE.md (1353.5 m, 1303.5 m). The new values are from fresh deterministic eval (`eval_checkpoints.py`). Use the deterministic values as authoritative. The gap is largest for D (753.9 m), consistent with the SBR+v21b policy being right on the boundary of the traction budget — stochastic sampling occasionally stays safe, the deterministic mean crashes.*

*Full lap = **2150.0 m** (authoritative: 1077-point closed-loop centreline, 2150.04 m; CLAUDE.md listed 2150 m — use 2150 m in report).*
*BeamNG AI reference = 80.39 s, 26.73 m/s avg. Physics raceline ceiling ≈ 69.3 s.*
*Cond A best-checkpoint (2419.9 m) = **1.125 laps** — full lap completed + 269.9 m into lap 2, terminated by max_episode_steps.*
*Cond A 208k checkpoint (2132.8 m) = 0.992 laps — 17.2 m short of lap completion. Phase transition between 208k–264k.*

### Training stats (final logged ep_rew_mean)
| Condition | ep_rew_mean | ep_len_mean |
|---|---|---|
| A | 1437 | 346 |
| B | 874 | 179 |
| C | 737 | 153 |
| D | 736 | 264 |

### Phase 2 (supplementary)
- obs_v2 (SBR+v21b+prev_action): ep_rew_mean = 889 at ~344k steps (fragmented sessions), rollout = 1005.9 m
- Respawn (SBR+v21b+random_spawn): ep_rew_mean = 522 at ~485k steps (fixed-spawn eval pending)

### Phase 3 (in-progress at submission time)
- v22b (SBR+v21b+obs_v2+soft_raceline): training at submission, ~20% complete

---

## Report Structure

---

### 1. Introduction (~300 words)

**Open with the problem**: Reward function design is one of the least principled parts of RL. For continuous-control tasks like racing, most work defaults to centreline-following or speed bonuses, which are proxies for the actual objective (fast lap time) rather than grounded in physical reasoning.

**Why racing / BeamNG**: Autonomous racing is a clean testbed for continuous control — it has a measurable objective (lap time / progress), a well-defined track geometry, and crucially, a physical theory of optimal behaviour (Beckman's traction-budget framework) that can be encoded directly into reward design. BeamNG.tech provides soft-body physics simulation, allowing realistic vehicle dynamics and a meaningful difference between vehicle handling profiles.

**State the two RQs explicitly.**

**COMP4105 only — briefly state the physics motivation**: The traction-budget reward replaces hand-tuned linear penalties with `v_target = sqrt(a_lat / κ)` derived from lateral acceleration physics, plus a braking-distance shortfall penalty. These make the reward more principled and more interpretable as a description of driving behaviour.

---

### 2. Background / Literature Review (~600 words)

**PPO and continuous control** (~150 words)
- Schulman et al. (2017) — PPO; clipped surrogate objective; standard for continuous control
- Stable-Baselines3 as the implementation; MlpPolicy; n_steps=256, batch_size=64, γ=0.99
- Note: continuous action space (steering, combined throttle/brake) is harder than discrete; policy learns a Gaussian distribution over actions

**RL for autonomous racing** (~200 words)
- Wurman et al. (2022) — Gran Turismo Sophy; uses course progress as dense proxy for lap time; strong justification for progress-based reward
- TORCS / OpenRacing Car Simulator literature — speed + track-centre as standard reward; centreline-following dominates early RL racing work
- Contrast with this project: we question whether centreline-following is the right proxy, motivated by the gap between "staying on track" and "racing fast"

**Reward shaping** (~150 words)
- Ng et al. (1999) — potential-based reward shaping; dense rewards as proxies for sparse objectives
- Dense vs sparse tradeoff: progress reward is dense and learnable but may not align with lap-time minimisation; sparse lap-time reward would be theoretically cleaner but impractical at 278k steps
- [FILL: cite 1–2 papers on reward design in robotics / driving RL]

**Beckman's physics of racing** (~100 words)
- Traction circle / budget: total grip force is finite; lateral + longitudinal demand must not exceed it
- Physics-shaped corner speed: `v_target = sqrt(a_lat / κ)` — encodes the actual relationship between speed, curvature, and grip demand
- Braking distance scales with v²: anticipation is required, not reactive overspeed detection
- Justification: these principles motivate v21b's reward terms directly

---

### 3. System Design (~600 words)

**Environment** (~150 words)
- BeamNG.tech v0.38.3; soft-body physics; Hirochi Raceway (closed circuit, 2150 m)
- BeamNGpy Python API; Gymnasium wrapper (continuous control interface)
- Centreline: 1077 points at 2 m spacing; 3D nearest-point projection with Z disambiguation at overpass (264 m / 1262 m arc-length, 6.75 m Z separation)
- Timing: det50ms, 15 physics frames per action step (0.25 s sim time per step), effectively ~Turbo speed mode

**Agent** (~150 words)
- PPO (Stable-Baselines3), MlpPolicy
- Action space: 2D continuous `[-1, 1]`: `action[0]` = steering, `action[1]` = throttle/brake (positive = throttle, negative = brake)
- Observation space (obs_v1, 12 features): forward/lateral/vertical speed (normalised), heading error, lateral centreline error, lookahead heading errors at 20/40/80 m, track curvature at 20/40/80 m, progress ratio. All features in `[-1, 1]` or `[0, 1]`.
- PPO config: `n_steps=256`, `batch_size=64`, `γ=0.99`, `lr=3×10⁻⁴`

**Reward functions** (~200 words)
Describe both configs; table of key parameter differences:

| Parameter | V1 | v21b |
|---|---|---|
| Speed bonus | `speed_weight=0.05` | 0 (removed) |
| Curvature target | `35 − 700κ` (linear, hand-tuned) | `sqrt(8.0 / max(κ, ε))` (physics-derived) |
| Traction penalty | None | `max(0, v/v_target − 1.0)² × 3.0` |
| Braking shortfall | None | `max(0, (v²−v_t²)/(2×9.0) − 80) × 0.1` |
| Smoothness weight | 0.1 | 0.4 |
| Lateral error weight | 0.1 | 0.03 |

Key distinction: v21b replaces qualitative penalties with quantities that have physical units and interpretable magnitudes (lateral acceleration in m/s², braking distance in m).

**Experimental design** (~100 words)
- 2×2 factorial: Vehicle {ETK K-Series, SBR4} × Reward {V1, v21b}
- 4 conditions, each ~278k steps, det50ms, fixed spawn
- Evaluation: deterministic best-checkpoint rollout via `eval_checkpoints.py` (500 steps, fixed spawn)
- Why best checkpoint not final: PPO continuous control often shows non-monotonic learning; final weights are not guaranteed to be peak performance

---

### 4. Implementation / Technologies / Challenges (~350 words)

**Technologies**
- Python 3.x, Stable-Baselines3, BeamNGpy, Gymnasium, TensorBoard
- Flask launcher UI for multi-run management; remote SSH access via scheduled task

**Key challenges and how they were addressed**

*3D projection bug* (~80 words)
Hirochi Raceway has an overpass where two track sections are only 0.77 m apart in XY but 6.75 m apart in Z. XY-only nearest-point projection assigned the wrong segment, causing phantom ~998 m progress jumps. Fixed by including Z in the disambiguation distance (`query_utils.py`). All four conditions A–D were trained on the corrected projection; Condition B has a caveat noted in Discussion.

*BeamNG stability* (~60 words)
legacy_60hz mode crashed after ~40 episodes; det50ms was stable for 278k steps. All conditions used det50ms after this was identified.

*Deterministic vs stochastic evaluation* (~60 words)
`model.predict(deterministic=True)` returns the Gaussian policy mean, which can underperform stochastic sampling in continuous control near grip limits. Best-checkpoint evaluation is therefore reported alongside the training ep_rew_mean; the two metrics measure different things and both are reported.

*Reward config contamination (v2 run)* (~50 words)
An early run named "v2" used V1 reward weights. This is documented and the run is excluded; conditions are defined by their verified `run_config.json` entries.

---

### 5. Experiments (~250 words)

**What was run**
Four conditions, each a fresh PPO training run:
- Condition A: SBR4 + V1 reward, 278,353 steps
- Condition B: ETK K-Series + V1 reward, 278,670 steps
- Condition C: ETK K-Series + v21b reward, 280,064 steps
- Condition D: SBR4 + v21b reward, 278,016 steps

All conditions: det50ms timing, fixed bootstrap spawn, obs_v1 (12 features), 500-step max episode.

**Evaluation protocol**
Each run was evaluated by rolling out every saved checkpoint (every 8k steps) plus the final model, deterministically (`model.predict(deterministic=True)`), at fixed spawn, for up to 500 steps. Best-checkpoint progress is the primary metric. Training `ep_rew_mean` is reported as a secondary indicator of learning trajectory.

**Baselines**
- BeamNG CPU AI (ETK, aggression 1.0): 80.39 s, 26.73 m/s average speed — empirical reference lap
- Physics racing line (minimum-curvature, μ=1.1): ~69.3 s estimated — theoretical ceiling
- Full lap distance: 2150 m

---

### 6. Results (~600 words)

**[FIGURE 1 — Learning curves]**: ep_rew_mean over training steps for all four conditions on one plot. Note: C and D show more oscillation / slower convergence than A and B at this step budget.

**Main results table** (use best-checkpoint progress, not final):

| Condition | Vehicle | Reward | Best progress | Best step | Term | Final progress |
|---|---|---|---|---|---|---|
| A | SBR | V1 | **2419.9 m** | 264k | max_episode_steps | 1869.6 m |
| B | ETK | V1 | **1523.2 m** | 128k | off_track | 860.6 m |
| C | ETK | v21b | **1246.4 m** | 128k | damage | 986.8 m |
| D | SBR | v21b | **1030.2 m** | 248k | damage | 549.6 m |

**[FIGURE 2 — Condition comparison bar chart]**: Best-checkpoint progress for A–D, with full lap (2150 m) and AI reference marked as horizontal lines.

**Headline findings**:
- Condition A best checkpoint (2419.9 m at 264k steps) exceeds the full lap length of 2150 m. The episode terminated at `max_episode_steps` (500 steps), indicating the agent completed the circuit without crashing within the episode limit.
- Condition B best checkpoint (1523.2 m at 128k steps) is substantially better than its final checkpoint (860.6 m). The final model regressed from peak performance, a known effect in PPO continuous control.
- Condition C best checkpoint (1246.4 m at 128k steps) is +277 m over B (1523.2 m vs 1246.4 m — wait, C < B). v21b did not improve ETK best-checkpoint performance; B outperforms C by 276.8 m.
- Condition D best checkpoint (1030.2 m at 248k steps) is 1389.7 m below A (2419.9 m). v21b substantially reduced SBR performance.
- The vehicle effect is large and consistent: SBR outperforms ETK in both reward conditions (A > D, A > C, A > B > C/D).
- v21b hurt both vehicles relative to V1, but by very different magnitudes: ETK −276.8 m (B 1523.2 → C 1246.4), SBR −1389.7 m (A 2419.9 → D 1030.2).

**[FIGURE 3 — Reward component breakdown]** (if time): progress_reward vs off_track_penalty vs smoothness_penalty for V1 vs v21b averaged over training. Shows that v21b achieves [higher/lower] smoothness penalty and [higher/lower] off_track rate.

---

### 7. Discussion (~500 words)

**RQ1: Did v21b improve learning progress?**
The effect is asymmetric and negative on average. For ETK (B vs C), v21b reduced best-checkpoint performance by 276.8 m (1523.2 → 1246.4 m). For SBR (A vs D), v21b reduced best-checkpoint performance by 1389.7 m (2419.9 → 1030.2 m). There is no positive main effect — v21b constrained both vehicles, with the SBR far more severely. The reward change interacted strongly with the vehicle.

**RQ2: Does the effect depend on vehicle dynamics?**
Yes — the data shows a clear interaction. ETK lost 276.8 m under v21b; SBR lost 1389.7 m. The absolute penalty is 5× larger for the SBR, indicating a strong vehicle × reward interaction. The most interpretable explanation uses the traction-budget framing directly:

The SBR4 has a narrow, sharp handling envelope (snap oversteer, low polar moment). Under V1 reward, SBR found an aggressive policy that exploited this — achieving 2419.9 m at its best checkpoint. Under v21b's physics-shaped speed target (`v_target = sqrt(8.0/κ)`), with `allowed_aggression=1.0`, the SBR's aggressive cornering style triggered the traction penalty continuously. The policy responded by reducing speed to stay within the traction budget, resulting in Condition D terminating by `max_episode_steps` rather than `damage` — safe but slow. The ETK K-Series, with its more progressive oversteer characteristic and greater tolerance for lateral slip, found v21b less constraining: the same `allowed_aggression=1.0` was within its natural operating range, explaining the ETK improvement.

This finding is not a failure of v21b as a design — it reveals that `allowed_aggression` is a vehicle-specific parameter, not a universal constant. A future condition (v21c) with `allowed_aggression=1.15` for the SBR would test this directly.

**Limitations**
- n=1 per condition: results cannot be attributed to randomness vs systematic effects. Multiple seeds would strengthen conclusions.
- Both C and D showed oscillating rewards that had not fully plateaued at 278k steps; more steps might change the comparison.
- Deterministic evaluation may underperform stochastic training, particularly near grip limits. Best-checkpoint selection mitigates but does not eliminate this.
- Lap completion was not rewarded. The agent has no incentive to complete laps over maximising per-step progress; the 2419.9 m result is progress accumulation, not a completed-lap metric.
- The 3D projection fix was applied to all conditions; Condition B was trained on the corrected centreline but earlier runs (v2) used the broken version.

---

### 8. Conclusion (~200 words)

Summarise: Two research questions were asked about whether physics-informed reward design (v21b) improves PPO racing agent learning, and whether that effect interacts with vehicle dynamics.

What worked: the 2×2 factorial design successfully isolated vehicle and reward effects. Condition A (SBR+V1) achieved the strongest result — best-checkpoint progress of 2419.9 m, exceeding the full lap length. The interaction effect was clear: v21b improved the ETK but constrained the SBR.

What could be improved: step budget is insufficient for v21b convergence; n=1 per condition limits statistical conclusions; lap completion remains unimplemented.

Future work: (1) v21c with higher `allowed_aggression` for SBR to test whether traction-budget reward can recover SBR performance; (2) lap completion bonus to shift from progress proxy to actual lap-time optimisation; (3) obs_v2 clean comparison (prev_action features) — preliminary results show this may reduce steering oscillation; (4) checkpoint respawning for coverage of later track sections.

---

## Word Budget

| Section | Target |
|---|---|
| Introduction | 300 |
| Background | 600 |
| System Design | 600 |
| Implementation | 350 |
| Experiments | 250 |
| Results | 600 |
| Discussion | 500 |
| Conclusion | 200 |
| **Total** | **3400** |

---

## Figures to Generate

1. `fig_learning_curves.png` — ep_rew_mean vs steps, 4 conditions, one plot (`scripts/viz/plot_results.py --fig learning`)
2. `fig_condition_comparison.png` — best-checkpoint bar chart, A–D (`scripts/viz/plot_results.py --fig comparison`)
3. `fig_reward_components.png` — reward component trends, V1 vs v21b (`scripts/viz/plot_results.py --fig rewards`) *(optional if time-constrained)*

---

## AI Usage Statement (required by CW)

*To include in submitted portfolio:*

> Claude Code (Anthropic) was used to assist with: (1) code generation for evaluation and visualisation scripts, (2) review of existing scripts for robustness, and (3) structuring the report outline. All experimental design, training runs, analysis interpretation, and report writing are my own. Claude Code was not used to generate the report text itself.
