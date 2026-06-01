# Project Hamilton — Report Draft Context

**Task:** Help draft the remaining sections of this undergraduate CS dissertation (COMP4105). Section 3 is already written (below). The report targets 3000–3800 words total. Write in clear academic prose: no bullet points in the body text, no hedging filler, no padding. Cite placeholders as [CITE AuthorYear].

---

## Research Questions

**RQ1:** Does replacing hand-tuned speed-based reward (V1) with physics-informed traction-budget reward (v21b) improve learning progress in a PPO continuous-control racing agent?

**RQ2:** Does the effect of reward function design on learning progress depend on vehicle dynamics — do vehicles with different handling characteristics respond differently to the same reward shaping?

---

## Project Summary

**Project Hamilton** trains a PPO-based autonomous racing agent inside BeamNG.tech (soft-body physics simulator) on Hirochi Raceway. The core experiment is a 2×2 factorial: two vehicles × two reward functions. The agent learns to drive continuously — steering and throttle/brake are both continuous, not discrete — using only compact vehicle-state and track-geometry observations, not camera images.

The motivating claim is that reward functions grounded in racing physics (traction budget, braking distance) are more principled than hand-tuned speed/curvature penalties, and may produce better or more interpretable behaviour. The experiment tests this directly.

---

## Experimental Conditions

| Cond | Vehicle | Reward | Steps | Best progress | Best ckpt | Termination | Final progress |
|---|---|---|---|---|---|---|---|
| A | SBR4 | V1 | 278k | **2419.9 m** | 264k | max_episode_steps | 1869.6 m |
| B | ETK K-Series | V1 | 278k | **1523.2 m** | 128k | off_track | 860.6 m |
| C | ETK K-Series | v21b | 280k | **1246.4 m** | 128k | damage | 986.8 m |
| D | SBR4 | v21b | 278k | **1030.2 m** | 248k | damage | 549.6 m |

**Full lap = 2150 m.**
Condition A best checkpoint (2419.9 m) = **1.125 laps** — agent completed the circuit and continued 270 m into lap 2, terminated by max episode steps (500 steps), not by crashing.

**Training ep_rew_mean at end of training:** A: 1437 | B: 874 | C: 737 | D: 736

**Baselines:**
- BeamNG CPU AI (ETK, aggression 1.0): 80.39 s, 26.73 m/s average — empirical reference lap
- Physics racing line (minimum-curvature, μ=1.1): ~69.3 s estimated — theoretical ceiling
- Full lap: 2150 m

---

## Key Findings

- v21b did not improve either vehicle. ETK loss: −276.8 m (B 1523.2 → C 1246.4). SBR loss: −1389.7 m (A 2419.9 → D 1030.2).
- Vehicle effect is large and consistent: SBR outperforms ETK in both reward conditions.
- **Interaction effect:** v21b's traction constraint is ~5× more damaging for SBR than ETK. The SBR4 has a narrow, sharp handling envelope (snap oversteer, low polar moment). Under V1 it found an aggressive policy that exploited this. Under v21b's physics-shaped speed target (`v_target = sqrt(8.0/κ)`) the SBR's cornering style continuously triggered the traction penalty; the policy responded by reducing speed. The ETK K-Series, with more progressive oversteer and greater tolerance for lateral slip, found the same `allowed_aggression=1.0` within its natural operating range.
- Deterministic evaluation gap largest for D (753.9 m vs earlier stochastic figures): SBR+v21b policy sits right on the traction boundary — stochastic sampling occasionally stays safe, the deterministic mean crashes.
- Condition B final checkpoint (860.6 m) is substantially below its best checkpoint (1523.2 m at 128k): PPO continuous-control is non-monotonic; the best checkpoint is always the right metric, not the final.

---

## System Overview (for context when drafting other sections)

**Simulator:** BeamNG.tech v0.38.3, soft-body physics. Track: Hirochi Raceway, 2150 m closed circuit.

**Vehicles:**
- *ETK K-Series (etkc):* RWD, high polar moment, progressive oversteer — forgiving under imprecise inputs.
- *SBR4 (sbr):* RWD, low CoG, snap oversteer — punishes imprecise actions; random exploration is destructive.

**Agent:** PPO (Stable-Baselines3, MlpPolicy). Action space: 2D continuous [−1,1] (steering; combined throttle/brake). Obs space: 12 features (speeds, heading/lateral error, 20/40/80 m lookahead heading + curvature, progress ratio). PPO config: n_steps=256, batch=64, γ=0.99, lr=3×10⁻⁴.

**Timing:** det50ms, 15 physics frames per action = 0.25 s sim time per step.

**Episode limits:** 500 steps max | lateral error > 10 m (off-track) | damage > 500 | 100 stuck steps.

**Evaluation:** `eval_checkpoints.py` — rolls out every checkpoint (every 8k steps) + final model, deterministically, fixed spawn, 500 steps. Best-checkpoint progress is primary metric.

---

## Reward Functions

**V1 — hand-tuned baseline**
Progress reward + flat speed bonus (weight 0.05) + linear curvature overspeed penalty (`35 − 700κ`) + smoothness penalty (weight 0.1) + lateral error penalty (weight 0.1).

**v21b — physics-informed**
Removes flat speed bonus. Replaces linear curvature target with `v_target = sqrt(8.0 / max(κ, ε))` — derived from Beckman's traction-budget principle: for a given lateral acceleration budget and corner curvature, this is the maximum speed at which grip demand stays within the tyre's limit. Adds traction overrun penalty and braking shortfall penalty. Increases smoothness weight, reduces lateral error weight.

| Parameter | V1 | v21b |
|---|---|---|
| Speed bonus | `0.05` (flat) | — |
| Curvature target | `35 − 700κ` | `sqrt(8.0 / max(κ, ε))` |
| Traction penalty | — | `max(0, v/v_target − 1)² × 3.0` |
| Braking shortfall | — | `max(0, (v²−v_t²) / (2×9) − 80) × 0.1` |
| Smoothness weight | 0.1 | 0.4 |
| Lateral error weight | 0.1 | 0.03 |

Key distinction: v21b's terms have physical units (m/s², metres) and magnitudes tied to known vehicle limits. V1's coefficients are empirically tuned without physical referents.

---

## Word Budget

| Section | Target | Status |
|---|---|---|
| 1. Introduction | 300w | to draft |
| 2. Background / Lit Review | 600w | to draft |
| 3. System Design | 600w | **drafted (below)** |
| 4. Implementation / Challenges | 350w | to draft |
| 5. Experiments | 250w | to draft |
| 6. Results | 600w | to draft |
| 7. Discussion | 500w | to draft |
| 8. Conclusion | 200w | to draft |
| **Total** | **3400w** | |

---

## Section Briefs for Remaining Sections

### 1. Introduction (~300w)
Open with the problem: reward function design is one of the least principled parts of RL. For continuous-control racing, most work defaults to centreline-following or speed bonuses — proxies for the actual objective rather than physical reasoning. Introduce BeamNG as a testbed: soft-body physics, measurable objective, and a physical theory of optimal behaviour (Beckman) that can be encoded directly into reward design. State RQ1 and RQ2 explicitly. Briefly state the physics motivation: v21b replaces hand-tuned penalties with `v_target = sqrt(a_lat / κ)` from lateral acceleration physics, plus a braking-distance shortfall penalty.

### 2. Background / Lit Review (~600w)
Cover three areas:

*PPO and continuous control (~150w):* Schulman et al. 2017 — clipped surrogate objective; standard for continuous control. Stable-Baselines3 as implementation. Note that continuous action spaces are harder than discrete: the policy learns a Gaussian distribution over actions.

*RL for autonomous racing (~200w):* Wurman et al. 2022 (Gran Turismo Sophy) — uses course progress as dense proxy for lap time. TORCS literature — speed + track-centre as standard reward. Contrast: this project questions whether centreline-following is the right proxy, motivated by the gap between "staying on track" and "racing fast."

*Reward shaping (~150w):* Ng et al. 1999 — potential-based shaping; dense rewards as proxies for sparse objectives. Dense vs sparse tradeoff: progress reward is learnable but may not align with lap-time minimisation.

*Beckman's physics of racing (~100w):* Traction circle: total grip force is finite; lateral + longitudinal demand must not exceed it. Physics-shaped corner speed: `v_target = sqrt(a_lat / κ)`. Braking distance scales with v²: anticipation is required. These principles motivate v21b's reward terms directly.

### 4. Implementation / Technologies / Challenges (~350w)
Technologies: Python, Stable-Baselines3, BeamNGpy, Gymnasium, TensorBoard.

Key challenges:

*3D projection bug (~80w):* Hirochi Raceway has an overpass where two track sections are <1 m apart in XY but 6.75 m apart in Z. XY-only nearest-point projection assigned the wrong segment, causing phantom ~998 m progress jumps. Fixed by including Z in disambiguation. All four conditions A–D were trained on the corrected projection.

*BeamNG stability (~60w):* legacy_60hz mode crashed after ~40 episodes; det50ms was stable for 278k steps.

*Deterministic vs stochastic evaluation (~60w):* `model.predict(deterministic=True)` returns the Gaussian mean, which can underperform stochastic sampling in continuous control near grip limits. Best-checkpoint evaluation mitigates but does not eliminate this — both metrics are reported.

*Reward config contamination (~50w):* An early run used V1 reward weights under a v2 name. Documented and excluded; conditions are defined by their verified run_config.json entries.

### 5. Experiments (~250w)
List the four conditions with exact step counts (A: 278,353 | B: 278,670 | C: 280,064 | D: 278,016). All conditions: det50ms timing, fixed bootstrap spawn, obs_v1 (12 features), 500-step max episode. Evaluation protocol: every 8k-step checkpoint + final model, deterministic, fixed spawn, 500 steps. Baselines: BeamNG CPU AI (80.39 s), physics racing line (~69.3 s), full lap (2150 m). Note why best checkpoint not final.

### 6. Results (~600w)
Lead with the main results table (use data above). Then narrate each headline finding:
- A completes 1.125 laps — first condition to exceed the full lap distance.
- B best checkpoint (128k) substantially outperforms B final (860.6 m) — non-monotonic learning, 662.6 m regression.
- v21b did not improve ETK: C (1246.4 m) < B (1523.2 m) by 276.8 m.
- v21b strongly reduced SBR performance: D (1030.2 m) vs A (2419.9 m), a 1389.7 m deficit.
- Vehicle effect: SBR > ETK in both reward conditions.
- Interaction: magnitude of v21b penalty ~5× larger for SBR.
Reference figures/tables that should appear here: learning curves (ep_rew_mean over steps, 4 conditions), bar chart of best-checkpoint progress with 2150 m and AI reference marked.

### 7. Discussion (~500w)
*RQ1:* v21b did not improve learning progress — negative main effect for both vehicles. ETK −276.8 m, SBR −1389.7 m.

*RQ2:* Strong vehicle × reward interaction. Interpretable via traction-budget framing: SBR's aggressive policy under V1 continuously exceeded v21b's traction limit at `allowed_aggression=1.0`; ETK's more progressive dynamics were within the same budget. This is not a failure of v21b — it reveals that `allowed_aggression` is vehicle-specific. A future condition (v21c, `allowed_aggression=1.15`) would test this directly.

*Limitations:* n=1 per condition; C and D hadn't fully converged at 278k steps; deterministic evaluation may underperform stochastic; lap completion not rewarded (agent has no incentive to finish circuits, only to accumulate dense progress).

### 8. Conclusion (~200w)
Summarise what was asked and what was found. What worked: 2×2 factorial cleanly isolated vehicle and reward effects; Condition A achieved 1.125 laps. What could be improved: step budget for v21b convergence; n=1 limits statistical confidence; lap completion still unimplemented. Future work: (1) v21c with higher `allowed_aggression` for SBR; (2) lap completion bonus to shift from progress proxy to lap-time optimisation; (3) obs_v2 clean comparison (prev_action features).

---

## Section 3 — System Design (drafted, ~700w)

### 3.1 Simulation Environment

The experiment used BeamNG.tech v0.38.3, a soft-body physics simulator providing high-fidelity vehicle dynamics including tyre deformation, suspension travel, and incremental damage modelling. All training and evaluation took place on Hirochi Raceway, a closed circuit of 2150 m. The track includes a bridge overpass where two road segments run within 1 m of each other horizontally but are separated by 6.75 m vertically — a geometry that required explicit 3D nearest-point disambiguation during progress tracking (Section 4).

Vehicle state was accessed via the BeamNGpy Python API. A custom Gymnasium-compatible wrapper (`BeamNGRacingEnv`) converted simulator state into observation vectors and reward scalars at each policy step. The track centreline was precomputed as a 1077-point closed loop at 2 m spacing. At each step, the nearest centreline segment was identified by minimum 3D Euclidean distance, from which arc-length progress, lateral error, local heading, and 20/40/80 m lookahead curvature values were derived.

Simulation timing used the `det50ms` profile: each policy step consumed 15 physics frames at 50 ms per frame, giving 0.25 s of simulated time per step. Episodes terminated under any of four conditions: 500 steps elapsed (max episode), lateral error exceeded 10 m (off-track), accumulated damage exceeded 500 units, or the vehicle made negligible forward progress for 100 consecutive steps (stuck).

### 3.2 Agent

The agent used Proximal Policy Optimisation [CITE Schulman 2017] implemented via Stable-Baselines3 with an MLP policy. The action space was a 2D continuous vector in [−1, 1]: `action[0]` controlled steering and `action[1]` provided a combined throttle/brake output, where positive values applied throttle and negative values applied braking. This unified axis prevents simultaneous throttle and brake application.

The observation vector (obs_v1, 12 features) comprised: forward, lateral, and vertical speeds normalised to vehicle-frame maximums (60, 20, 10 m/s); heading error and lateral centreline error; lookahead heading errors at 20, 40, and 80 m; track curvature at 20, 40, and 80 m ahead; and a progress ratio in [0, 1]. All features were clipped to [−1, 1]. This gives the policy a compact representation of current vehicle state, alignment, and forward track geometry at three scales — analogous to a driver's awareness of their position and what the road ahead requires.

PPO hyperparameters were fixed across all conditions: `n_steps=256`, `batch_size=64`, γ=0.99, learning rate 3×10⁻⁴.

### 3.3 Reward Functions

Two reward configurations were compared. **V1** is a hand-tuned baseline awarding a flat speed bonus and applying a linear curvature overspeed penalty of `35 − 700κ` with weak smoothness and lateral error terms. **v21b** removes the flat speed bonus and replaces the linear curvature target with `v_target = sqrt(8.0 / max(κ, ε))`, derived from Beckman's traction-budget principle [CITE Beckman]: for a given lateral acceleration budget and corner curvature, this is the maximum speed at which grip demand remains within the tyre's limit. A traction overrun penalty fires when the vehicle exceeds this speed, and a braking shortfall penalty penalises arriving at a tight corner with more kinetic energy than an 80 m braking distance can absorb. Smoothness weight was increased and centreline-following weight reduced, giving the policy latitude to take a wider racing line.

| Parameter | V1 | v21b |
|---|---|---|
| Speed bonus | `0.05` (flat) | — |
| Curvature target | `35 − 700κ` | `sqrt(8.0 / max(κ, ε))` |
| Traction penalty | — | `max(0, v/v_target − 1)² × 3.0` |
| Braking shortfall | — | `max(0, (v²−v_t²) / (2×9) − 80) × 0.1` |
| Smoothness weight | 0.1 | 0.4 |
| Lateral error weight | 0.1 | 0.03 |

The key distinction is interpretability: v21b's penalty terms have physical units (m/s², metres of braking distance) and magnitudes tied to known vehicle limits, whereas V1's coefficients are empirically tuned without a principled physical referent.

### 3.4 Experimental Design

A 2×2 factorial design crossed vehicle ({ETK K-Series, SBR4}) with reward function ({V1, v21b}), yielding four conditions. The ETK K-Series is a rear-wheel-drive touring car with a high polar moment and progressive oversteer, forgiving under imprecise inputs. The SBR4 is a lightweight rear-wheel-drive track car with a low centre of gravity and snap oversteer — sensitive to traction overruns and punishing of aggressive cornering inputs. Each condition was trained from a fresh random initialisation for approximately 278k timesteps under identical PPO hyperparameters, timing profile, and observation configuration.

Evaluation used `eval_checkpoints.py`, which rolled out every saved checkpoint (every 8k steps) plus the final model deterministically at a fixed spawn point for up to 500 steps. Best-checkpoint progress — the maximum across all checkpoints — is the primary reported metric. Final-checkpoint progress is also recorded, but PPO continuous-control training is non-monotonic: the final weights do not reliably represent peak performance, and in all four conditions the best checkpoint preceded the final checkpoint.
