# Section 3 Draft — System Design (~600 words)

---

## 3. System Design

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

---

## Context for continuation

**Research questions:**
- RQ1: Does replacing hand-tuned speed-based reward (V1) with physics-informed traction-budget reward (v21b) improve learning progress?
- RQ2: Does the effect depend on vehicle dynamics (ETK K-Series vs SBR4)?

**Key results (deterministic best-checkpoint eval):**

| Cond | Vehicle | Reward | Best progress | Best ckpt | Termination | Final progress |
|---|---|---|---|---|---|---|
| A | SBR | V1 | **2419.9 m** | 264k | max_episode_steps | 1869.6 m |
| B | ETK | V1 | **1523.2 m** | 128k | off_track | 860.6 m |
| C | ETK | v21b | **1246.4 m** | 128k | damage | 986.8 m |
| D | SBR | v21b | **1030.2 m** | 248k | damage | 549.6 m |

Full lap = 2150 m. Condition A best checkpoint (2419.9 m) = 1.125 laps — agent completed the circuit and continued into lap 2.

**Training ep_rew_mean at end of training:**
- A: 1437, B: 874, C: 737, D: 736

**Headline findings:**
- v21b did not improve either vehicle. ETK loss: −276.8 m (B→C). SBR loss: −1389.7 m (A→D).
- Vehicle effect large and consistent: SBR outperforms ETK in both reward conditions.
- Interaction: v21b's traction constraint is ~5× more damaging for SBR than ETK, consistent with SBR's narrower handling envelope triggering the traction penalty continuously under the same `allowed_aggression=1.0`.
- Deterministic eval gap is largest for D (753.9 m vs stochastic figures), consistent with SBR+v21b sitting on the traction boundary — stochastic sampling occasionally stays safe, the deterministic mean crashes.

**Sections still to draft:** Introduction (300w), Background/Lit Review (600w), Implementation/Challenges (350w), Experiments (250w), Results (600w), Discussion (500w), Conclusion (200w).

**Sections already drafted:** Section 3 above (system design, ~700w).
