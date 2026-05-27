# BeamNG RL Research — TODO

Structured around the full experimental map. Each avenue is a comparative study
with its own findings to discuss — not just optimisations bolted on.

---

## 🔴 Backbone — 2×2 Factorial (Conditions A & C)

*Must be complete before any avenue can be meaningfully interpreted in context.*

| | V1 Reward | V2 Reward |
|---|---|---|
| **ETK** | B ✅ `v1_etk_2x` — 955.6 m, 165k steps | C ⬜ needs training |
| **SBR4** | A ⬜ needs training | D ⬜ optional |

- [ ] **Train Condition A** — run name `v1_subaru`: sbr + V1 + det50ms 2× + 165k steps, fresh from scratch
- [ ] **Train Condition C** — run name `v1_etk_v2`: etkc + V2 + det50ms 2× + 165k steps, fresh from scratch
- [ ] Run `eval_checkpoints.py` on all three completed conditions (B, A, C) to find best checkpoint per run — the final checkpoint is not necessarily peak performance
- [ ] Collect **BeamNG CPU AI baseline** lap time on Hirochi Raceway (lap time, sector times, average speed) — used as the performance ceiling when reporting results ("agent achieved X% of AI average speed")

---

## 🟡 Avenue 1 — Exploration Strategy (Checkpoint Respawning)

*Comparison run: ETK + V1 + random spawn vs Condition B fixed start.*  
*Question: does coverage improve? Does lap completion become reachable? Does early-lap learning change?*

### Environment (`beamng_racing_env.py`)
- [ ] Add new `live_spawn_mode` value `"random_checkpoint"` — at each episode reset, sample `live_spawn_progress_m` uniformly from `[0, total_lap_length)` and spawn via the existing centreline path
- [ ] Verify `episode_start_progress_m` / `episode_progress_m` accounting is correct for arbitrary spawn points (should already be — progress is tracked relative to reset point, not lap zero)
- [ ] Verify `progress_jump_penalty` doesn't misfire when spawning near the wrap boundary — check the `near_progress_wrap` flag in reset info still works for non-zero spawn points

### Training script (`train_live_ppo_run.py`)
- [ ] Add `--spawn-mode` CLI arg and `launcher_session.json` key — selects `bootstrap` vs `random_checkpoint`
- [ ] Capture `spawn_mode` in `run_config.json` so the run is self-documenting

### Training run
- [ ] Create named run `v1_etk_respawn` — same as Condition B, only difference is spawn mode

---

## 🟡 Avenue 2 — Racing Line Reward

*Comparison run: ETK + V1 + AI racing line component vs Condition B centreline-only.*  
*Question: does a reference line change what the agent learns? Does it brake earlier? Does it help or hurt raw progress?*

### Extract the racing line (data pipeline)
- [ ] Inspect `data/hirochi_track/items.level.json` to identify the correct `highlight_ids` for Hirochi Raceway's AI waypoint roads (`raceline_loader.py` infrastructure already exists — needs the right road IDs)
- [ ] Run the loader and export the AI racing line as `data/hirochi_track/ai_raceline.json` — resampled at 2m spacing (matching centreline)
- [ ] Visually verify the extracted line is correct — sanity check against the centreline using the debug draw tooling

### Reward system (`beamng_racing_env.py`)
- [ ] Add `raceline_json_path: Path | None` constructor param — if provided, load a second `TrackCentreline` object for the racing line alongside the existing centreline
- [ ] Add two new `RewardConfig` fields: `raceline_proximity_weight` (penalises distance from AI line) and `raceline_heading_weight` (rewards heading alignment with AI line tangent)
- [ ] Add the raceline terms to `_compute_reward()` and include them in `reward_info` for logging
- [ ] Add new `REWARD_CONFIGS` entry `"v1_raceline"` — all V1 values plus the two new weights tuned to be meaningful but not dominating

### Training run
- [ ] Create named run `v1_etk_raceline` — etkc + `v1_raceline` reward + det50ms 2× + 165k steps

---

## 🟠 Avenue 3 — Behavioural Cloning Pre-training

*Comparison run: SBR + BC-initialized policy then RL vs Condition A cold start.*  
*Question: does a human-demonstration warmstart survive early SBR training where random exploration destroys the car?*

### Telemetry collection (new script)
- [ ] Write `scripts/eval/drive_and_log.py` — launches BeamNG, sets up the env (no PPO), logs `(observation_vector, action_vector)` pairs at each step while a human drives via keyboard/controller. Saves to `data/bc_telemetry/<car>/<timestamp>.npz`
- [ ] Drive 5–10 laps in each car and save telemetry — SBR is the priority

### BC training (new script)
- [ ] Write `scripts/train/train_bc.py` — loads telemetry `.npz` files, accesses the PPO policy network's `mlp_extractor` and `action_net` layers, trains via MSE on `(obs → action)` pairs, saves output as an SB3-compatible `.zip`
- [ ] Confirm BC-trained policy drives sensibly using `watch_model.py` before using it as an RL starting point

### RL fine-tuning (`train_live_ppo_run.py`)
- [ ] Add `--bc-weights PATH` CLI arg — loads pre-trained weights before calling `model.learn()` instead of random init
- [ ] Capture `bc_weights` path in `run_config.json`
- [ ] Create named run `v1_subaru_bc` — sbr + V1 + det50ms 2× + BC-init + 165k steps

---

## 🟢 Avenue 4 — Lap Completion Mechanics

*Not a standalone run — adds the completion signal, combines with respawning.*  
*Question: what does the agent actually need to close a lap?*

### Environment (`beamng_racing_env.py`)
- [ ] Implement lap completion check in `_check_termination()` — replace `lap_completed = False` at line 597 with `lap_completed = self.episode_progress_m >= self.track.total_lap_length` (episode\_progress\_m accounting is already correct for this)
- [ ] Add `lap_completion_bonus` field to `RewardConfig` (default `0.0`)
- [ ] Add the bonus to `_compute_reward()` and include it in `reward_info`
- [ ] Add new `REWARD_CONFIGS` entry `"v1_lapbonus"` — V1 values with completion bonus enabled

### Training run
- [ ] Create named run `v1_etk_respawn_lapbonus` — same as `v1_etk_respawn` with lap bonus active (respawning + bonus is the condition most likely to actually complete a lap)

---

## ⚪ Infrastructure / Cross-cutting

*Applies across multiple avenues or improves data quality for all of them.*

- [x] **`watch_model.py` vehicle model fix** — reads `vehicle_model` from `run_config.json` ✅
- [x] **Progress jump bug (3D nearest-point projection)** — fixed. Root cause: Hirochi Raceway has an overpass where segment ~132 (prog 264m, Z=25.1m) and segment ~631 (prog 1262m, Z=31.8m) are only 0.77m apart in XY but 6.75m apart in Z. XY-only disambiguation picked the wrong segment, causing the observed ~998m progress jump. Fix: `TrackCentreline.query()` now includes Z in the disambiguation distance while keeping arc-length, heading, and lateral error in XY. Verified: lower road resolves to 264m, upper road to 1262m, both correctly.
- [ ] **`checkpoints_eval.csv` for `v1_etk_2x`** — run `eval_checkpoints.py` now so the best checkpoint is known; the 160k final may not be peak performance
- [ ] **Log racing line deviation in all rollout CSVs** — once the AI raceline is extracted, log lateral distance from the AI line in every rollout CSV for all conditions retrospectively (not just the raceline-reward condition) so you can compare how close each agent got to the ideal line without retraining
- [ ] **`launcher_session.json` keys for new options** — ensure `spawn_mode`, `raceline_json_path`, and `bc_weights` are settable from the Flask launcher UI

---

## Training validity note

The 3D projection bug affects observations at any point where the track overlaps itself in XY (elevation). At those steps: `progress_ratio`, lookahead heading errors, and curvature features in the observation vector are computed from the wrong centreline segment. The `progress_jump_penalty` fires and zeroes the progress reward, so the agent was not rewarded for phantom progress. However, the observation corruption means the policy learned to associate those states with wrong geometry. Training is **not fully invalid** — 955.6 m was achieved and the policy demonstrably drives past the affected region — but Conditions A and C should be trained on the corrected projection. Condition B's results carry a caveat worth noting in the Discussion section.
