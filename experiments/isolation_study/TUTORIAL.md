# SGCRL Isolation Study — Complete Tutorial

A practical guide to running the Single-Goal Contrastive RL (SGCRL) isolation
study on the NYU Torch HPC cluster (Greene, SLURM-managed GPU nodes).

---

## Table of Contents

1. [Overview](#1-overview)
2. [Setup on NYU Torch HPC](#2-setup-on-nyu-torch-hpc)
3. [Quick Test (CPU, 1 000 episodes)](#3-quick-test)
4. [Running the Full Experiments](#4-running-the-full-experiments)
5. [Configuration Reference](#5-configuration-reference)
6. [Experiment Design](#6-experiment-design)
7. [Analysing Results](#7-analysing-results)
8. [Results Directory Structure](#8-results-directory-structure)

---

## 1. Overview

### What this project does

This codebase runs a **factor isolation study** for SGCRL — a method where an
agent learns successor-feature embeddings ψ(s) via contrastive (InfoNCE) learning
on trajectory data, then navigates by taking actions that move toward states with
high ψ-similarity to the goal.

The environment is an 11 × 11 tabular gridworld (121 states, deterministic
transitions, discrete actions).  Because the state space is tiny:

- The **environment itself is pure numpy** (no GPU benefit).
- The **ψ-embedding network** (an `nn.Embedding` layer) benefits from GPU
  because it uses Adam optimisation and larger batch sizes (512 vs 128).
- The main HPC benefit is **running many seeds in parallel**, allowing
  statistically meaningful conclusions across seeds 42–46.

### Study A — Effect of stochastic success on replay composition

Goal is at grid cell (8, 8) — an interior cell with all four cardinal
neighbours open, so all four approach directions (above, below, left, right)
are physically reachable.

| Condition | Agent | Research question |
|-----------|-------|-------------------|
| A1-baseline | `NeuralSGCRLAgent` (deterministic env) | What does SGCRL learn under clean signal? |
| A2-allreplay | `StochasticNeuralAgent_AllReplay` | Does adding ALL trajectories to replay matter? |
| A2-successonly | `StochasticNeuralAgent_SuccessOnly` | What happens when only successes enter replay? |

The stochastic success probabilities are:
- `from_above` → 10 %
- `from_left`  → 25 %
- `from_right` → 50 %
- `from_below` → 80 %

The key finding is that replay buffer composition under stochastic reward
creates systematic biases in how ψ represents state-to-goal reachability,
causing the agent to over-explore low-reward approach directions.

### Study B — Representation plasticity under changing dynamics

Goal is at (10, 10).  The environment passes through three wall layouts
(FourRooms → corridor-shift → L-wall), each for 25 000 episodes.

| Condition | Agent | Research question |
|-----------|-------|-------------------|
| B1-baseline | Deterministic, **static** walls | Baseline plasticity in a fixed environment |
| B2-changing | Deterministic, **changing** walls | Does ψ adapt when the optimal route changes? |

---

## 2. Setup on NYU Torch HPC

### 2.1 SSH into Greene

```bash
ssh netid@greene.hpc.nyu.edu
```

### 2.2 Request an interactive GPU node (for initial setup)

```bash
srun --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G \
     --gres=gpu:1 --time=00:30:00 --pty /bin/bash
```

### 2.3 Load modules

```bash
module purge
module load anaconda3/2023.03
module load cuda/11.8
```

Add these lines to your `~/.bashrc` to load them automatically.

### 2.4 Create and activate the conda environment

```bash
conda create -n sgcrl python=3.10 -y
conda activate sgcrl
```

### 2.5 Clone the repository and install

```bash
cd $SCRATCH   # or wherever you keep projects
git clone https://github.com/your-org/sgcrl-isolation-study.git sgcrl
cd sgcrl/experiments/isolation_study
pip install -r requirements.txt
pip install -e .
```

### 2.6 Verify GPU availability

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); \
           print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Expected output on a GPU node:

```
CUDA: True
Device: NVIDIA A100-SXM4-80GB
```

### 2.7 Run the agent self-test

```bash
python agent.py
```

Expected output ends with `=== All tests passed ===`.

---

## 3. Quick Test

Run condition A1 for 1 000 episodes on CPU to verify the pipeline end-to-end:

```bash
python run_experiment.py \
    --condition a1 \
    --seed 0 \
    --episodes 1000 \
    --device cpu \
    --output_dir /tmp/sgcrl_test \
    --log_freq 100 \
    --snap_freq 100 \
    --checkpoint_freq 500
```

You should see log lines every 100 episodes and the output directory populated:

```
/tmp/sgcrl_test/
  run_config.json
  run_log.txt
  env_config.json
  episode_log.jsonl
  metrics.json
  psi_final.npy
  metrics/
  checkpoints/
  snapshots/
```

Run time on CPU: ~ 20–30 seconds for 1 000 episodes.

---

## 4. Running the Full Experiments

### 4.1 Single condition, single seed

```bash
python run_experiment.py \
    --condition a1 \
    --seed 42 \
    --episodes 50000 \
    --device cuda \
    --output_dir results/study_a/A1-baseline/seed_42
```

### 4.2 Multi-seed sweep (sequential, no SLURM)

```bash
python run_sweep.py \
    --condition a1 \
    --seeds 42 43 44 45 46 \
    --episodes 50000 \
    --device cuda \
    --output_dir results/study_a
```

### 4.3 Submit a single-condition array job to SLURM

```bash
# Run 5 seeds of A1 in parallel (one GPU per seed)
sbatch slurm/run_array.slurm --condition a1

# Study B
sbatch slurm/run_array.slurm --condition b2 --episodes_per_phase 25000
```

### 4.4 Submit all 5 conditions at once

```bash
bash slurm/run_all.slurm
```

This submits 5 × 5 = 25 independent GPU jobs.  Each job takes ≤ 4 hours.

### 4.5 Submit a single condition, single seed

```bash
sbatch slurm/run_condition.slurm --condition a2ar --seed 43
```

### 4.6 Resume from a checkpoint (SLURM preemption recovery)

```bash
python run_experiment.py \
    --condition a1 \
    --seed 42 \
    --episodes 50000 \
    --device cuda \
    --output_dir results/study_a/A1-baseline/seed_42 \
    --resume results/study_a/A1-baseline/seed_42/checkpoints/ep_025000.pt
```

### 4.7 Monitor running jobs

```bash
squeue -u $USER
# Or watch live:
watch -n 30 squeue -u $USER
```

### 4.8 Output files created during a run

| File | Written | Description |
|------|---------|-------------|
| `run_config.json` | At start | All hyperparameters |
| `run_log.txt` | Appended | Timestamped progress lines |
| `env_config.json` | At start | Environment configuration |
| `episode_log.jsonl` | Per episode | One JSON object per episode |
| `snapshots/index.json` | Per snap | Index of all snapshot files |
| `snapshots/psi_ep*.npy` | Per snap_freq | Full ψ embedding table |
| `snapshots/sim_ep*.npy` | Per snap_freq | (11,11) similarity map |
| `snapshots/route_ep*.npy` | Per snap_freq | Best policy-rollout trajectory |
| `snapshots/policy_ep*.npy` | Per snap_freq | (121,5) policy distribution |
| `checkpoints/ep_*.pt` | Per checkpoint_freq | Full checkpoint (model + replay + metrics) |
| `checkpoints/final.pt` | At end | Final checkpoint |
| `metrics/all_metrics.json` | At end | Summary of all 9 metrics |
| `metrics/m1_rolling_success_rate.npy` | At end | M1 rolling success rate array |
| `metrics.json` | At end | Legacy format (for old generate_plots.py) |
| `snapshots.pkl` | At end | Legacy format (for old generate_plots.py) |
| `psi_final.npy` | At end | Final ψ table (num_states × rep_dim) |
| `approach_stats.json` | At end | A2 conditions: per-direction attempt/success counts |
| `phase_info.json` | At end | B2 condition: phase boundaries and names |

---

## 5. Configuration Reference

### run_experiment.py arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--condition` | required | `a1`, `a2ar`, `a2so`, `b1`, `b2` |
| `--seed` | 42 | Random seed (numpy + torch) |
| `--study` | auto | `A` or `B` — inferred from condition |
| `--output_dir` | auto | Directory for all outputs |
| `--episodes` | 50000 | Total episodes (Study A) |
| `--episodes_per_phase` | 25000 | Episodes per phase (Study B) |
| `--device` | auto | `cuda` or `cpu` |
| `--rep_dim` | 64 | ψ embedding dimension |
| `--use_mlp` | False | Add a 2-layer MLP projection head |
| `--lr` | 1e-3 | Adam learning rate |
| `--batch_size` | 512 | InfoNCE batch size (state pairs) |
| `--temperature` | 0.07 | InfoNCE temperature |
| `--grad_clip` | 1.0 | Gradient norm clip (0 = off) |
| `--replay_capacity` | 5000 | Max trajectories in replay buffer |
| `--gamma` | 0.99 | Geometric discount for future-state sampling |
| `--max_steps` | 200 | Max steps per episode |
| `--log_freq` | 100 | Print progress every N episodes |
| `--snap_freq` | 500 | Save snapshot every N episodes |
| `--checkpoint_freq` | 5000 | Save checkpoint every N episodes |
| `--n_rollouts` | 5 | Rollouts for policy_rollout evaluation |
| `--resume` | None | Path to checkpoint .pt file |
| `--approach_probs` | None | JSON string or file for Study A approach probs |

### Which argument relates to which research question

| Argument | Research question |
|----------|------------------|
| `--condition` | Selects which factor is isolated (stochastic replay vs dynamics change) |
| `--episodes` / `--episodes_per_phase` | How much training is enough to see convergence? |
| `--rep_dim` | Does higher-capacity representation help adaptation? |
| `--temperature` | How sharp is the InfoNCE contrastive objective? |
| `--replay_capacity` | How much history does the agent retain for representation learning? |
| `--approach_probs` | What success probability distribution causes which biases? |

---

## 6. Experiment Design

### Study A

**Research question:** How does stochastic success (direction-dependent reward)
interact with replay buffer composition to bias the ψ-representation?

**Environment:** FourRooms 11×11, goal at (8,8).  The goal cell has four open
neighbours, so all four approach directions are physically reachable.

**Why (8,8) and not (10,10)?**  The corner (10,10) only has two neighbours
(above and left), making the approach-direction analysis degenerate — two
directions are physically impossible.  Cell (8,8) avoids this.

**The three conditions:**

- **A1-baseline:** Deterministic env (no stochastic success).  Shows what ψ
  learns without noise.  Expected: near-optimal success rate, tight ψ ridge
  along the shortest path.

- **A2-allreplay:** Stochastic success.  ALL trajectories (successes and
  failures) go into replay.  The contrastive critic sees the full transition
  structure.  Result: approach distribution is slightly biased by which routes
  happen to reach the goal first.

- **A2-successonly:** Stochastic success.  ONLY successful trajectories go into
  replay.  The replay buffer becomes a biased sample weighted by the success
  probability of each direction.  Result: the representation develops a
  self-reinforcing feedback loop that strongly over-represents the first
  direction to ever succeed.

**What `approach_probs` means:**
```json
{
  "from_above": 0.10,
  "from_left":  0.25,
  "from_right": 0.50,
  "from_below": 0.80,
  "stay":       0.00
}
```
When the agent first steps onto the goal cell, the environment draws a Bernoulli
random variable with this probability.  If it fires, the episode is a "success"
(reward = 1).  If not, the episode ends without reward.  The agent transitions
back to start in either case.

### Study B

**Research question:** When the maze walls change, does the ψ-representation
adapt or stay locked to the old policy?

**Environment:** Three phases, 25 000 episodes each.
1. **FourRooms** — standard layout with four rooms and symmetric doors.
2. **Corridor-shift** — doors shift, forcing a different route to (10,10).
3. **L-wall** — a bottleneck structure that creates a single corridor path.

**B1-baseline:** Static FourRooms for 75 000 episodes.  Shows natural ψ drift
without any forced change.

**B2-changing:** All three phases.  The representation must re-learn the goal
proximity landscape after each wall change.

**Key metrics for Study B:**
- **M9 (recovery time):** How many episodes does it take for success rate to
  recover to 50 % of the pre-transition level?
- **M4 (policy KL):** Does the policy distribution change at phase transitions?
- **M5 (ψ L2 velocity):** How fast does the representation reshape?

---

## 7. Analysing Results

### 7.1 Generate all plots (single seed, default paths)

```bash
python generate_plots.py
```

### 7.2 Multi-seed plots

```bash
python generate_plots.py \
    --results_dir results/study_a \
    --seeds 42 43 44 45 46 \
    --output_dir results/figures
```

Shaded bands show ± 1 standard deviation across seeds.

### 7.3 Only Study A

```bash
python generate_plots.py \
    --condition a1 a2ar a2so \
    --results_dir results/study_a \
    --seeds 42 43
```

### 7.4 Plots produced

| Figure | Content |
|--------|---------|
| `study_a_m1_success_rate.png` | Rolling success rate (3 conditions) |
| `study_a_approach_allreplay.png` | Attempt counts + true vs empirical rates (A2-allreplay) |
| `study_a_approach_successonly.png` | Same for A2-successonly |
| `study_a_approach_over_time.png` | Approach direction fractions over training |
| `study_b_m1_success_rate.png` | B1 vs B2 success rate with phase markers |
| `study_b_m4_adaptation.png` | Policy KL divergence B1 vs B2 |
| `study_b_m5_representation.png` | Mean ψ L2 change rate B1 vs B2 |
| `study_b_m8_alignment.png` | ψ route alignment gap B1 vs B2 |
| `study_a_psi_grid_baseline.png` | 4×4 psi similarity grid over training (A1) |
| `study_a_psi_grid_allreplay.png` | 4×4 psi similarity grid (A2-allreplay) |
| `study_b_psi_grid_changing.png` | 4×4 psi similarity grid, all 3 phases (B2) |
| `study_a_trajectories_baseline.png` | Last 100 episode trajectories (A1) |
| `study_a_trajectories_allreplay.png` | Last 100 episode trajectories (A2-allreplay) |
| `study_b_trajectories_changing.png` | Last 100 trajectories in final phase (B2) |

### 7.5 Reading episode_log.jsonl

Each line is a JSON object:

```json
{"episode": 1234, "phase": 0, "success": true, "steps": 47}
{"episode": 1235, "phase": 0, "success": false, "steps": 200, "approach_dir": "from_above"}
```

Parse with:

```python
import json
with open("episode_log.jsonl") as f:
    log = [json.loads(line) for line in f]
success_rate = sum(e["success"] for e in log) / len(log)
```

### 7.6 Loading ψ snapshots

```python
import numpy as np, json

with open("snapshots/index.json") as f:
    index = json.load(f)

# Load first snapshot
entry = index[0]
psi = np.load(f"snapshots/{entry['psi_file']}")   # (121, rep_dim)
sim = np.load(f"snapshots/{entry['sim_file']}")   # (11, 11)
route = np.load(f"snapshots/{entry['route_file']}")  # (T,) state indices
```

---

## 8. Results Directory Structure

```
results/
├── study_a/
│   ├── A1-baseline/
│   │   ├── seed_42/
│   │   │   ├── run_config.json          # All hyperparameters
│   │   │   ├── run_log.txt              # Timestamped progress log
│   │   │   ├── env_config.json          # Maze configuration
│   │   │   ├── episode_log.jsonl        # One JSON per line: {ep, success, steps}
│   │   │   ├── metrics.json             # Legacy metrics (for old plots)
│   │   │   ├── psi_final.npy            # Final ψ table (121 × rep_dim)
│   │   │   ├── snapshots.pkl            # Legacy snapshots (for old plots)
│   │   │   ├── metrics/
│   │   │   │   ├── all_metrics.json     # Summary of all 9 metrics
│   │   │   │   ├── m1_rolling_success_rate.npy
│   │   │   │   ├── m4_policy_adaptation.json
│   │   │   │   ├── m5_representation_rate.json
│   │   │   │   ├── m8_route_alignment.json
│   │   │   │   └── m9_recovery.json
│   │   │   ├── checkpoints/
│   │   │   │   ├── ep_005000.pt
│   │   │   │   ├── ep_010000.pt
│   │   │   │   ├── ...
│   │   │   │   └── final.pt
│   │   │   └── snapshots/
│   │   │       ├── index.json           # [{episode, phase, psi_file, ...}, ...]
│   │   │       ├── psi_ep000000.npy
│   │   │       ├── sim_ep000000.npy     # (11, 11) similarity map
│   │   │       ├── route_ep000000.npy
│   │   │       ├── policy_ep000000.npy  # (121, 5) policy distribution
│   │   │       └── ...
│   │   ├── seed_43/
│   │   │   └── ...
│   │   └── seed_44/
│   │       └── ...
│   ├── A2-allreplay/
│   │   └── seed_42/
│   │       ├── ...                      # same structure
│   │       ├── approach_stats.json      # per-direction attempt/success counts
│   │       └── metrics/
│   │           ├── m6_suboptimality_gap.npy
│   │           ├── m7_approach_entropy.npy
│   │           └── approach_stats.json
│   └── A2-successonly/
│       └── ...
├── study_b/
│   ├── B1-baseline/
│   │   └── seed_42/
│   │       └── ...
│   └── B2-changing/
│       └── seed_42/
│           ├── ...
│           └── phase_info.json          # {transitions: [25000, 50000], names: [...]}
└── figures/
    ├── study_a_m1_success_rate.png
    ├── study_a_approach_allreplay.png
    ├── study_a_approach_successonly.png
    ├── study_a_approach_over_time.png
    ├── study_b_m1_success_rate.png
    ├── study_b_m4_adaptation.png
    ├── study_b_m5_representation.png
    ├── study_b_m8_alignment.png
    ├── study_a_psi_grid_baseline.png
    ├── study_a_psi_grid_allreplay.png
    ├── study_b_psi_grid_changing.png
    ├── study_a_trajectories_baseline.png
    ├── study_a_trajectories_allreplay.png
    └── study_b_trajectories_changing.png
```

### Checkpoint format

Each `.pt` file is a `torch.save` dict with:

```python
{
    "episode":          int,            # last completed episode
    "model_state_dict": ...,            # nn.Embedding weights
    "optimizer_state_dict": ...,        # Adam state
    "replay_buffer":    list[list[int]],# list of trajectories
    "metrics_state":    dict,           # MetricsCollector.get_state()
    "config":           dict,           # full run config
    "cur_phase":        int,            # current phase index (Study B)
}
```

Load with:

```python
import torch
payload = torch.load("checkpoints/ep_025000.pt", map_location="cpu")
print(f"Checkpoint at episode {payload['episode']}")
```

---

*Last updated: April 2026*
