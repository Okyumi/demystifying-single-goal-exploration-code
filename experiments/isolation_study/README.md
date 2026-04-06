# Isolation Study — Systematic Factor Isolation Experiments for SGCRL

This experiment suite isolates two factors that affect SGCRL performance:
1. **Stochastic success** (Study A) — same maze, same dynamics, but success is probabilistic
2. **Changing dynamics** (Study B) — same goal, but wall layout changes across phases

## Quick Start

```bash
# Run Study A (all 3 conditions: baseline, allreplay, successonly)
python experiments/isolation_study/run_study_a.py --seed 42

# Run Study B (both conditions: baseline, changing)
python experiments/isolation_study/run_study_b.py --seed 42

# Generate comparison plots
python experiments/isolation_study/compare_results.py --study both --seed 42
```

## Study A: Stochastic Success Isolation

**Question:** Does stochastic success change (a) which path the agent converges to, (b) how the representation evolves, (c) whether the agent discovers the optimal approach direction?

**Setup:** FourRooms maze (11x11), goal at (10,10), 3000 episodes, max_steps=120.

| Condition | Success Model | Description |
|-----------|--------------|-------------|
| A1-baseline | Deterministic | Reaching the goal = success (p=1.0 from any direction) |
| A2-stochastic-allreplay | Direction-dependent, all replay | from_above=0.10, from_left=0.25, from_right=0.50, from_below=0.80. ALL trajectories in replay buffer. |
| A2-stochastic-successonly | Direction-dependent, success-only replay | Same probabilities. Only successful trajectories in replay buffer. |

**Key insight:** The two stochastic variants test whether replay buffer composition matters. AllReplay lets the contrastive critic learn from all transitions (better state reachability estimates). SuccessOnly only learns from "winning" trajectories (biased but reward-relevant).

**What to look for:**
- Does the baseline converge to a single route? (M3 exploitation ratio)
- Do stochastic variants find the optimal approach direction (from_below, p=0.80)? (M6 suboptimality gap)
- Does AllReplay learn better representations than SuccessOnly? (M5, M8)
- Does the agent lock onto the first partially-successful direction? (M7 approach entropy)

## Study B: Changing Dynamics Isolation

**Question:** Does the agent's representation and policy adapt when the dynamics change? How fast? Does it overexploit the old route?

**Setup:** Goal fixed at (10,10), 3000 total episodes, max_steps=120.

| Condition | Dynamics | Description |
|-----------|---------|-------------|
| B1-baseline | Static FourRooms | Same FourRooms for all 3000 episodes |
| B2-changing | 3 phases x 1000 ep | Phase 0: FourRooms, Phase 1: Corridor-shift, Phase 2: L-wall |

**Phase designs:**
- **Phase 0 (FourRooms):** Horizontal wall row 5 (doors at cols 2,8), vertical wall col 5 (doors at rows 2,8)
- **Phase 1 (Corridor-shift):** Horizontal wall row 5 (door at col 2 only), vertical wall col 5 (door at row 8 only). Forces bottom-left route.
- **Phase 2 (L-wall):** Vertical wall col 3 (door at row 4), horizontal wall row 7 (door at col 8). Forces tight L-shaped detour.

**What to look for:**
- Does the baseline's success rate plateau smoothly? (M1)
- Does the changing condition show drops at transitions? (M1, M9 recovery time)
- Do policy and representation adapt at transitions? (M4 KL spikes, M5 L2 spikes)
- Does the representation lock onto the old route after transitions? (M8 alignment)

## Agent Configuration

```
rep_dim = 16
lr_psi = 0.01
batch_size = 128
replay_capacity = 2000
max_steps = 120
gamma = 0.99
entropy_coeff = 0.1
episodes_per_update = 1
normalize = True
```

## Metrics

See [METRICS_DOCUMENTATION.md](METRICS_DOCUMENTATION.md) for full details on all 9 metrics.

| Metric | Name | Study A | Study B |
|--------|------|---------|---------|
| M1 | Success Rate (rolling window=50) | Yes | Yes |
| M2 | Path Diversity (Jaccard + clustering) | Yes | Yes |
| M3 | Exploitation Ratio | Yes | Yes |
| M4 | Policy Adaptation Index (KL divergence) | Yes | Yes |
| M5 | Representation Adaptation Rate (psi L2 velocity) | Yes | Yes |
| M6 | Suboptimality Gap | Yes | No |
| M7 | Approach Direction Entropy | Yes | No |
| M8 | Representation-Route Alignment | Yes | Yes |
| M9 | Phase-Transition Recovery Time | No | Yes |

## Visualizations

### V1: psi-similarity heatmap WITH policy route overlay
The key visualization. Background shows psi(s).psi(g) heatmap (RdYlGn colormap, walls=gray). Overlay shows the greedy policy trajectory as a thick blue line with arrows. Generated every 25 episodes.

### V2: Trajectory overlay
Green=success, red=failure trajectories on the maze grid. One per phase.

### V3: Metric comparison plots
For each metric, baseline and treatment plotted on the same axes (blue=baseline, orange=treatment). Phase transition lines drawn for Study B.

### V4: psi-similarity evolution video
MP4 at 6fps showing heatmap + greedy route + episode label + phase label per frame.

### V5: Approach direction analysis (Study A only)
Bar charts of attempts per direction, true vs empirical success rates, rolling approach direction preference.

## Output Structure

```
results/
  study_a_seed_42/
    A1-baseline/
      config.json, metrics.json, env_config.json, psi_final.npy
      psi_evolution.mp4
      figures/
        success_rate.png, psi_similarity_grid.png, trajectories_last200.png
    A2-stochastic-allreplay/
      ... (same + approach_history.json, approach_direction_stats.png, approach_over_time.png)
    A2-stochastic-successonly/
      ...
    comparisons/
      m1_success_rate_vs_allreplay.png, m1_success_rate_vs_successonly.png
      m3_exploitation_ratio_vs_*.png, m4_m5_adaptation_vs_*.png
      m6_suboptimality_gap_*.png, m7_approach_entropy_*.png
      m8_alignment_vs_*.png, m1_allreplay_vs_successonly.png
  study_b_seed_42/
    B1-baseline/
      ...
    B2-changing/
      ...
    comparisons/
      m1_success_rate.png, m2_path_diversity.png, m3_exploitation_ratio.png
      m4_m5_adaptation.png, m8_alignment.png, m9_recovery.png
```

## Interpretation Guide

### Study A: What to expect
- **Baseline** should converge quickly to a single route with high success rate (~0.7-0.9)
- **Stochastic variants** will likely lock onto the first discovered approach direction (often from_left, p=0.25) and never explore from_below (p=0.80)
- **AllReplay** may have slightly different representation dynamics than SuccessOnly since it learns from all transitions
- The suboptimality gap (M6) should remain non-zero if the agent overexploits

### Study B: What to expect
- **Baseline** should show smooth learning and eventual plateau
- **Changing** should show drops at episode 1000 and 2000 (phase transitions)
- M4 (KL) and M5 (L2) should spike at transitions if the agent adapts
- M9 recovery time tells you how many episodes the agent needs to recover
- If the agent fails to adapt, M1 drops to zero and stays there

### Common patterns indicating SGCRL limitations
1. **Overexploitation:** M3 near 1.0, M2 entropy near 0, M8 gap large
2. **Representation lock-in:** M5 flat at transitions, M8 gap doesn't decrease
3. **Policy inertia:** M4 flat at transitions (agent keeps old policy)
4. **Direction lock-in (Study A):** M7 entropy drops fast, M6 gap persists
