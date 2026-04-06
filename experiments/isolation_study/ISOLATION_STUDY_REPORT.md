# Isolation Study: Stochastic Success and Changing Dynamics in Single-Goal GCRL

## Motivation

Goal-conditioned reinforcement learning (GCRL) agents trained with successor-feature representations ($\psi$) exhibit a known pattern: they converge to a single route to the goal and stop exploring alternatives. We call this **overexploitation**. The mechanism is straightforward — the $\psi$ representation encodes cumulative discounted state occupancy, so once a successful route is found, the representation sharpens around it, making the policy even more likely to repeat that route. This is a positive feedback loop.

Two natural questions:

1. **What happens when success is stochastic?** If the same route sometimes fails (e.g., a door is randomly blocked), does the agent discover alternative paths, or does it double down on the one it knows?
2. **What happens when the environment changes?** If the layout shifts mid-training, can the agent adapt its representation and policy, or does the old representation trap it?

These questions matter because real-world navigation is rarely deterministic and rarely static. An agent that cannot handle either is fragile.

We designed a controlled isolation study with two sub-experiments (Study A and Study B) to test each factor independently.

---

## Setup

### Environment

All experiments use an 11×11 gridworld. Start state: top-left corner $(0,0)$. Goal state: bottom-right corner $(10,10)$. The agent has 4 actions (up, down, left, right) and receives a step penalty with a terminal reward at the goal. Maximum episode length: 120 steps.

### Agent

The agent learns a successor-feature representation $\psi(s) \in \mathbb{R}^{16}$ via temporal-difference updates. The policy is derived from $\psi$ — it selects actions that move toward states whose representation is most similar to the goal's representation (cosine similarity in $\psi$-space). Key hyperparameters:

| Parameter | Value |
|-----------|-------|
| $\psi$ dimension | 16 |
| Learning rate | 0.01 |
| Batch size | 128 |
| Replay capacity | 2000 |
| Discount ($\gamma$) | 0.99 |
| Entropy coefficient | 0.1 |
| Max steps per episode | 120 |

Policy rollouts use a **stochastic softmax** over $\psi$-similarity (5 rollouts, best trajectory kept). This avoids the greedy-argmax trap where the policy gets permanently stuck in local optima.

### Metrics

- **M1**: Rolling success rate (window=50)
- **M2**: Path diversity — number of distinct trajectory clusters (Jaccard distance, sampled from last 200 trajectories)
- **M3**: Exploitation ratio — fraction of recent trajectories that match the single most common cluster
- **M4**: Policy adaptation index — KL divergence between consecutive $\psi$ snapshots
- **M5**: Representation drift — L2 and cosine distance between consecutive $\psi$ snapshots
- **M7**: Approach direction entropy — Shannon entropy over the distribution of directions from which the agent reaches the goal
- **M8**: Representation-route alignment — gap between mean $\psi$-similarity on-route vs. off-route cells

---

## Study A: Stochastic Success

### Design

The goal is always at $(10,10)$. The environment is a standard FourRooms layout (4 rooms separated by walls with narrow doorways). Crucially, **approach directions have different success probabilities**:

| Direction | Success Probability |
|-----------|-------------------|
| from_below | 0.80 |
| from_right | 0.50 |
| from_left | 0.25 |
| from_above | 0.10 |

When an approach fails, the episode counts as a failure regardless of whether the agent reached the goal cell. This simulates a stochastic environment where some strategies are inherently less reliable.

Three conditions, each run for **10,000 episodes** (seed 42):

- **A1 — Baseline (deterministic)**: Standard FourRooms, no stochastic success. All approaches succeed with probability 1.0.
- **A2 — Stochastic, all-replay**: Stochastic success enabled. All episodes (successes and failures) are added to the replay buffer.
- **A2 — Stochastic, success-only**: Stochastic success enabled. Only successful episodes enter the replay buffer.

### Results

#### Success Rate

<!-- Figure: results/figures/study_a_m1_success_rate.png -->

| Condition | Mean Success Rate | Peak | Final |
|-----------|------------------|------|-------|
| A1 (baseline) | 87.0% | 100% | ~99% (converged) |
| A2 (all-replay) | 16.6% | 38.0% | ~10% |
| A2 (success-only) | 19.4% | 44.0% | ~8% |

A1 converges to near-perfect performance by episode ~3000 and holds. Both stochastic conditions plateau around 15–20% and never improve. The stochastic success mechanism severely caps performance — not just by making individual attempts fail, but by preventing the agent from learning reliable routes in the first place.

#### Approach Direction Lock-in

<!-- Figure: results/figures/study_a_approach_allreplay.png -->
<!-- Figure: results/figures/study_a_approach_successonly.png -->
<!-- Figure: results/figures/study_a_approach_over_time.png -->

This is the most striking result. The approach direction distributions reveal extreme lock-in:

| Condition | from_above | from_left | from_right | from_below |
|-----------|-----------|-----------|------------|------------|
| A2 (all-replay) | 3,256 attempts (10.0% success) | 5,155 attempts (25.8% success) | 0 | 0 |
| A2 (success-only) | 3 attempts (0% success) | 7,700 attempts (25.3% success) | 0 | 0 |

The agent **never discovers** from_right (50% success) or from_below (80% success). These are the two highest-probability approaches, yet the agent never tries them. Instead, it locks into from_left (25%) — a mediocre strategy — and sometimes from_above (10%), which is the worst option.

Under success-only replay, the lock-in is even more extreme: 7,700 of 7,703 attempts come from_left. The agent tried from_above exactly 3 times, never succeeded, and abandoned it.

Why? The $\psi$ representation sharpens around early successes. If the first few successes happen to come from_left, the representation concentrates there, and the policy only generates from_left trajectories from that point on. The agent literally cannot "see" that from_below exists, because the representation near that approach has been pushed to near-zero similarity with the goal.

The all-replay condition is slightly less locked because failure trajectories inject some diversity into the representation. But it is not enough to discover the better approaches.

#### Psi Representation

<!-- Figure: results/figures/study_a_psi_grid_baseline.png -->
<!-- Figure: results/figures/study_a_psi_grid_allreplay.png -->

The $\psi$ grid snapshots show this visually. In A1 (baseline), the representation gradually illuminates the full path from start to goal — high similarity values spread along the route, and off-route cells go dark (low similarity). In A2 (all-replay), only a narrow corridor of cells near the from_left approach direction light up. The bottom-right region (from_below) stays completely dark throughout training.

---

## Study B: Changing Dynamics

### Design

Two conditions, run for **15,000 episodes** total (seed 42):

- **B1 — Static baseline**: Standard FourRooms layout for all 15,000 episodes. Single phase.
- **B2 — Changing environment**: Three phases of 5,000 episodes each, with the wall layout changing between phases:
  - **Phase 0 (ep 0–4999)**: Standard FourRooms — 4 rooms, doors at $(2,5)$, $(8,5)$, $(5,2)$, $(5,8)$
  - **Phase 1 (ep 5000–9999)**: Corridor-shift — doors reduced to $(2,5)$ and $(5,8)$ only, forcing a different bottleneck structure
  - **Phase 2 (ep 10000–14999)**: L-wall — entirely different wall structure with an L-shaped barrier, bottlenecks at $(4,3)$ and $(7,8)$

The goal remains fixed at $(10,10)$ across all phases. The agent must substantially change its policy each time.

### Results

#### Success Rate

<!-- Figure: results/figures/study_b_m1_success_rate.png -->

| Condition | Phase 0 | Phase 1 | Phase 2 | Overall Mean |
|-----------|---------|---------|---------|-------------|
| B1 (static) | 90.9% | (same layout) | (same layout) | 90.9% |
| B2 (changing) | ~75% | ~81% | ~77% | 77.4% |

B1 converges and holds steady. B2 shows a clear **dip-and-recover** pattern at each phase transition. When the walls change at episode 5000, performance drops sharply (from ~100% to ~40%) but recovers within ~1000–2000 episodes. The same happens at episode 10000. The agent does adapt — but not instantly, and not completely.

#### Policy Adaptation (M4 — KL Divergence)

<!-- Figure: results/figures/study_b_m4_adaptation.png -->

The M4 metric makes the phase transitions vivid. B1's KL divergence is near-flat after the initial learning period (~0.03–0.06). B2 shows a massive spike at the phase 1→2 boundary (episode 10000): KL divergence jumps to **0.52**, then decays back to baseline within ~1000 episodes. The phase 0→1 transition at episode 5000 shows a smaller spike (~0.14).

The asymmetry is notable. The transition from corridor_shift to l_wall is a more dramatic restructuring than fourrooms to corridor_shift, and the KL spike reflects this — the agent must overhaul its policy more aggressively.

#### Representation Drift (M5) and Alignment (M8)

<!-- Figure: results/figures/study_b_m5_representation.png -->
<!-- Figure: results/figures/study_b_m8_alignment.png -->

M5 (representation drift) tracks the rate at which $\psi$ changes between snapshots. In B2, we see spikes at phase transitions — the representation is being restructured. In B1, the rate decays monotonically as the representation stabilizes.

M8 (alignment gap between on-route and off-route $\psi$ values) shows that in B2, the representation realigns to new routes after each transition. The alignment gap fluctuates more in B2 than B1, reflecting the ongoing tension between old and new route information in the representation.

#### Psi Grid with Policy Overlay

<!-- Figure: results/figures/study_b_psi_grid_changing.png -->

The $\psi$ grids tell the full story. In phase 0, the representation learns the standard fourrooms path. At the phase 1 boundary (corridor_shift), the representation partially resets — old high-similarity cells fade, and new cells along the forced corridor brighten. At the phase 2 boundary (l_wall), the restructuring is even more dramatic: the entire left side of the grid lights up as the agent discovers the L-shaped passage.

The policy overlay (blue lines) shows the learned route at each snapshot. The routes clearly shift between phases, confirming that the agent genuinely adapts rather than getting stuck on the old path.

#### Trajectory Plots

<!-- Figure: results/figures/study_b_trajectories_changing.png -->

The final-phase trajectories for B2 show the agent navigating the l_wall layout. Most trajectories are successful (green), traveling down the left corridor and across the bottom. The few failures (red/brown) tend to get stuck in the upper-left corner, suggesting occasional residual interference from old representations.

---

## Summary of Findings

| Question | Answer |
|----------|--------|
| Does stochastic success promote exploration of better strategies? | **No.** The agent locks into the first approach that works, even when much better alternatives exist. Success-only replay makes this worse. |
| Can the agent adapt when the environment changes? | **Yes, partially.** The agent recovers from layout changes within ~1000–2000 episodes but shows temporary performance drops. Larger structural changes cause larger disruption. |
| Does the $\psi$ representation help or hurt adaptability? | **Both.** It enables fast initial learning but creates strong inertia. Once a route is encoded, the representation resists change. Environmental forcing (changing walls) can overcome this inertia; stochastic success alone cannot. |

The core mechanism is the same in both studies: the $\psi$ representation acts as a positive feedback loop. Early successes shape the representation, which shapes the policy, which shapes future experiences, which reinforce the representation. Breaking this loop requires an external force (changing environment structure). Internal randomness (stochastic success) is not enough — the agent absorbs the randomness and doubles down on its initial strategy.

---

## Conclusions

1. **Stochastic success does not cure overexploitation.** Even with only 25% success probability on the locked-in route, the agent never explores alternatives with up to 80% success. The representation feedback loop is too strong.

2. **Success-only replay amplifies lock-in.** Removing failure trajectories from replay eliminates the only source of representational diversity. The agent under success-only replay tried from_above exactly 3 times in 10,000 episodes.

3. **Environmental changes force adaptation.** When walls physically change, the agent must adapt — and it does, with a characteristic dip-and-recover pattern. The KL divergence spike at transitions quantifies how much the policy must change.

4. **Adaptation speed depends on structural similarity.** The fourrooms→corridor_shift transition (small change) recovers faster than corridor_shift→l_wall (large change). This suggests the representation carries structural inertia proportional to how different the new layout is.

5. **The $\psi$ representation is a double-edged sword.** It enables efficient learning but creates a brittleness that purely exploration-based methods would not have. Any practical deployment of $\psi$-based GCRL agents needs an explicit mechanism to counteract representational lock-in.

---

## Reproduction

All code is in `experiments/isolation_study/`. Run individual conditions with:

```bash
python run_single.py --condition A1-baseline --seed 42
python run_single.py --condition A2-allreplay --seed 42
python run_single.py --condition A2-successonly --seed 42
python run_single.py --condition B1-baseline --seed 42
python run_single.py --condition B2-changing --seed 42
```

Generate all figures:

```bash
python generate_plots.py
```

Results are saved in `results/` with per-condition subdirectories containing metrics, configs, $\psi$ snapshots, and approach statistics.
