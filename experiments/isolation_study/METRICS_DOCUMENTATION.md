# Metrics Documentation — Isolation Study

This document provides comprehensive documentation for all 9 metrics (M1-M9) used in the isolation study experiments.

## Overview

The metrics are designed to answer two key questions:
1. **Does SGCRL overexploit** when success is stochastic? (Study A)
2. **Does SGCRL adapt** when dynamics change? (Study B)

---

## M1: Success Rate (Rolling Window)

**What it measures:** Fraction of episodes where the agent reaches the goal (deterministic) or achieves stochastic success, computed over a sliding window of 50 episodes.

**Why it's useful:** The most fundamental performance metric. Provides a smooth view of learning progress, revealing both learning speed and steady-state capability.

**How to interpret:**
- Higher = better performance
- Drops after phase transitions indicate the cost of adapting to new dynamics
- Plateaus indicate convergence to a stable policy
- A flat zero means the agent is completely stuck

**What behavior it captures:** Learning speed, adaptation speed after dynamics changes, steady-state performance level.

**Implementation:** Rolling convolution with a uniform kernel of width 50. For sequences shorter than the window, uses cumulative average.

---

## M2: Path Diversity (Jaccard + Clustering)

**What it measures:** Three sub-metrics computed on successful trajectories within each phase:
1. **Mean Jaccard distance** — average pairwise distance between trajectory cell-sets
2. **Number of route clusters** — distinct groups found by single-linkage clustering (threshold=0.5)
3. **Shannon entropy** — entropy over the cluster size distribution

**Why it's useful:** Reveals whether the agent explores multiple viable paths or locks onto one. Combined with M3, provides a complete picture of exploration-exploitation balance.

**How to interpret:**
- High mean Jaccard = diverse paths being used
- Many clusters = multiple distinct routes discovered
- High entropy = even usage across routes
- All low = one dominant path (overexploitation)

**What behavior it captures:** Route variety, exploration breadth, whether the agent discovers alternative strategies.

**Implementation:** Trajectories are converted to cell-sets (frozenset of visited states). Jaccard distance = 1 - |A∩B|/|A∪B|. Single-linkage agglomerative clustering with distance threshold 0.5.

---

## M3: Exploitation Ratio

**What it measures:** Fraction of successful trajectories in the largest route cluster.

**Why it's useful:** Directly quantifies how much the agent relies on a single route. A value close to 1.0 means near-total overexploitation.

**How to interpret:**
- Close to 1.0 = extreme overexploitation (one dominant path)
- 0.5 = two balanced routes
- Lower = more distributed usage across multiple paths
- 0.0 = no successful trajectories

**What behavior it captures:** Route preference, overexploitation severity, whether the agent has discovered multiple viable paths but prefers one.

**Implementation:** Uses the same clustering as M2. Exploitation ratio = max(cluster_sizes) / sum(cluster_sizes).

---

## M4: Policy Adaptation Index (KL Divergence)

**What it measures:** KL(pi_t || pi_{t-delta}) — KL divergence between the current policy and a snapshot from delta episodes ago, computed as the mean KL divergence across all states.

**Why it's useful:** Measures how much the actor's behavior changed between snapshots. Essential for detecting whether the agent adapts its behavior after dynamics changes.

**How to interpret:**
- Spikes at phase transitions = actor is actively adapting its behavior (good)
- Flat within phases = stable, converged policy (normal)
- Flat at transitions = stuck on old policy (problematic — agent isn't responding to changes)
- Continuously high = unstable learning

**What behavior it captures:** Policy adaptation to dynamics changes, learning stability, responsiveness to environmental changes.

**Implementation:** Policy pi(a|s) is computed for all states at each snapshot. KL divergence is computed element-wise and averaged across states: mean_s[ sum_a pi_new(a|s) * log(pi_new(a|s) / pi_old(a|s)) ].

---

## M5: Representation Adaptation Rate (psi L2 Velocity)

**What it measures:** Mean L2 distance between psi(s) at time t and psi(s) at time t-delta, averaged over all states. Also computes cosine distance as a complementary measure.

**Why it's useful:** Measures how fast the critic is reshaping the representation landscape. Complementary to M4 — M4 measures policy change while M5 measures the underlying representation change that drives it.

**How to interpret:**
- High early = active learning, representations being shaped
- Decreasing over time = settling, representations converging
- Spike at phase transition = re-learning triggered (good — representation is plastic)
- No spike at transition = stuck representation (bad — representations don't adapt)

**What behavior it captures:** Representation plasticity, critic adaptation speed, whether the contrastive learning is responsive to new data.

**Implementation:** Full psi embedding tables are snapshot at regular intervals. L2 distance = mean over all states of ||psi_new(s) - psi_old(s)||_2.

---

## M6: Suboptimality Gap (Stochastic Experiments Only)

**What it measures:** Difference between the optimal approach direction's success rate and the agent's actual approach direction's success rate, computed over a rolling window.

**Why it's useful:** Directly measures whether the agent has found the best strategy under stochastic success. A gap of zero means the agent consistently uses the highest-probability approach direction.

**How to interpret:**
- 0 = optimal (agent always uses the best direction)
- >0 = agent is using a worse direction (suboptimal)
- The gap equals p_best - p_agent_route
- Decreasing gap = agent is learning to find better directions
- Stable non-zero gap = permanently locked into suboptimal strategy

**What behavior it captures:** Whether the agent converges to the optimal strategy under stochastic success, the cost of overexploitation in terms of missed success probability.

**Implementation:** For each episode, the true success probability of the approach direction used is compared to the maximum available probability. Gap = max_prob - used_prob, smoothed with rolling window.

---

## M7: Approach Direction Entropy (Stochastic Experiments Only)

**What it measures:** Shannon entropy over the distribution of approach directions used by the agent, computed over a rolling window of 100 episodes.

**Why it's useful:** Quantifies exploration breadth specifically for the approach direction choice. High entropy means the agent is trying many directions; low entropy means it's locked onto one.

**How to interpret:**
- Maximum entropy (~1.386 = log(4)) = uniform exploration of all 4 directions
- 0 = always same direction (complete lock-in)
- Decreasing over time = convergence to a strategy (may be good or bad depending on which direction)
- Sudden drops = agent locked in after finding partial success

**What behavior it captures:** Exploration breadth, sensitivity to stochastic success signals, whether the agent systematically tries alternatives.

**Implementation:** Over a rolling window, counts the fraction of each direction used and computes Shannon entropy: -sum(p * log(p)).

---

## M8: Representation-Route Alignment

**What it measures:** For each snapshot, computes the mean psi-similarity of states ON the greedy policy trajectory vs states NOT on it. The alignment score is the difference (on_route - off_route).

**Why it's useful:** Shows whether the psi-similarity landscape has formed a "ridge" along the current route. This is the most direct measure of representation-behavior coupling in SGCRL.

**How to interpret:**
- Large positive gap = strong psi-trace along the route (representation locked to path)
- Small gap = flat landscape (exploration phase or no meaningful learning)
- Negative gap = representation favors alternative paths over the greedy route
- Increasing gap over time = representation becoming more specialized to current route

**What behavior it captures:** How tightly the representation is locked to a specific route, representation-behavior coupling, whether the contrastive learning creates path-specific features.

**Implementation:** The greedy policy is rolled out from start to produce a trajectory. States are partitioned into on-route and off-route sets. Mean psi(s).psi(g) is computed for each set. Gap = mean_on - mean_off.

---

## M9: Phase-Transition Recovery Time

**What it measures:** Number of episodes after a phase transition until the rolling success rate recovers to 50% of the pre-transition level.

**Why it's useful:** Provides a concrete, intuitive measure of adaptation speed. "How many episodes does it take to get back to half its previous performance?"

**How to interpret:**
- Short recovery (e.g., <50 episodes) = fast adaptation
- Long recovery (e.g., >200 episodes) = slow adaptation
- Never recovered (None) = agent completely stuck after transition
- Compare across transitions to see if adaptation gets harder over time

**What behavior it captures:** Agent's practical ability to recover from dynamics changes, representation plasticity in terms of real-world performance impact.

**Implementation:** Pre-transition level is computed as the mean rolling success rate over the window before the transition. Target = 50% of that level. Scans forward from the transition point until the rolling rate meets or exceeds the target.

---

## Metric Relationships

| Metric | Best paired with | Relationship |
|--------|-----------------|--------------|
| M1 | M9 | M1 shows the raw curve, M9 quantifies recovery |
| M2 | M3 | M2 shows diversity, M3 shows dominance of the top route |
| M4 | M5 | M4 = policy change, M5 = representation change that drives it |
| M6 | M7 | M6 = cost of wrong direction, M7 = exploration breadth |
| M8 | M3 | M8 = representation lock-in, M3 = behavioral lock-in |
| M5 | M8 | M5 = rate of change, M8 = direction of change (towards route) |

## Study-Specific Metrics

| Metric | Study A (Stochastic) | Study B (Changing Dynamics) |
|--------|---------------------|---------------------------|
| M1 | Yes | Yes |
| M2 | Yes | Yes |
| M3 | Yes | Yes |
| M4 | Yes | Yes |
| M5 | Yes | Yes |
| M6 | Yes (stochastic only) | No |
| M7 | Yes (stochastic only) | No |
| M8 | Yes | Yes |
| M9 | No (no phase transitions) | Yes (changing only) |
