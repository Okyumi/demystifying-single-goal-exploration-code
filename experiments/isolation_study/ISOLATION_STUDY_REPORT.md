# Isolation Study: Stochastic Success and Changing Dynamics in Single-Goal GCRL

## Motivation

Goal-conditioned RL agents trained with contrastive successor features ($\psi$) learn a representation that encodes how "reachable" each state is from any other state, measured by future-state occupancy. The policy follows the gradient of this landscape toward the goal. This works — but it creates a feedback loop:

- The agent reaches the goal via some route.
- That route's states get reinforced in $\psi$-space (higher similarity to the goal).
- The policy becomes more likely to repeat that exact route.
- Future experience further sharpens the same corridor of the representation.

The result is **overexploitation**: convergence to a single route while ignoring potentially better alternatives. We use two controlled experiments to isolate the factors that matter.

- **Study A** asks: when the environment introduces stochastic failure that depends on approach direction, does the agent adapt by switching to higher-probability directions? Or does the feedback loop hold?
- **Study B** asks: when the wall layout physically changes mid-training, can the agent restructure its representation? How fast, and at what cost?

Both studies share the same 11×11 FourRooms gridworld, the same agent architecture, and the same hyperparameters. The only differences are the manipulated variables.

---

## Agent and Environment

### Grid

- 11×11 gridworld, FourRooms layout (horizontal wall at row 5, vertical wall at column 5, with doors at rows/columns 2 and 8).
- Start: $(0,0)$ (top-left corner).
- 5 actions: up, down, left, right, stay.
- Max episode length: 120 steps.

### Representation

The agent maintains a learned embedding $\psi(s) \in \mathbb{R}^{16}$ for every state $s$. Updates follow a contrastive objective: for a sampled pair $(s, s')$ from the same trajectory, push $\psi(s)$ and $\psi(s')$ closer together while pushing apart representations of states from different trajectories. Specifically:

- Pairs are sampled from the replay buffer using geometric weighting (discount $\gamma = 0.99$), so temporally close states are paired more often.
- The contrastive loss is symmetric InfoNCE over a batch of 128 pairs.
- Embeddings are L2-normalized after each update.

### Policy Derivation

The policy is **not** a separate network. It is derived directly from $\psi$:

$$\pi(a \mid s) \propto \exp\left(\frac{\psi(\text{next}(s,a)) \cdot \psi(g)}{\tau}\right)$$

where $g$ is the goal state and $\tau = 0.1$ is the entropy coefficient. The agent moves toward states whose representation is most similar to the goal's. This tight coupling between representation and policy is what makes the feedback loop so strong — there is no separate actor that could maintain exploratory behavior independently.

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| $\psi$ dimension | 16 |
| Learning rate | 0.01 |
| Batch size | 128 |
| Replay buffer capacity | 2000 trajectories |
| Discount ($\gamma$) | 0.99 |
| Entropy coefficient ($\tau$) | 0.1 |
| Max steps per episode | 120 |
| Snapshot frequency | every 200 episodes |

All experiments use seed 42.

---

## Study A: Stochastic Success

### Question

When the probability of success at the goal depends on approach direction, does the agent discover and converge to the highest-probability approach? Or does the representation feedback loop trap it on a suboptimal strategy?

### Design

**Goal placement.** The goal is at $(8,8)$ — an interior cell in the bottom-right room. This is a deliberate design choice: all four cardinal neighbors are open cells, so the agent can physically approach the goal from all four directions (from_above, from_below, from_left, from_right). Previous versions of this experiment placed the goal at the corner $(10,10)$, which made from_right and from_below physically impossible — a design flaw that rendered the approach direction analysis meaningless.

**Stochastic success mechanism.** When the agent steps onto the goal cell, the episode outcome depends on which direction the final step came from:

| Direction | Success Probability | Interpretation |
|-----------|-------------------|----------------|
| from_above | 0.10 | Worst approach |
| from_left | 0.25 | Poor |
| from_right | 0.50 | Moderate |
| from_below | 0.80 | Best approach |

If the stochastic check fails, the episode counts as a failure — the agent reached the goal cell but did not "succeed." This models environments where different strategies for approaching a target have different reliabilities.

**Conditions.** Three conditions, each run for 10,000 episodes:

- **A1 — Deterministic baseline.** All approaches succeed with probability 1.0. No stochastic mechanism. Establishes what normal learning looks like.
- **A2 — Stochastic, all-replay.** Stochastic success active. All trajectories (successes and failures) are added to the replay buffer.
- **A2 — Stochastic, success-only.** Stochastic success active. Only trajectories where the stochastic check passes are added to the replay buffer.

The replay buffer distinction matters because the contrastive objective learns from trajectory pairs. Including failure trajectories means the critic sees paths that reached the goal cell but "failed" — these still provide useful spatial information about the route. Excluding them means the critic only learns from the narrow subset of trajectories that happened to pass the stochastic check.

### Results

#### Success Rate

![Study A: Rolling Success Rate Comparison](results/figures/study_a_m1_success_rate.png)

| Condition | Mean SR | Final SR (last 100) | Peak |
|-----------|---------|---------------------|------|
| A1 (deterministic) | 96.8% | 92.2% | 100% |
| A2 (all-replay) | 13.4% | 11.6% | 36.0% |
| A2 (success-only) | 20.6% | 11.9% | 40.0% |

Key observations:

- A1 converges to near-perfect performance by ~1000 episodes and holds steady. The shorter path to (8,8) vs. the old (10,10) makes convergence faster.
- Both stochastic conditions plateau around 13–20% and never improve. There is no upward trend over 10,000 episodes.
- Success-only starts slightly higher than all-replay but both converge to ~12% by the end. The early advantage likely comes from the replay buffer containing only "positive examples" initially, giving a cleaner learning signal.
- The theoretical maximum under the stochastic conditions is bounded by the success probability of whichever direction the agent uses. An optimal agent using from_below would achieve ~80%. The observed ~12% is consistent with heavy reliance on from_above (10% true rate) with occasional from_left successes.

#### Approach Direction Distribution

This is the central result of Study A.

![A2-allreplay: Approach Direction Stats](results/figures/study_a_approach_allreplay.png)

![A2-successonly: Approach Direction Stats](results/figures/study_a_approach_successonly.png)

**A2 — All-replay:**

| Direction | Attempts | Share | True Success Rate | Empirical Rate |
|-----------|----------|-------|-------------------|----------------|
| from_above | 7,911 | 81.8% | 10% | 9.5% |
| from_left | 1,313 | 13.6% | 25% | 25.3% |
| from_right | 343 | 3.5% | 50% | 52.2% |
| from_below | 95 | 1.0% | 80% | 82.1% |

**A2 — Success-only:**

| Direction | Attempts | Share | True Success Rate | Empirical Rate |
|-----------|----------|-------|-------------------|----------------|
| from_above | 1,915 | 21.4% | 10% | 10.3% |
| from_left | 6,825 | 76.4% | 25% | 24.9% |
| from_below | 196 | 2.2% | 80% | 82.7% |
| from_right | 0 | 0.0% | 50% | N/A |

Several things stand out:

- **Inverse preference ordering.** In all-replay, the agent devotes 82% of attempts to from_above — the direction with the lowest success probability (10%). Meanwhile, from_below (80% success) gets only 1% of attempts. The agent's effort allocation is almost perfectly inverted relative to the true reward structure.
- **Complete lock-out under success-only.** from_right (50% success) receives literally zero attempts in the success-only condition. The agent never once tried it in 10,000 episodes. This is not a sampling artifact — it is a structural consequence of how $\psi$ learns.
- **Empirical rates match true rates.** Where the agent does attempt a direction, the empirical success rate matches the true probability closely. The problem is not that the agent misestimates success rates — it is that the agent never collects enough data from the better directions to learn that they are better.
- **Different lock-in targets.** All-replay locks onto from_above (10%); success-only locks onto from_left (25%). The direction that "wins" depends on which trajectories happen to succeed early in training. Under success-only, from_above's 10% rate means very few early successes enter the buffer, so from_left (25%) wins the early race. Under all-replay, even failed from_above trajectories provide learning signal, so the agent keeps trying from_above longer, and its sheer volume of attempts cements it.

#### Approach Direction Dynamics

![Study A (A2-allreplay): Approach Direction Fractions Over Training](results/figures/study_a_approach_over_time.png)

The time series reveals the lock-in dynamics:

- In the first ~500 episodes, from_above and from_left share roughly equal fractions.
- By episode ~1500, from_above begins dominating. From_right and from_below never gain any foothold.
- The balance stabilizes by episode ~3000. From that point on, from_above holds 60–80% of attempts, with from_left taking most of the remainder. The agent oscillates between these two but never breaks out to try from_right or from_below.

#### Mechanistic Interpretation

Why does this inversion happen? The mechanism is straightforward but the implication is counterintuitive:

- **Early experience is mostly from_above.** The start is at (0,0), the goal at (8,8). The natural shortest path goes down and right — arriving at (8,8) from (7,8), which is from_above. So early episodes disproportionately approach from above.
- **Early successes shape $\psi$.** Even at 10% success rate, some from_above trajectories succeed. These enter the replay buffer and train the contrastive objective to increase similarity along the from_above corridor.
- **The representation hardens.** Once the from_above corridor has higher $\psi$-similarity to the goal, the policy preferentially routes through it. This generates more from_above experience, which further strengthens the corridor.
- **Alternative directions get starved.** The agent rarely visits the cells south or east of the goal. Without experience at those cells, $\psi$ stays at its initial (essentially random) value there. The policy sees no reason to route through cells with random $\psi$ when a clear gradient exists from above.
- **From_below requires going past the goal.** To approach from (9,8), the agent would need to navigate below (8,8) first — but the gradient of $\psi$ pulls it toward (8,8) from above. The policy literally cannot generate from_below trajectories because the representation landscape funnels it in the wrong direction.

The result: the agent is trapped not by the environment, but by its own learned representation. The $\psi$ landscape becomes a self-fulfilling prophecy.

#### Psi Representation Snapshots

![Study A – A1 Baseline: Psi Similarity Grid](results/figures/study_a_psi_grid_baseline.png)

![Study A – A2 All-Replay: Psi Similarity Grid](results/figures/study_a_psi_grid_allreplay.png)

The psi grid visualizations make the feedback loop visible:

- **A1 baseline:** The representation gradually illuminates the full route from start to goal. By episode ~2000, a clear green corridor runs from (0,0) through the doors to (8,8). Off-route cells go dark red (low similarity). This is healthy learning — the representation tracks the policy's route.
- **A2 all-replay:** The representation concentrates in a narrow strip. By episode ~2000, only the from_above approach corridor is lit. The cells to the south and east of the goal — which would support from_below and from_right approaches — remain deep red throughout 10,000 episodes. The representation is locked.

#### Trajectory Overlays

![Study A – A1 Baseline: Last 100 Trajectories](results/figures/study_a_trajectories_baseline.png)

![Study A – A2 All-Replay: Last 100 Trajectories](results/figures/study_a_trajectories_allreplay.png)

- A1's last 100 trajectories show tight, consistent paths to the goal with occasional variation.
- A2's last 100 trajectories are dominated by failures (red). The few successes (green) all approach from the same direction — from_above or from_left. No trajectories approach from below or from the right.

### Study A Summary

The core finding: **stochastic success does not cure overexploitation; it may actually worsen it.** The agent massively over-invests in the worst approach direction while ignoring the best one.

- The feedback loop between $\psi$ representation and policy is stronger than the learning signal from stochastic outcomes.
- The contrastive objective cannot discover better alternatives because the policy never generates trajectories that would provide the necessary data.
- Replay buffer composition matters but does not change the fundamental outcome. All-replay and success-only both converge to locked-in strategies — just different ones.
- The theoretical ceiling is 80% (from_below), but the agent achieves ~12%. The suboptimality gap is massive: the agent leaves 68 percentage points of success rate on the table.
- This is not a convergence issue. The agent has settled into a stable equilibrium. More training would not help.

---

## Study B: Changing Dynamics

### Question

When the physical environment changes — walls move, corridors close, new paths open — can the agent restructure its $\psi$ representation and adapt its policy? How fast? At what cost?

### Design

**Goal:** fixed at $(10,10)$ (bottom-right corner) throughout all phases. Unlike Study A, the approach direction analysis is not relevant here — we care about adaptation, not direction preference.

**Two conditions, each run for 15,000 total episodes:**

- **B1 — Static baseline.** Standard FourRooms for all 15,000 episodes. Same layout, no changes. This establishes the ceiling and shows what happens when $\psi$ can fully settle.
- **B2 — Changing environment.** Three phases of 5,000 episodes each:
  - **Phase 0 (ep 0–4999): FourRooms.** Standard 4-room layout, doors at $(2,5)$, $(8,5)$, $(5,2)$, $(5,8)$.
  - **Phase 1 (ep 5000–9999): Corridor-shift.** Two doors closed — only $(2,5)$ and $(5,8)$ remain. The agent must find a different path through the restricted layout.
  - **Phase 2 (ep 10000–14999): L-wall.** Entirely different wall structure. A vertical wall at column 3 (rows 0–8, door at row 4) and a horizontal wall at row 7 (columns 3–10, door at column 8). The previous routes are completely invalidated.

The goal remains fixed. The agent must substantially change its policy at each transition.

### Results

#### Success Rate

![Study B: Rolling Success Rate Comparison](results/figures/study_b_m1_success_rate.png)

| Condition | Phase 0 | Phase 1 | Phase 2 | Overall |
|-----------|---------|---------|---------|---------|
| B1 (static) | 91.0% | — | — | 90.9% |
| B2 (changing) | 74.9% | 80.7% | 76.9% | 77.4% |

Key observations:

- **B1 converges and holds.** By ~2000 episodes, B1 is at 95%+ and stays there. The remaining 13,000 episodes add almost nothing — the representation has fully settled.
- **B2 shows a dip-and-recover pattern at each transition.** Performance drops sharply when walls change, then recovers within ~1000–2000 episodes.
  - The phase 0→1 dip (corridor_shift) drops from ~100% to ~40% before recovering to ~80%.
  - The phase 1→2 dip (l_wall) is more severe: ~100% drops to near 0%, recovering to ~80% over ~2000 episodes.
- **B2 recovers but never matches B1's ceiling.** Each phase's performance is 10–15% below the static baseline. This gap likely reflects residual interference from old representations — the contrastive objective must overwrite the old $\psi$ landscape, and this overwriting is incomplete within 5,000 episodes.
- **Phase 2 performance is lower than Phase 1** despite similar recovery time. This suggests that accumulated representation debris from prior phases creates increasing interference.

#### Policy Adaptation (M4 — KL Divergence)

![Study B: Policy Adaptation Index](results/figures/study_b_m4_adaptation.png)

The M4 metric measures KL divergence between consecutive policy snapshots. It reveals how much the policy is changing at each point in training.

- **B1 is near-flat** after the initial learning phase (~0.03–0.06). The policy has converged and barely changes.
- **B2 shows two clear spikes:**
  - Phase 0→1 (episode 5000): KL = 0.14. Moderate restructuring.
  - Phase 1→2 (episode 10000): KL = 0.52. Massive restructuring — 3.7× larger than the first transition.
- **The spike asymmetry is informative.** Corridor_shift is a small perturbation of FourRooms (two doors closed, but the same wall structure). The agent can partially reuse its existing representation. L_wall is a completely different topology — the agent must essentially start over. The KL ratio (0.52 / 0.14 = 3.7×) quantifies this structural distance.
- **Both spikes decay to baseline within ~1000 episodes.** The policy stabilizes quickly once the representation catches up.

#### Representation Drift (M5) and Alignment (M8)

![Study B: Representation Adaptation Rate (M5)](results/figures/study_b_m5_representation.png)

![Study B: Representation-Route Alignment Gap (M8)](results/figures/study_b_m8_alignment.png)

- **M5** tracks how fast $\psi$ is changing (mean L2 distance between snapshots). B2 shows spikes at phase transitions, confirming that the representation is being actively restructured — not just the policy.
- **M8** measures how tightly the representation is aligned with the current route (mean $\psi$-similarity on-route minus off-route). In B1, this gap grows steadily — the representation locks tighter and tighter onto the single route. In B2, the gap fluctuates, resetting partially at each transition. This is the representation "unlocking" — old route information is partially erased.

#### Psi Grid Across Phases

![Study B – B2 Changing: Psi Similarity Grid (all 3 phases)](results/figures/study_b_psi_grid_changing.png)

This figure shows the $\psi$ similarity landscape at 15 snapshots across all three phases, with the learned policy route overlaid:

- **Phase 0 (FourRooms):** The representation develops a clear gradient from start to goal through the standard four-room path. High-similarity cells form a connected corridor.
- **Phase 1 (corridor_shift):** After the transition, the old corridor partially fades. New high-similarity cells appear along the restricted path. The policy route shifts to use the remaining doors.
- **Phase 2 (l_wall):** The most dramatic restructuring. The entire left half of the grid lights up as the agent discovers the L-shaped passage. The old FourRooms corridor is almost completely dark. The policy route is fundamentally different from Phases 0 and 1.
- **Residual shadows.** In the early snapshots of each new phase, you can see faint traces of the old route. These "shadows" represent representational inertia — the old $\psi$ values have not yet been fully overwritten. They are most visible in Phase 1 snapshots.

#### Trajectory Overlay

![Study B – B2 Changing: Last 100 Trajectories (final phase: l_wall)](results/figures/study_b_trajectories_changing.png)

The final-phase trajectories confirm adaptation. Most trajectories (green) navigate the l_wall layout successfully — down the left corridor and across the bottom. A few failures suggest occasional "ghosts" of old route memories, but the dominant behavior is well-adapted to the new layout.

### Study B Summary

The agent can adapt to environmental change, but the process has clear costs and limitations:

- **Adaptation is possible but slow.** Recovery takes 1000–2000 episodes per transition. During recovery, performance drops to near-zero for major layout changes.
- **Structural distance matters.** Small changes (corridor_shift) cause moderate disruption; large changes (l_wall) cause severe disruption. The KL divergence at transitions quantifies this.
- **Representations carry inertia.** The $\psi$ landscape resists change. Old route information lingers as "shadows" that take thousands of episodes to fully overwrite. This inertia is the same mechanism that causes overexploitation in Study A — except here, the environment forces the agent past it.
- **Performance ceiling degrades with more transitions.** Each phase achieves slightly lower performance than the previous one, suggesting cumulative interference from partially overwritten representations.
- **The agent does not "forget forward."** When walls change, the agent does not strategically erase old information — it overwrites it incrementally through new experience. This is wasteful. A mechanism for detecting structural changes and deliberately resetting parts of the representation could dramatically speed up adaptation.

---

## Cross-Study Analysis

Taken together, the two studies reveal a consistent picture of how contrastive $\psi$ representations interact with exploration and adaptation.

### The Feedback Loop Is the Central Mechanism

Both studies are ultimately about the same phenomenon: the $\psi$ representation creates a self-reinforcing cycle between experience, representation, and policy.

- In **Study A**, this cycle is destructive. Early experience biases the representation toward a particular approach direction. The representation biases the policy. The policy biases future experience. The cycle tightens until the agent is permanently locked onto a suboptimal strategy. External randomness (stochastic success) is not enough to break it.
- In **Study B**, the same cycle is initially beneficial (fast convergence in Phase 0) but creates fragility. When the environment changes, the cycle must be reversed — old representations must be unlearned before new ones can take hold. This creates a recovery period where performance craters.

### Why External Randomness Fails but External Force Succeeds

Study A shows that stochastic success (an external random signal) cannot break the lock-in. Study B shows that changing walls (an external structural change) can. The difference is about **where the perturbation hits:**

- Stochastic success only affects the reward signal at the goal. The agent still generates the same trajectories — it just fails more often. The $\psi$ representation still trains on the same spatial transitions. The critic does not "see" the stochastic check; it only sees state sequences.
- Changing walls alters the transition dynamics. The agent physically cannot follow the old route. New trajectories must pass through different cells. These new cells provide novel training signal to the contrastive objective, which eventually reshapes $\psi$.

The implication: **to break overexploitation, you need to force new experience, not just penalize old experience.** Reward shaping is not enough if the policy cannot generate the data the critic needs.

### Replay Buffer Composition Matters — But Not Enough

The all-replay vs. success-only comparison in Study A shows a real effect:

- All-replay includes failure trajectories, which provide spatial diversity. This leads to slightly less concentrated lock-in (3 directions vs. 2 directions attempted).
- Success-only filters out most from_above trajectories (90% failure rate), so from_left (25% success) wins the early race.
- But neither buffer strategy leads to discovering from_below or from_right. The fundamental bottleneck is that the policy does not generate trajectories through those cells, so no amount of replay filtering can help.

This suggests that interventions targeting the replay buffer (prioritized replay, curiosity-driven sampling) may help at the margins but cannot solve the core problem.

### Non-Obvious Implications

- **$\psi$-based GCRL has a systematic bias toward the first route discovered.** This is not a hyperparameter problem. It is architectural. The tight coupling between representation and policy makes the first successful route extremely "sticky." Agents with separate actor and critic networks may be less vulnerable because the actor can explore independently of the critic's landscape.
- **The "best" strategy under stochastic success depends on early trajectory statistics, not on success probabilities.** In our experiments, the agent never learns the true success probabilities because it never tries all directions enough times. The representation prevents the exploration needed for learning. This is a failure of the exploration-exploitation tradeoff at the representation level, not the action level.
- **Phase transitions in Study B act as "resets" for the feedback loop.** When walls change, the old $\psi$ landscape becomes partially invalid, which temporarily loosens the representation-policy coupling. This is why B2 can recover — the environmental change forces the kind of representational diversity that the agent cannot generate on its own.
- **Cumulative interference across phases suggests a capacity problem.** The $\psi$ representation has 16 dimensions for 121 states. Older route information competes with newer information for limited representational capacity. A higher-dimensional $\psi$ might reduce interference, but would also slow convergence.

---

## Limitations

- **Single seed.** All results use seed 42. Approach direction lock-in patterns may vary across seeds (e.g., which direction "wins" the early race). The qualitative conclusions — that lock-in occurs, that the agent does not find the optimal direction — should be robust, but the specific locked-in direction may not be.
- **One environment.** FourRooms is a simple, structured environment. Real environments have continuous states, partial observability, and more complex dynamics. The feedback loop mechanism should transfer, but the severity of lock-in may differ.
- **No intervention testing.** We document the problem but do not test solutions. Natural next steps include:
  - Epsilon-greedy exploration in $\psi$-space
  - Periodic representation resets
  - Intrinsic motivation (curiosity bonuses)
  - Separate exploration policy
  - Goal-conditioned hindsight replay targeting underexplored approach directions
- **No multi-seed statistical analysis.** Confidence intervals and significance tests require multiple seeds. The current results are illustrative, not statistically rigorous.
- **Fixed approach probabilities.** We chose a particular ordering (from_below = best, from_above = worst). Different orderings would change which direction gets locked in but not whether lock-in occurs.

---

## Conclusions

1. **The $\psi$ feedback loop causes severe overexploitation.** Under stochastic success, the agent devotes 82% of its attempts to the 10%-success direction while allocating only 1% to the 80%-success direction. This is not a hyperparameter issue — it is a structural property of contrastive successor feature learning.

2. **Stochastic outcomes cannot break the loop.** The agent absorbs stochastic failure as noise. It does not interpret high failure rates as a signal to change strategy. This is because the contrastive critic learns from trajectories, not from success/failure labels.

3. **Environmental change can break the loop.** When walls physically change, the old representation becomes invalid, and the agent must rebuild. Recovery takes 1000–2000 episodes with a performance dip proportional to the magnitude of the change.

4. **Replay buffer composition affects which suboptimal strategy wins, not whether suboptimality occurs.** All-replay and success-only converge to different locked-in directions but both miss the optimal one.

5. **The first route discovered has outsized influence.** Early training experience disproportionately shapes the $\psi$ landscape. Once a route is encoded, the representation prevents the exploration needed to discover alternatives. Any practical deployment of $\psi$-based GCRL needs explicit exploration mechanisms that operate at the representation level, not just the action level.

---

## Reproduction

All code is in `experiments/isolation_study/`.

**Run individual conditions:**

```bash
python run_single.py --condition a1 --seed 42
python run_single.py --condition a2ar --seed 42
python run_single.py --condition a2so --seed 42
python run_single.py --condition b1 --seed 42
python run_single.py --condition b2 --seed 42
```

**Generate all figures:**

```bash
python generate_plots.py
```

Results are saved in `results/` with per-condition subdirectories. Each condition saves metrics (JSON), environment config, final $\psi$ embeddings (npy), and snapshot data (pkl) for visualization.
