# Continual SGCRL Experiments

Experiments testing whether SGCRL overexploits the first successful path it finds and fails to explore alternatives — especially when environment dynamics change or when success is stochastic.

## Two Experiment Modes

### 1. Continual Maze (`--mode continual`)

**Setup:** 6-phase 11×11 maze. The goal is fixed at (10,10). The wall layout changes every N episodes, forcing a substantially different optimal path each time.

| Phase | Name | Description |
|-------|------|-------------|
| 0 | `fourrooms` | Standard FourRooms — route via top-right and bottom-right rooms |
| 1 | `corridor_shift` | Doors moved — forces route via bottom-left room |
| 2 | `zigzag` | Three horizontal barriers → right-down-left-down-right zig-zag |
| 3 | `l_wall` | L-shaped wall with bottleneck door |
| 4 | `center_block` | 3×3 center block + right-side wall |
| 5 | `double_spiral` | Two vertical walls + horizontal → long winding path |

**Key question:** When the maze changes and the old path no longer works, does the agent re-explore, or does it get stuck trying to reuse the old policy?

### 2. Stochastic Success (`--mode stochastic`)

**Setup:** Open 11×11 maze (no internal walls). The agent can reach the goal from any direction, but success is probabilistic — depends on the approach direction:

| Approach | Success Probability |
|----------|-------------------|
| from_above | 10% |
| from_left | 25% |
| from_right | 50% |
| from_below | 80% |

**Key question:** If SGCRL discovers the 25%-success route first, will it keep exploiting that route and fail to discover the 80%-success route?

## Metrics

### Standard
1. **Success rate** — per-phase and rolling
2. **Path diversity** — Jaccard distance, route clusters, Shannon entropy
3. **Trajectory preference** — fraction using dominant path
4. **Adaptation speed** — episodes until first success after phase change
5. **ψ-similarity evolution** — heatmap snapshots over training
6. **Representation drift** — L2 / cosine change across phase transitions
7. **Exploitation ratio** — fraction using dominant path

### Adaptation-specific
8. **Policy Adaptation Index** — KL divergence between successive policy snapshots. Large spikes at phase transitions indicate the actor is changing behavior. Small or absent spikes indicate the agent is stuck on the old policy.
9. **Representation Adaptation Rate** — rolling L2 change of ψ embeddings between snapshots. Measures how actively the critic is reshaping the representation landscape.

## Visual Outputs

1. **Trajectory overlay plots** — per-phase, showing all trajectories (green=success, red=failure) overlaid on the maze grid
2. **ψ-similarity evolution video** — MP4 of heatmaps showing how ψ(s)·ψ(g) changes over training
3. **Approach direction analysis** (stochastic mode) — bar charts of attempt counts and empirical vs true success rates per direction
4. **Approach preference over time** (stochastic mode) — rolling fraction of each approach direction

## Running

```bash
# Continual maze (6 phases × 600 episodes, ~30s)
python experiments/continual_maze/run_experiment.py --mode continual --seed 42

# Stochastic success (4000 episodes, ~27s)
python experiments/continual_maze/run_experiment.py --mode stochastic --seed 42

# Custom settings
python experiments/continual_maze/run_experiment.py --mode continual --episodes_per_phase 1000 --seed 123
python experiments/continual_maze/run_experiment.py --mode stochastic --total_episodes 8000 --seed 123
```

Results are saved to `experiments/continual_maze/results/<mode>_seed_<N>/`.

## File Structure

```
experiments/continual_maze/
  envs/
    continual_maze.py     # ContinualMaze (6 wall layouts) + StochasticSuccessMaze
  agent.py                # TabularSGCRLAgent + StochasticSGCRLAgent
  metrics/
    metrics.py            # All metrics including policy adaptation index
  visualization.py        # Trajectory plots, ψ video, metric figures
  run_experiment.py       # Main runner (CLI)
  README.md               # This file
```

## Key Findings (Seed 42)

### Continual Maze
- **FourRooms → Corridor Shift:** Agent adapts quickly (2 episodes) because the corridor_shift layout is still solvable via similar paths. But exploitation ratio hits 100% — agent locks onto a single route.
- **Corridor Shift → Zigzag:** Complete failure (0% success). The zigzag layout requires a fundamentally different path strategy that the agent's frozen representations cannot discover.
- **Policy KL at transitions (0.38) is 47× larger than within-phase (0.008)** — the policy does change at boundaries, but mostly because the dynamics change, not because the agent is actively exploring new routes.
- **Representation adaptation rate decreases monotonically** — the critic slows down its updates over time, even as new phases demand new representations.

### Stochastic Success
- **Overexploitation confirmed:** Agent attempted `from_left` (p=0.25) 3101 times and `from_above` (p=0.10) once. It never tried `from_below` (p=0.80) or `from_right` (p=0.50).
- The ψ-similarity trace formation described in the paper locks the representation landscape into the first successful approach direction, suppressing alternatives.
- The empirical success rate closely matches the true probability for the attempted directions, but the agent has no mechanism to discover it is suboptimal.
