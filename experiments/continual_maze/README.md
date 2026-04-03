# Continual Maze SGCRL Experiment

This experiment tests whether SGCRL (Single-Goal Contrastive RL) overexploits a discovered path and fails to adapt when the maze dynamics change.

## Hypothesis

Once SGCRL finds one successful trajectory, it overexploits that path and fails to explore alternatives — especially when maze dynamics change.

## Experiment Design

### 3-Phase Continual Maze

The maze is a 10x10 grid with a **fixed goal** at (9,9) and **start** at (0,0). The wall layout changes across three phases:

- **Phase 1 (FourRooms):** Standard 4-room layout with doors. Agent must navigate through room sequence to reach the goal.
- **Phase 2 (Shortcut):** A new direct passage opens in the vertical wall (bottom half), creating a shorter path. Does the agent switch to it?
- **Phase 3 (Blocked):** Bottom-half doors are blocked, and a new top-half passage opens. The agent *must* re-explore through the top rooms.

Each phase runs for a configurable number of episodes (default: 500).

### Agent

Tabular SGCRL with lookup-table ψ embeddings (no neural networks). The agent selects actions by computing ψ(s')·ψ(g) similarity for each neighbor state and sampling from a softmax policy. Representations are updated via a vectorized InfoNCE contrastive loss.

### Metrics

1. **Success rate** — per-phase fraction of episodes reaching the goal
2. **Path diversity** — Jaccard distance between trajectories, number of distinct routes, route entropy
3. **Trajectory preference** — fraction of successful episodes using each route
4. **Adaptation speed** — episodes from phase transition to first success
5. **ψ-similarity evolution** — ψ(s)·ψ(g) heatmaps at regular intervals
6. **Representation drift** — L2 distance and cosine similarity of ψ vectors at phase transitions
7. **Exploitation ratio** — fraction of successful episodes using the dominant path

## Quick Start

```bash
# Run with default config (3 seeds, 500 episodes/phase)
python experiments/continual_maze/run_experiment.py

# Run with quick config for testing (1 seed, 100 episodes/phase)
python experiments/continual_maze/run_experiment.py --config experiments/continual_maze/configs/quick.json

# Run with specific seeds
python experiments/continual_maze/run_experiment.py --seeds 0 1 2 3 4

# Run with replay buffer clearing at phase transitions
python experiments/continual_maze/run_experiment.py --config experiments/continual_maze/configs/clear_replay.json

# Analyze results
python experiments/continual_maze/analyze_results.py experiments/continual_maze/results/run_XXX/
```

## Output Structure

```
results/run_XXX/
  config.json                  # experiment configuration
  summary.json                 # cross-seed aggregated summary
  metrics_seed_42.json         # per-seed detailed metrics
  psi_snapshots_seed_42.npz    # ψ-similarity maps over training
  plots/
    success_rate.png
    exploitation_ratio.png
    path_diversity.png
    psi_evolution_seed_42.png
    representation_drift.png
```

## Configuration

See `configs/default.json` for all parameters. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `episodes_per_phase` | 500 | Training episodes per maze phase |
| `rep_dim` | 16 | ψ embedding dimension |
| `lr_psi` | 0.01 | Learning rate for contrastive updates |
| `entropy_coeff` | 0.1 | Softmax temperature (lower = more greedy) |
| `clear_replay_on_phase_change` | false | Whether to clear replay buffer at transitions |
| `seeds` | [42, 123, 456] | Random seeds for reproducibility |

## Interpreting Results

- **High exploitation ratio** (close to 1.0) across phases supports the hypothesis that SGCRL overexploits a single path.
- **Low path diversity** (few distinct routes, low entropy) indicates the agent consistently uses the same path.
- **Slow adaptation** (many episodes to first success after phase change) suggests the learned representations are rigid.
- **Small representation drift** at phase transitions would indicate ψ embeddings resist change even when the environment changes.
