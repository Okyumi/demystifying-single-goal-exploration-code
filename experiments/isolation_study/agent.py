"""
Neural SGCRL agents for the isolation study (HPC-ready, PyTorch).

Replaces the tabular TabularSGCRLAgent with a PyTorch nn.Embedding-based agent.
The key change: psi(s) is stored as an nn.Embedding layer rather than a numpy
array, enabling GPU training, Adam optimisation, and easy checkpointing.

Classes:
  NeuralSGCRLAgent                 — deterministic maze (Study A/B baseline)
  StochasticNeuralAgent_AllReplay  — stochastic success, ALL trajectories in replay
  StochasticNeuralAgent_SuccessOnly — stochastic success, ONLY successes in replay

The environment interface (envs.py) is unchanged: all methods that interact with
the env use cpu tensors / numpy.

Backward-compatibility aliases are provided so existing code that imports
TabularSGCRLAgent / StochasticSGCRLAgent_AllReplay / StochasticSGCRLAgent_SuccessOnly
still works (they map to the neural versions).
"""

from __future__ import annotations

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any


# ---------------------------------------------------------------------------
# Optional MLP head (activated by --use_mlp in run_experiment.py)
# ---------------------------------------------------------------------------

class _MLPHead(nn.Module):
    """2-layer MLP that maps embedding → projected representation."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# NeuralSGCRLAgent
# ---------------------------------------------------------------------------

class NeuralSGCRLAgent:
    """Single-Goal Contrastive RL agent using nn.Embedding.

    Parameters
    ----------
    env : ContinualMaze or StochasticSuccessMaze
    rep_dim : int
        Dimensionality of the psi embeddings.
    lr : float
        Adam learning rate.
    batch_size : int
        Number of (s, s') pairs per contrastive update.
    replay_capacity : int
        Maximum number of trajectories stored in replay buffer.
    max_steps : int
        Maximum environment steps per episode.
    gamma : float
        Geometric discount for future-state sampling.
    temperature : float
        InfoNCE temperature (denominator of logits).  Default 0.07.
    device : str or torch.device
        'cpu', 'cuda', or 'cuda:N'.
    grad_clip : float
        Max gradient norm (0 = no clipping).
    use_mlp : bool
        If True, stack a 2-layer MLP on top of the embedding.
    """

    def __init__(
        self,
        env,
        *,
        rep_dim: int = 64,
        lr: float = 1e-3,
        batch_size: int = 512,
        replay_capacity: int = 5000,
        max_steps: int = 200,
        gamma: float = 0.99,
        temperature: float = 0.07,
        device: str = "cpu",
        grad_clip: float = 1.0,
        use_mlp: bool = False,
        # Legacy aliases
        lr_psi: Optional[float] = None,
        entropy_coeff: Optional[float] = None,
        episodes_per_update: int = 1,
        normalize: bool = True,
    ):
        self.env = env
        self.rep_dim = rep_dim
        self.lr = lr_psi if lr_psi is not None else lr
        self.batch_size = batch_size
        self.replay_capacity = replay_capacity
        self.max_steps = max_steps
        self.gamma = gamma
        self.temperature = temperature
        self.grad_clip = grad_clip
        self.use_mlp = use_mlp
        self.episodes_per_update = episodes_per_update
        self.normalize = normalize  # kept for compat; normalisation done in forward pass

        self.device = torch.device(device)

        self.goal = env.goal_state
        self.start = env.start_state

        # ---- model ----
        self.embedding = nn.Embedding(env.num_states, rep_dim).to(self.device)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.1)

        if use_mlp:
            hidden = max(rep_dim, 128)
            self.mlp = _MLPHead(rep_dim, hidden, rep_dim).to(self.device)
        else:
            self.mlp = None

        params = list(self.embedding.parameters())
        if self.mlp is not None:
            params += list(self.mlp.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.lr)

        self.replay: List[List[int]] = []

    # ------------------------------------------------------------------
    # Internal: get L2-normalised embeddings for a batch of state indices
    # ------------------------------------------------------------------

    def _psi(self, states: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised representations for state indices (on device)."""
        x = self.embedding(states)
        if self.mlp is not None:
            x = self.mlp(x)
        return F.normalize(x, p=2, dim=-1)

    # ------------------------------------------------------------------
    # Action selection (runs on CPU — env is pure numpy)
    # ------------------------------------------------------------------

    def select_action(self, state: int) -> int:
        """Softmax policy from psi-similarity to goal."""
        with torch.no_grad():
            next_states = [self.env.step(state, a)
                           for a in range(self.env.NUM_ACTIONS)]
            ns_t = torch.tensor(next_states, dtype=torch.long,
                                device=self.device)
            goal_t = torch.tensor([self.goal], dtype=torch.long,
                                  device=self.device)
            psi_ns = self._psi(ns_t)           # (A, D)
            psi_g  = self._psi(goal_t)         # (1, D)
            sims = (psi_ns * psi_g).sum(dim=1) # (A,)
            logits = sims / self.temperature
            logits -= logits.max()
            probs = torch.softmax(logits, dim=0).cpu().numpy()
        return int(np.random.choice(self.env.NUM_ACTIONS, p=probs))

    def eval_action(self, state: int) -> int:
        """Greedy action (argmax over psi-similarities)."""
        with torch.no_grad():
            next_states = [self.env.step(state, a)
                           for a in range(self.env.NUM_ACTIONS)]
            ns_t = torch.tensor(next_states, dtype=torch.long,
                                device=self.device)
            goal_t = torch.tensor([self.goal], dtype=torch.long,
                                  device=self.device)
            psi_ns = self._psi(ns_t)
            psi_g  = self._psi(goal_t)
            sims = (psi_ns * psi_g).sum(dim=1)
        return int(sims.argmax().item())

    # ------------------------------------------------------------------
    # Episode collection
    # ------------------------------------------------------------------

    def collect_episode(self) -> Tuple[List[int], bool]:
        traj = [self.start]
        reached_goal = False
        for _ in range(self.max_steps):
            a = self.select_action(traj[-1])
            ns = self.env.step(traj[-1], a)
            traj.append(ns)
            if self.env.is_goal(ns):
                reached_goal = True
                break
        self.replay.append(traj)
        if len(self.replay) > self.replay_capacity:
            self.replay.pop(0)
        return traj, reached_goal

    def run_eval_episode(self) -> Tuple[List[int], bool]:
        traj = [self.start]
        reached_goal = False
        for _ in range(self.max_steps):
            a = self.eval_action(traj[-1])
            ns = self.env.step(traj[-1], a)
            traj.append(ns)
            if self.env.is_goal(ns):
                reached_goal = True
                break
        return traj, reached_goal

    # ------------------------------------------------------------------
    # Policy rollout (multi-sample, pick best)
    # ------------------------------------------------------------------

    def greedy_rollout(self) -> List[int]:
        """Deterministic greedy rollout from start."""
        traj = [self.start]
        for _ in range(self.max_steps):
            a = self.eval_action(traj[-1])
            ns = self.env.step(traj[-1], a)
            traj.append(ns)
            if self.env.is_goal(ns):
                break
        return traj

    def policy_rollout(self, n_rollouts: int = 5) -> List[int]:
        """Stochastic policy rollouts; return the one closest to the goal."""
        goal_coord = np.array(np.unravel_index(
            self.goal, (self.env.height, self.env.width)))
        best_traj = None
        best_dist = float("inf")
        best_len  = float("inf")

        for _ in range(n_rollouts):
            traj = [self.start]
            for _ in range(self.max_steps):
                a = self.select_action(traj[-1])
                ns = self.env.step(traj[-1], a)
                traj.append(ns)
                if self.env.is_goal(ns):
                    break

            min_dist = float("inf")
            for s in traj:
                coord = np.array(np.unravel_index(
                    s, (self.env.height, self.env.width)))
                d = int(np.abs(coord - goal_coord).sum())
                if d < min_dist:
                    min_dist = d

            if min_dist < best_dist or (
                    min_dist == best_dist and len(traj) < best_len):
                best_dist = min_dist
                best_len  = len(traj)
                best_traj = traj

        return best_traj

    # ------------------------------------------------------------------
    # Contrastive update (InfoNCE)
    # ------------------------------------------------------------------

    def update_representations(self) -> float:
        """InfoNCE update. Returns float loss (or 0.0 if too few trajectories)."""
        if len(self.replay) < 2:
            return 0.0

        s_list, sp_list = [], []
        traj_ids = [random.randint(0, len(self.replay) - 1)
                    for _ in range(self.batch_size)]
        for idx in traj_ids:
            traj = self.replay[idx]
            if len(traj) < 2:
                continue
            i = random.randint(0, len(traj) - 2)
            remaining = len(traj) - i
            weights = self.gamma ** np.arange(remaining)
            weights /= weights.sum()
            j = i + int(np.random.choice(remaining, p=weights))
            s_list.append(traj[i])
            sp_list.append(traj[j])

        if not s_list:
            return 0.0

        B = len(s_list)
        s_t  = torch.tensor(s_list,  dtype=torch.long, device=self.device)
        sp_t = torch.tensor(sp_list, dtype=torch.long, device=self.device)

        self.optimizer.zero_grad()
        psi_s  = self._psi(s_t)   # (B, D) — L2 normalised
        psi_sp = self._psi(sp_t)  # (B, D) — L2 normalised

        # InfoNCE: logits[i,j] = psi_s[i] · psi_sp[j] / T
        logits = (psi_s @ psi_sp.T) / self.temperature  # (B, B)
        labels = torch.arange(B, device=self.device)
        loss = F.cross_entropy(logits, labels)

        loss.backward()
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(
                list(self.embedding.parameters()) +
                (list(self.mlp.parameters()) if self.mlp is not None else []),
                self.grad_clip,
            )
        self.optimizer.step()

        return float(loss.item())

    # ------------------------------------------------------------------
    # Snapshot / analysis helpers
    # ------------------------------------------------------------------

    def get_psi_snapshot(self) -> np.ndarray:
        """Return (num_states, rep_dim) numpy array of L2-normalised embeddings."""
        with torch.no_grad():
            all_states = torch.arange(
                self.env.num_states, dtype=torch.long, device=self.device)
            return self._psi(all_states).cpu().numpy()

    def get_similarity_map(self) -> np.ndarray:
        """Return (height, width) numpy array of cosine similarities to goal."""
        with torch.no_grad():
            all_states = torch.arange(
                self.env.num_states, dtype=torch.long, device=self.device)
            goal_t = torch.tensor([self.goal], dtype=torch.long,
                                  device=self.device)
            psi_all  = self._psi(all_states)   # (S, D)
            psi_goal = self._psi(goal_t)        # (1, D)
            sims = (psi_all * psi_goal).sum(dim=1).cpu().numpy()
        return sims.reshape(self.env.height, self.env.width)

    def get_policy_distribution(self) -> np.ndarray:
        """Return (num_states, num_actions) numpy array of softmax probabilities."""
        S = self.env.num_states
        A = self.env.NUM_ACTIONS
        policy = np.zeros((S, A), dtype=np.float32)

        with torch.no_grad():
            goal_t = torch.tensor([self.goal], dtype=torch.long,
                                  device=self.device)
            psi_goal = self._psi(goal_t)  # (1, D)

            for s in range(S):
                next_states = [self.env.step(s, a) for a in range(A)]
                ns_t = torch.tensor(next_states, dtype=torch.long,
                                    device=self.device)
                psi_ns = self._psi(ns_t)  # (A, D)
                sims = (psi_ns * psi_goal).sum(dim=1) / self.temperature
                sims = sims - sims.max()
                probs = torch.softmax(sims, dim=0).cpu().numpy()
                policy[s] = probs

        return policy

    def clear_replay(self):
        self.replay.clear()

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str, extra: Optional[Dict[str, Any]] = None):
        """Save model, optimiser, replay buffer, and optional extras."""
        payload = {
            "model_state_dict": self.embedding.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "replay_buffer": self.replay,
            "config": {
                "rep_dim": self.rep_dim,
                "lr": self.lr,
                "batch_size": self.batch_size,
                "replay_capacity": self.replay_capacity,
                "max_steps": self.max_steps,
                "gamma": self.gamma,
                "temperature": self.temperature,
                "use_mlp": self.use_mlp,
            },
        }
        if self.mlp is not None:
            payload["mlp_state_dict"] = self.mlp.state_dict()
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load model + optimiser from checkpoint, return full payload dict."""
        payload = torch.load(path, map_location=self.device)
        self.embedding.load_state_dict(payload["model_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        if "replay_buffer" in payload:
            self.replay = payload["replay_buffer"]
        if "mlp_state_dict" in payload and self.mlp is not None:
            self.mlp.load_state_dict(payload["mlp_state_dict"])
        return payload

    # ------------------------------------------------------------------
    # Metrics state (for checkpoint/resume)
    # ------------------------------------------------------------------

    def get_state_for_checkpoint(self) -> Dict[str, Any]:
        return {}  # subclasses may extend


# ---------------------------------------------------------------------------
# Stochastic variants
# ---------------------------------------------------------------------------

class StochasticNeuralAgent_AllReplay(NeuralSGCRLAgent):
    """Neural SGCRL agent for StochasticSuccessMaze that adds ALL trajectories
    to replay (regardless of stochastic success outcome).

    collect_episode() returns (traj, success, approach_dir) to match the
    StochasticSuccessMaze interface.
    """

    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.approach_counts:    Dict[str, int] = {}
        self.approach_successes: Dict[str, int] = {}

    def collect_episode(self) -> Tuple[List[int], bool, Optional[str]]:
        traj = [self.start]
        success = False
        approach_dir = None

        for _ in range(self.max_steps):
            a = self.select_action(traj[-1])
            ns = self.env.step(traj[-1], a)
            traj.append(ns)

            if self.env.is_goal(ns):
                prev = traj[-2]
                success, approach_dir, _ = \
                    self.env.check_stochastic_success(prev, ns)
                self.approach_counts[approach_dir] = \
                    self.approach_counts.get(approach_dir, 0) + 1
                if success:
                    self.approach_successes[approach_dir] = \
                        self.approach_successes.get(approach_dir, 0) + 1
                break

        # ALL trajectories → replay
        self.replay.append(traj)
        if len(self.replay) > self.replay_capacity:
            self.replay.pop(0)

        return traj, success, approach_dir

    def run_eval_episode(self) -> Tuple[List[int], bool, Optional[str]]:
        traj = [self.start]
        success = False
        approach_dir = None
        for _ in range(self.max_steps):
            a = self.eval_action(traj[-1])
            ns = self.env.step(traj[-1], a)
            traj.append(ns)
            if self.env.is_goal(ns):
                prev = traj[-2]
                success, approach_dir, _ = \
                    self.env.check_stochastic_success(prev, ns)
                break
        return traj, success, approach_dir

    def get_approach_stats(self) -> Dict[str, Dict[str, Any]]:
        stats = {}
        for d in self.approach_counts:
            attempts  = self.approach_counts[d]
            successes = self.approach_successes.get(d, 0)
            stats[d]  = {
                "attempts":       attempts,
                "successes":      successes,
                "empirical_rate": successes / attempts if attempts > 0 else 0.0,
            }
        return stats

    def get_state_for_checkpoint(self) -> Dict[str, Any]:
        return {
            "approach_counts":    dict(self.approach_counts),
            "approach_successes": dict(self.approach_successes),
        }


class StochasticNeuralAgent_SuccessOnly(NeuralSGCRLAgent):
    """Neural SGCRL agent for StochasticSuccessMaze that adds ONLY successful
    trajectories to replay.

    This is the behaviour of the original continual_maze agents: the contrastive
    critic only sees trajectories that happened to produce a stochastic success.
    """

    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.approach_counts:    Dict[str, int] = {}
        self.approach_successes: Dict[str, int] = {}

    def collect_episode(self) -> Tuple[List[int], bool, Optional[str]]:
        traj = [self.start]
        success = False
        approach_dir = None

        for _ in range(self.max_steps):
            a = self.select_action(traj[-1])
            ns = self.env.step(traj[-1], a)
            traj.append(ns)

            if self.env.is_goal(ns):
                prev = traj[-2]
                success, approach_dir, _ = \
                    self.env.check_stochastic_success(prev, ns)
                self.approach_counts[approach_dir] = \
                    self.approach_counts.get(approach_dir, 0) + 1
                if success:
                    self.approach_successes[approach_dir] = \
                        self.approach_successes.get(approach_dir, 0) + 1
                break

        # ONLY successes → replay
        if success:
            self.replay.append(traj)
            if len(self.replay) > self.replay_capacity:
                self.replay.pop(0)

        return traj, success, approach_dir

    def run_eval_episode(self) -> Tuple[List[int], bool, Optional[str]]:
        traj = [self.start]
        success = False
        approach_dir = None
        for _ in range(self.max_steps):
            a = self.eval_action(traj[-1])
            ns = self.env.step(traj[-1], a)
            traj.append(ns)
            if self.env.is_goal(ns):
                prev = traj[-2]
                success, approach_dir, _ = \
                    self.env.check_stochastic_success(prev, ns)
                break
        return traj, success, approach_dir

    def get_approach_stats(self) -> Dict[str, Dict[str, Any]]:
        stats = {}
        for d in self.approach_counts:
            attempts  = self.approach_counts[d]
            successes = self.approach_successes.get(d, 0)
            stats[d]  = {
                "attempts":       attempts,
                "successes":      successes,
                "empirical_rate": successes / attempts if attempts > 0 else 0.0,
            }
        return stats

    def get_state_for_checkpoint(self) -> Dict[str, Any]:
        return {
            "approach_counts":    dict(self.approach_counts),
            "approach_successes": dict(self.approach_successes),
        }


# ---------------------------------------------------------------------------
# Backward-compatibility aliases (so old imports keep working)
# ---------------------------------------------------------------------------
TabularSGCRLAgent             = NeuralSGCRLAgent
StochasticSGCRLAgent_AllReplay  = StochasticNeuralAgent_AllReplay
StochasticSGCRLAgent_SuccessOnly = StochasticNeuralAgent_SuccessOnly


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from envs import make_fourrooms_maze, make_stochastic_fourrooms_maze

    print("=== NeuralSGCRLAgent self-test ===")
    torch.manual_seed(0)
    np.random.seed(0)

    # --- Test 1: deterministic agent ---
    env = make_fourrooms_maze()
    agent = NeuralSGCRLAgent(
        env, rep_dim=16, lr=1e-3, batch_size=32,
        replay_capacity=100, max_steps=50,
        temperature=0.07, device="cpu",
    )
    print(f"  embedding shape : {agent.embedding.weight.shape}")

    losses = []
    for ep in range(20):
        traj, success = agent.collect_episode()
        loss = agent.update_representations()
        losses.append(loss)
    print(f"  20 episodes done. last loss={losses[-1]:.4f}")

    snap = agent.get_psi_snapshot()
    assert snap.shape == (121, 16), f"Bad snapshot shape: {snap.shape}"
    sim_map = agent.get_similarity_map()
    assert sim_map.shape == (11, 11), f"Bad sim map shape: {sim_map.shape}"
    pol = agent.get_policy_distribution()
    assert pol.shape == (121, 5), f"Bad policy shape: {pol.shape}"
    traj = agent.policy_rollout(n_rollouts=3)
    assert isinstance(traj, list) and len(traj) > 0

    # --- Test 2: MLP head ---
    agent_mlp = NeuralSGCRLAgent(
        env, rep_dim=16, lr=1e-3, batch_size=16,
        replay_capacity=50, max_steps=30,
        temperature=0.07, device="cpu", use_mlp=True,
    )
    for _ in range(5):
        agent_mlp.collect_episode()
    loss = agent_mlp.update_representations()
    print(f"  MLP-head agent loss={loss:.4f}")

    # --- Test 3: stochastic all-replay ---
    env_s = make_stochastic_fourrooms_maze()
    agent_s = StochasticNeuralAgent_AllReplay(
        env_s, rep_dim=16, lr=1e-3, batch_size=32,
        replay_capacity=100, max_steps=50,
        temperature=0.07, device="cpu",
    )
    for ep in range(20):
        traj, success, adir = agent_s.collect_episode()
        agent_s.update_representations()
    stats = agent_s.get_approach_stats()
    print(f"  AllReplay approach stats: {stats}")

    # --- Test 4: stochastic success-only ---
    agent_so = StochasticNeuralAgent_SuccessOnly(
        env_s, rep_dim=16, lr=1e-3, batch_size=32,
        replay_capacity=100, max_steps=50,
        temperature=0.07, device="cpu",
    )
    for ep in range(20):
        traj, success, adir = agent_so.collect_episode()
        agent_so.update_representations()
    print(f"  SuccessOnly replay size: {len(agent_so.replay)}")

    # --- Test 5: checkpoint round-trip ---
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = str(pathlib.Path(tmpdir) / "test.pt")
        agent.save_checkpoint(ckpt_path, extra={"episode": 20})
        agent2 = NeuralSGCRLAgent(env, rep_dim=16, device="cpu")
        payload = agent2.load_checkpoint(ckpt_path)
        assert payload["episode"] == 20, "Checkpoint round-trip failed"
        snap2 = agent2.get_psi_snapshot()
        np.testing.assert_allclose(snap, snap2, atol=1e-6)
        print("  Checkpoint round-trip OK")

    print("=== All tests passed ===")
