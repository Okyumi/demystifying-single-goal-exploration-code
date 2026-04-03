# 2D Maze Experiment Guide

Comprehensive documentation for reproducing and understanding the 2D maze experiments from the "Demystifying the Mechanisms Behind Emergent Exploration in Goal-conditioned RL" codebase.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Environment and Maze Setup](#2-environment-and-maze-setup)
3. [Training Pipeline](#3-training-pipeline)
4. [Model and Critic Architecture](#4-model-and-critic-architecture)
5. [Goal Representation](#5-goal-representation)
6. [Reward Construction](#6-reward-construction)
7. [Evaluation Logic](#7-evaluation-logic)
8. [Configuration and Hyperparameters](#8-configuration-and-hyperparameters)
9. [Ablation-Specific Code Paths](#9-ablation-specific-code-paths)
10. [Tabular SGCRL Implementation](#10-tabular-sgcrl-implementation)
11. [Running Experiments](#11-running-experiments)
12. [File Reference](#12-file-reference)

---

## 1. Project Overview

This codebase implements **Single-Goal Contrastive RL (SGCRL)** in two settings:

1. **Continuous SGCRL** (neural network-based): A distributed RL pipeline built on [dm-acme](https://github.com/deepmind/acme), [JAX](https://github.com/google/jax), [Haiku](https://github.com/deepmind/dm-haiku), and [Launchpad](https://github.com/deepmind/launchpad). The agent learns contrastive representations (phi/psi) of states and goals, then uses the inner product of these representations as an implicit reward signal to drive exploration.

2. **Tabular SGCRL** (no neural nets): A pure-NumPy implementation where each state has a learnable embedding vector. This isolates the algorithmic mechanism from function approximation and demonstrates that the exploration behavior is a property of the contrastive objective, not the neural network.

### Key Paper Findings

- SGCRL implicitly maximizes a "ghost" reward: `psi(s)^T psi(g)`
- Before finding the goal, contrastive learning makes representations of visited states orthogonal to `psi(g)` ("pruning")
- After finding the goal, representations along successful paths align with `psi(g)` ("trace formation")
- This works even in tabular settings -- the mechanism is low-rank representations

---

## 2. Environment and Maze Setup

### 2.1 File: `point_env.py`

The 2D point navigation environment (`PointEnv`) is a continuous-action gym environment where an agent moves in a grid-based maze.

#### Wall Layouts

Mazes are defined as binary NumPy arrays in the `WALLS` dictionary (`point_env.py:10-92`). `0` = open, `1` = wall.

**FourRooms (11x11)** -- the primary maze used in experiments:

```python
'FourRooms':
    np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              [1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1],
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0]])
```

Other available mazes: `Small` (4x4), `Cross` (7x7), `U` (9x3), `Impossible` (9x9), `Spiral11x11`, `Wall11x11`, `Maze11x11`.

#### State Space

- **Observation**: `[x_pos, y_pos, goal_x, goal_y]` -- 4-dimensional float vector
- **Observation space**: `Box(low=[0,0,0,0], high=[H,W,H,W])` where H,W are maze dimensions
- Positions are **continuous** floats within each grid cell (not discrete cell indices)

#### Action Space

- **Action**: `Box(low=[-1,-1], high=[1,1])` -- 2D continuous velocity
- **Action noise**: Gaussian with `std=0.01` added to each action (`point_env.py:124`)

#### Step Function (`point_env.py:190-214`)

```python
def step(self, action):
    # Add action noise
    action += np.random.normal(0, self._action_noise, (2,))
    action = np.clip(action, -1, 1)

    # Sub-step integration (10 sub-steps)
    num_substeps = 10
    dt = 1.0 / num_substeps
    for _ in np.linspace(0, 1, num_substeps):
        for axis in range(2):
            new_state = self.state.copy()
            new_state[axis] += dt * action[axis]
            if not self._is_blocked(new_state):
                self.state = new_state

    dist = np.linalg.norm(self.goal - self.state)
    rew = float(dist < 1.0)  # Binary reward: 1 if within distance 1.0 of goal
    done = False  # Never terminates early
    return obs, rew, done, {}
```

Key details:
- Movement is integrated over **10 sub-steps** per action for smoother collision handling
- Each axis is updated independently (prevents diagonal wall clipping)
- **Wall collision** check: `_is_blocked()` discretizes position to grid cell via `floor()` and checks the wall array
- Episode **never terminates early** (`done=False` always). Termination comes from the `StepLimitWrapper`

#### Episode Length

- 11x11 mazes: **100 steps** per episode
- Smaller mazes: **50 steps** per episode

#### Reset (`point_env.py:155-175`)

```python
def reset(self, random=True):
    if self._fixed_start_end is not None:
        self.state = self._fixed_start_end[0]
        self.goal = self._fixed_start_end[1]
    else:
        self.goal = self._sample_empty_state()
        self.state = self._sample_empty_state()
    if random:
        # Small random perturbation to start position (up to 0.1 per axis)
        shift = np.random.uniform(-0.1, 0.1, size=self.state.shape)
        ...
```

When `fixed_start_end` is provided (which it is in the SGCRL experiments), the start and goal are fixed, with a small random perturbation (up to 0.1) applied to the start position.

### 2.2 Environment Loading: `env_utils.py`

The `load()` function (`env_utils.py:36-68`) routes environment creation:

```python
def load(env_name, fixed_start_end=None, extra_dim=8):
    if env_name.startswith('point_'):
        CLASS = point_env.PointEnv
        kwargs['walls'] = env_name.split('_')[-1]  # e.g., "FourRooms"
        kwargs['fixed_start_end'] = fixed_start_end
        if '11x11' in env_name:
            max_episode_steps = 100
        else:
            max_episode_steps = 50
    gym_env = CLASS(**kwargs)
    obs_dim = gym_env.observation_space.shape[0] // 2  # = 2 for point envs
    return gym_env, obs_dim, max_episode_steps
```

### 2.3 Environment Wrapping: `contrastive/utils.py:294-320`

The raw gym environment is wrapped in three layers:

```python
def make_environment(env_name, start_index, end_index, seed, fixed_start_end=None, extra_dim=8):
    gym_env, obs_dim, max_episode_steps = env_utils.load(env_name, fixed_start_end, extra_dim)
    env = gym_wrapper.GymWrapper(gym_env)                          # Gym -> dm_env
    env = step_limit.StepLimitWrapper(env, step_limit=max_episode_steps)  # Episode termination
    env = ObservationFilterWrapper(env, indices)                   # Select observation indices
    return env, obs_dim
```

- `GymWrapper`: Converts Gym API to dm_env API
- `StepLimitWrapper`: Enforces max episode steps (100 for FourRooms)
- `ObservationFilterWrapper`: Selects which observation indices to expose (state + goal coordinates)

---

## 3. Training Pipeline

### 3.1 Entry Point: `lp_contrastive.py`

The main training script uses **Launchpad** for distributed execution.

#### Program Structure (`lp_contrastive.py:118-178`)

```python
def get_program(params):
    config = contrastive.ContrastiveConfig(**params)
    agent = contrastive.DistributedContrastive(
        seed=seed,
        environment_factory=env_factory_no_extra,
        environment_factory_fixed_goals=env_factory_no_extra_fixed_goals,
        network_factory=network_factory,
        config=config,
        num_actors=config.num_actors,  # default: 4
        max_number_of_steps=config.max_number_of_steps)
    return agent.build()
```

#### Distributed Topology (`contrastive/distributed_layout.py:263-314`)

The `build()` method creates a Launchpad program with these nodes:

1. **Replay** (`ReverbNode`): Reverb replay buffer storing full episodes
2. **Counter** (`CourierNode`): Tracks actor_steps, learner_steps
3. **Coordinator**: `StepsLimiter` that stops training at `max_number_of_steps`
4. **Learner** (`CourierNode`): Single learner performing gradient updates
5. **Evaluators** (`CourierNode`): Separate eval loops with fixed goals
6. **Actors** (`CourierNode`, x `num_actors`): Parallel data collection

### 3.2 Episode Collection and Replay Buffer

#### Episode Adder (`contrastive/builder.py:207-213`)

Episodes are stored as full trajectories (not individual transitions):

```python
def make_adder(self, replay_client):
    return adders_reverb.EpisodeAdder(
        client=replay_client,
        max_sequence_length=self._config.max_episode_steps + 1)  # 101 for FourRooms
```

#### Replay Tables (`contrastive/builder.py:92-115`)

```python
def make_replay_tables(self, environment_spec):
    min_replay_traj = self._config.min_replay_size // self._config.max_episode_steps
    max_replay_traj = self._config.max_replay_size // self._config.max_episode_steps
    # SampleToInsertRatio rate limiter ensures balanced sampling
    limiter = rate_limiters.SampleToInsertRatio(
        min_size_to_sample=min_replay_traj,
        samples_per_insert=self._config.samples_per_insert)  # default: 256
    return [reverb.Table(
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        max_size=max_replay_traj)]
```

Key parameters:
- `min_replay_size`: 10,000 (steps) = ~100 episodes before training starts
- `max_replay_size`: 1,000,000 (steps) = ~10,000 episodes
- `samples_per_insert`: 256

#### Goal Sampling (HER-style): `contrastive/builder.py:117-205`

The `make_dataset_iterator()` method implements **Hindsight Experience Replay**-style goal relabeling:

```python
@tf.function
def flatten_fn(sample):
    # For each state in the trajectory, sample a FUTURE state as the goal
    # Future state is sampled with geometric discount: P(j) ~ gamma^(j-i)
    is_future_mask = tf.cast(arange[:, None] < arange[None], tf.float32)
    discount = self._config.discount ** tf.cast(arange[None] - arange[:, None], tf.float32)
    probs = is_future_mask * discount

    goal_index = tf.random.categorical(logits=tf.math.log(probs), num_samples=1)[:, 0]

    state = sample.data.observation[:-1, :obs_dim]
    next_state = sample.data.observation[1:, :obs_dim]
    goal = sample.data.observation[:, :obs_dim]  # Use STATE positions as goals
    goal = tf.gather(goal, goal_index[:-1])       # Sample future state as goal

    new_obs = tf.concat([state, goal], axis=1)    # [s, g_future]
```

This is critical: **goals are relabeled from future states in the same trajectory**, weighted by the discount factor gamma. This means the contrastive loss trains on (state, future_state) pairs from the replay buffer, not on the fixed goal directly.

#### Transpose Shuffle

After goal relabeling, the dataset applies a "transpose shuffle" (`builder.py:172-183`):
1. Batch `batch_size` transitions
2. Transpose the batch (permute axes 0 and 1)
3. Unbatch

This creates a batch where row i contains transitions from different trajectories, ensuring diverse negatives in the contrastive loss.

### 3.3 Training Loop: `contrastive/learning.py`

#### The `step()` Method (`learning.py:456-567`)

Each learner step:

1. Sample a batch from the replay buffer
2. (Optional) Replace some goals with the fixed goal as positive examples (`goal_pos_actor_steps`)
3. Determine whether to use the fixed goal as a negative example (`goal_neg_actor_steps`)
4. Run the JIT-compiled `update_step` function
5. Log metrics

#### The `update_step()` Function (`learning.py:320-390`)

Each gradient step performs three updates:

1. **Critic loss** -> update Q-network (sa_encoder + g_encoder)
2. **Actor loss** -> update policy network
3. **Alpha loss** -> update entropy temperature (SAC-style)

All three use Adam optimizer with learning rate 3e-4.

---

## 4. Model and Critic Architecture

### 4.1 Network Creation: `contrastive/networks.py:160-361`

```python
def make_networks(spec, obs_dim, repr_dim=64, repr_norm=False,
                  hidden_layer_sizes=(256, 256), twin_q=False,
                  use_image_obs=False, config=None):
```

#### Critic (Q-Network): Dual Encoder

The critic consists of two separate encoders:

**sa_encoder** (state-action encoder, the "phi" network):
```python
sa_encoder = hk.nets.MLP(
    list(hidden_layer_sizes) + [repr_dim],    # [256, 256, 64]
    w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
    activation=jax.nn.relu,
    name='sa_encoder')
sa_repr = sa_encoder(jnp.concatenate([state, action], axis=-1))
```

**g_encoder** (goal encoder, the "psi" network):
```python
g_encoder = hk.nets.MLP(
    list(hidden_layer_sizes) + [repr_dim],    # [256, 256, 64]
    w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
    activation=jax.nn.relu,
    name='g_encoder')
g_repr = g_encoder(goal)
```

Both produce `repr_dim`-dimensional vectors (default: 64).

**Inner product critic**:
```python
def _combine_repr(sa_repr, g_repr):
    return jax.numpy.einsum('ik,jk->ij', sa_repr, g_repr)
```

This computes a `(batch_size x batch_size)` matrix of similarities between all state-action and goal pairs. The diagonal entries are the "positive" pairs (matched), off-diagonal are "negatives".

**Full critic output**:
```python
def _critic_fn(obs, action):
    sa_repr, g_repr, hidden = _repr_fn(obs, action)
    critic_val = _combine_repr(sa_repr, g_repr)  # (B, B) logit matrix
    return critic_val, sa_repr, g_repr
```

#### Policy (Actor) Network

```python
def _actor_fn(obs):
    network = hk.Sequential([
        hk.nets.MLP(
            list(hidden_layer_sizes),           # [256, 256]
            activation=jax.nn.relu,
            activate_final=True),
        NormalTanhDistribution(num_dimensions,   # 2 for point envs
                               min_scale=1e-6),
    ])
    return network(obs)
```

The policy outputs a **TanhTransformedNormal** distribution over 2D continuous actions. During evaluation, it uses the mode (deterministic).

#### Optional: Representation Normalization

When `repr_norm=True` (`networks.py:280-287`):
```python
sa_repr = sa_repr / jnp.linalg.norm(sa_repr, axis=1, keepdims=True)
g_repr = g_repr / jnp.linalg.norm(g_repr, axis=1, keepdims=True)
```

This normalizes embeddings to the unit sphere, making the inner product equivalent to cosine similarity.

#### Optional: ResidualMLP (`networks.py:30-84`)

When `use_residual_mlp=True`, both encoders and the policy use `ResidualMLP` instead of standard MLP:

```python
class ResidualMLP(hk.Module):
    # skip connections every `skip_every` layers (default: 4)
    # uses swish activation + LayerNorm
    # VarianceScaling(1.0, "fan_in", "uniform") initialization
```

### 4.2 Contrastive Loss: `contrastive/learning.py:108-265`

#### CPC Loss (default, `use_cpc=True`)

The CPC (Contrastive Predictive Coding) loss is a **softmax cross-entropy** over the logit matrix:

```python
def loss_fn(_logits):
    if config.use_cpc:
        labels = I  # Identity matrix (B x B)
        return (optax.softmax_cross_entropy(logits=_logits, labels=labels)
                + 0.01 * jax.nn.logsumexp(_logits, axis=1)**2)
```

- **Positives**: Diagonal of the logit matrix (matched state-action / future-goal pairs)
- **Negatives**: Off-diagonal entries (mismatched pairs from the batch)
- **Regularizer**: `0.01 * logsumexp(logits)^2` prevents logits from growing unbounded

#### TD Loss (alternative, `use_td=True`)

When `use_td=True` with `twin_q=True`, the loss becomes a binary cross-entropy with importance weighting, implementing C-learning:

```python
# Twin Q: use minimum of two Q-values for stability
next_v = jnp.min(next_q, axis=-1)
w = next_v / (1 - next_v)  # importance weights
w = jnp.clip(w, 0, 20.0)

loss_pos = sigmoid_binary_cross_entropy(logits=pos_logits, labels=1)
loss_neg1 = w * sigmoid_binary_cross_entropy(logits=neg_logits, labels=1)
loss_neg2 = sigmoid_binary_cross_entropy(logits=neg_logits, labels=0)

loss = (1-gamma) * loss_pos + gamma * loss_neg1 + loss_neg2
```

### 4.3 Actor Loss: `contrastive/learning.py:267-314`

```python
def actor_loss(policy_params, q_params, alpha, transitions, key):
    # Half original goals, half random (rolled) goals
    new_goal = jnp.concatenate([goal, jnp.roll(goal, 1, axis=0)], axis=0)

    # Get Q-values for current policy actions
    q_action, sa_repr, sf_repr = networks.q_network.apply(q_params, new_obs, action)
    actor_loss = -jnp.diag(q_action)  # Maximize Q (diagonal = matched pairs)

    # Entropy bonus
    if config.use_action_entropy:
        actor_loss -= alpha * (-log_prob)  # Maximize entropy
```

Key detail: `random_goals=0.5` means the actor is trained with **50% original goals** and **50% randomly-permuted goals** from the batch. This provides goal-diversity during policy optimization.

### 4.4 Alpha (Temperature) Loss

SAC-style adaptive temperature (`learning.py:93-105`):

```python
def alpha_loss(log_alpha, policy_params, transitions, key):
    # Eq 18 from SAC paper (Haarnoja et al. 2018)
    alpha = jnp.exp(log_alpha)
    alpha_loss = alpha * stop_gradient(-log_prob - target_entropy)
```

Default: `entropy_coefficient=0.0` with `target_entropy = -num_actions` (= -2 for point envs).

---

## 5. Goal Representation

### 5.1 Fixed Goal Coordinates: `lp_contrastive.py:87-97`

```python
fixed_goal_dict = {
    'point_FourRooms':   [np.array([0,0]),  np.array([10,8])],     # start, goal
    'point_Spiral11x11': [np.array([5,5]),  np.array([10,10])],
    'point_Maze11x11':   [np.array([0,0]),  np.array([5,4])],
    'point_Impossible':  [np.array([9,0]),  np.array([7,9])],
    'point_Wall11x11':   [np.array([2,0]),  np.array([0,0])],
}
```

For **FourRooms**: start at **(0, 0)** (top-left), goal at **(10, 8)** (bottom-right area).

### 5.2 How the Goal is Encoded

In `point_env.py:152-153`:
```python
def _get_obs(self):
    return np.concatenate([self.state, self.goal]).astype(np.float32)
```

The observation is `[x, y, goal_x, goal_y]`. The networks split this at `obs_dim=2`:
- `state = obs[:, :2]` -- agent position
- `goal = obs[:, 2:]` -- goal position

The goal goes through `g_encoder` (a separate MLP) to produce `g_repr`, the goal representation.

### 5.3 Goal Overriding

The `--fixed_goal` flag allows overriding the goal at runtime:
```bash
python lp_contrastive.py --env point_FourRooms --fixed_goal="5,5"
```

This is parsed in `lp_contrastive.py:193-200` and stored in `config.fixed_goal`.

---

## 6. Reward Construction

### 6.1 Explicit (Environment) Reward

From `point_env.py:213`:
```python
rew = float(dist < 1.0)  # Binary: 1 if within L2 distance 1.0 of goal
```

However, this reward is **not used by the contrastive learner**. It is only used by the `SuccessObserver` for tracking success rate.

### 6.2 Implicit Reward via Contrastive Representations

The actual "reward" driving the agent is the **inner product** `sa_repr^T g_repr`, which emerges from the contrastive loss. The actor maximizes:

```python
actor_loss = -jnp.diag(q_action)  # q_action = sa_repr @ g_repr.T
```

The diagonal of `sa_repr @ g_repr.T` gives `sa_repr[i]^T g_repr[i]` -- the inner product for each transition's own (state-action, goal) pair. The actor maximizes this, effectively maximizing the "psi-similarity" between the current state-action representation and the goal representation.

### 6.3 The "Ghost Reward"

As the paper shows, the contrastive loss implicitly creates a reward landscape where:
- States far from the goal (never visited alongside goal trajectories) have representations orthogonal to `psi(g)` -> low implicit reward
- States along successful paths get representations aligned with `psi(g)` -> high implicit reward
- This creates an exploration pressure: the agent is pushed away from well-visited states and toward unexplored regions

---

## 7. Evaluation Logic

### 7.1 Success Observer: `contrastive/utils.py:38-65`

```python
class SuccessObserver(observers_base.EnvLoopObserver):
    def observe(self, env, timestep, action):
        assert timestep.reward in [0, 1]
        self._rewards.append(timestep.reward)

    def get_metrics(self):
        return {
            'success': float(np.sum(self._rewards) >= 1),        # This episode
            'success_1000': np.mean(self._success[-1000:]),       # Rolling 1000-episode average
        }
```

**Success criterion**: An episode is successful if the agent receives **at least one reward of 1** (i.e., gets within L2 distance 1.0 of the goal at any point during the episode).

### 7.2 Distance Observer: `contrastive/utils.py:197-251`

Tracks several distance metrics per episode:
- `init_dist`: L2 distance at episode start
- `final_dist`: L2 distance at episode end
- `delta_dist`: Improvement (`init_dist - final_dist`)
- `min_dist`: Closest approach to goal during episode
- Smoothed versions over windows of 10, 100, 1000 episodes

### 7.3 Region Visit Observer: `contrastive/utils.py:71-194`

Tracks how many steps per episode the agent spends in defined rectangular regions:
- Saves data to JSON files in `./safety_region_visits_data/{env}_{seed}/`
- Files are saved in blocks of `save_every` episodes (default: 1000)
- Used for the safety region avoidance experiments

### 7.4 Evaluator Configuration: `contrastive/agents.py:53-74`

The evaluator uses:
- **Fixed goals** (always from `fixed_goal_dict`, even if training uses random goals)
- **Deterministic policy** (`eval_mode=True` -> uses distribution mode, not sampling)
- Separate environment factory to ensure evaluation always tests with the canonical start/goal

---

## 8. Configuration and Hyperparameters

### 8.1 Full Config Class: `contrastive/config.py`

```python
@dataclasses.dataclass
class ContrastiveConfig:
    # --- Logging ---
    add_uid: bool = True
    time_delta_minutes: int = 5
    log_dir: str = 'logs/'

    # --- Training ---
    max_number_of_steps: int = 8_000_000
    num_actors: int = 4
    num_sgd_steps_per_step: int = 64    # Gradient updates per learner step

    # --- Environment ---
    fix_goals: bool = False

    # --- Loss ---
    batch_size: int = 256
    actor_learning_rate: float = 3e-4
    learning_rate: float = 3e-4         # Critic learning rate
    reward_scale: float = 1
    discount: float = 0.99
    n_step: int = 1
    tau: float = 0.005                  # Target network soft update rate

    # --- Network ---
    hidden_layer_sizes: Tuple[int, ...] = (256, 256)
    repr_dim: Union[int, str] = 64      # Representation dimension
    repr_norm: bool = False             # L2-normalize representations

    # --- Entropy ---
    use_action_entropy: bool = True
    entropy_coefficient: Optional[float] = None  # None = adaptive (SAC)
    target_entropy: float = 0.0         # Overridden to -num_actions

    # --- Replay ---
    min_replay_size: int = 10_000
    max_replay_size: int = 1_000_000
    samples_per_insert: float = 256
    prefetch_size: int = 4

    # --- Algorithm Selection ---
    use_cpc: bool = False               # CPC loss (softmax cross-entropy)
    use_td: bool = False                # TD / C-learning loss
    twin_q: bool = False                # Twin Q-networks
    random_goals: float = 0.5           # Fraction of random goals in actor loss

    # --- Ablation Controls ---
    goal_neg_actor_steps: int = 0       # Steps to use goal as negative
    goal_pos_actor_steps: int = 0       # Steps to inject goal as positive
    goal_pos_frac: float = 0.05         # Fraction of positive goal injection
    cold_q_init: bool = False           # Near-zero Q initialization
    backward_loss: bool = False         # Transpose logits before loss
    weight_reset_interval: int = 0      # Periodic network reset (0 = disabled)
    use_residual_mlp: bool = False      # ResidualMLP instead of MLP

    # --- Safety Region ---
    region_bounds: Tuple = None         # (lower, upper) axis-aligned box
    negative_goal_repr: bool = True     # Use -1 * g_encoder(goal) in region
    stop_grad_fixed: bool = True        # Stop gradient through substituted goal repr
```

### 8.2 Default Experiment Parameters

From `lp_contrastive.py:204-212` (the `main()` function):

```python
params = {
    'seed': seed_idx,                       # default: 31
    'use_random_actor': True,
    'entropy_coefficient': 0.0,             # Fixed, not adaptive
    'env_name': env_name,
    'max_number_of_steps': FLAGS.num_steps, # default: 2_000_000
}
```

Note: `entropy_coefficient=0.0` means **no entropy bonus** by default. The paper's experiments may use adaptive entropy (set `entropy_coefficient=None`).

### 8.3 Algorithm Selection

```python
if alg == 'contrastive_cpc':   # Default
    params['use_cpc'] = True
elif alg == 'c_learning':
    params['use_td'] = True
    params['twin_q'] = True
elif alg == 'nce+c_learning':
    params['use_td'] = True
    params['twin_q'] = True
    params['add_mc_to_td'] = True
```

### 8.4 Key Hyperparameters for FourRooms Experiments

| Parameter | Value | Source |
|-----------|-------|--------|
| Grid size | 11x11 | `point_env.py:34` |
| Start position | (0, 0) | `lp_contrastive.py:89` |
| Goal position | (10, 8) | `lp_contrastive.py:89` |
| Episode length | 100 steps | `env_utils.py:57` |
| Observation dim | 2 (position only) | `env_utils.py:66` |
| Action dim | 2 (x,y velocity) | `point_env.py:126-128` |
| Action noise | 0.01 std | `point_env.py:124` |
| Success threshold | L2 distance < 1.0 | `point_env.py:213` |
| Repr dim | 64 | `config.py:61` |
| Hidden layers | (256, 256) | `config.py:38` |
| Batch size | 256 | `config.py:30` |
| Discount (gamma) | 0.99 | `config.py:34` |
| Actor LR | 3e-4 | `config.py:31` |
| Critic LR | 3e-4 | `config.py:32` |
| Tau (target update) | 0.005 | `config.py:37` |
| Num actors | 4 | `config.py:24` |
| SGD steps per step | 64 | `config.py:57` |
| Min replay size | 10,000 | `config.py:48` |
| Max replay size | 1,000,000 | `config.py:49` |
| Random goals (actor) | 0.5 | `config.py:69` |
| Num training steps | 2,000,000 (default) | `lp_contrastive.py:32` |

---

## 9. Ablation-Specific Code Paths

### 9.1 Goal as Negative Example (`goal_neg_actor_steps`)

**Flag**: `--goal_neg_actor_steps=N`

When `actor_steps < N`, the fixed goal is appended as an **extra negative column** in the CPC logit matrix (`learning.py:195-230`):

```python
if (fixed_goal is not None) and use_goal_neg:
    g_fixed = jnp.broadcast_to(fixed_goal, (B, fixed_goal.shape[-1]))
    obs_fixed = jnp.concatenate([s, g_fixed], axis=1)
    fixed_logits, _, _ = networks.q_network.apply(q_params, obs_fixed, transitions.action)

    _logits = jnp.concatenate([_logits, fixed_logits[:, None]], axis=1)
    labels = jnp.concatenate([I, jnp.zeros((B, 1))], axis=1)
```

This adds the fixed goal as an explicit negative in the contrastive loss, which should accelerate the "pruning" phase (making visited-state representations orthogonal to the goal).

Two JIT-compiled update functions are created at initialization (`learning.py:408-419`):
```python
self._update_step_true  = make_update(True)   # goal is a negative
self._update_step_false = make_update(False)  # stop using it
```

### 9.2 Goal as Positive Example (`goal_pos_actor_steps`)

**Flag**: `--goal_pos_actor_steps=N`

When `actor_steps < N`, a fraction (`goal_pos_frac`, default 5%) of replay transitions have their goals replaced with the fixed goal, and actions are randomized (`learning.py:499-537`):

```python
if self.config.fixed_goal is not None and actor_steps < self.config.goal_pos_actor_steps:
    idx = jax.random.choice(k1, B, (max(1, int(B * frac)),), replace=False)
    obs = obs.at[idx, obs_dim:2*obs_dim].set(
            jnp.broadcast_to(fixed_goal, (idx.size, obs_dim)))
    actions = actions.at[idx].set(random_actions)
```

### 9.3 Safety Region / Representation Injection (`region_bounds`)

**Flag**: `--region_bounds=x_lo,y_lo:x_hi,y_hi`

When an observation's goal falls inside the region, its goal representation is replaced with a modified version (`networks.py:250-274`):

```python
if (config.fixed_goal is not None) and (config.region_bounds is not None):
    mask = _in_region(goal, lower, upper)     # (batch,) boolean

    if config.negative_goal_repr:
        goal_fixed_repr = -1 * g_encoder(goal_fixed)   # Negate the repr
    else:
        goal_fixed_repr = 0.5 * g_encoder(goal_fixed)  # Scale down

    g_repr = _replace_with_goal(mask, g_repr, goal_fixed_repr, config.stop_grad_fixed)
```

- `negative_goal_repr=True`: Goal representation is **negated** in the region, creating a repulsive signal
- `negative_goal_repr=False`: Goal representation is **scaled to 0.5**, weakening the signal
- `stop_grad_fixed=True`: Gradients do not flow through the substituted representation

### 9.4 Backward Loss (`backward_loss`)

**Flag**: `--backward_loss`

Transposes the logit matrix before computing the softmax cross-entropy (`learning.py:232-234`):

```python
if config.backward_loss:
    _logits = _logits.T
    return optax.softmax_cross_entropy(logits=_logits, labels=labels)
```

This changes the contrastive loss from "given anchor s, classify positive g+" to "given anchor g, classify positive s+".

### 9.5 Cold Q Initialization (`cold_q_init`)

**Flag**: `--cold_q_init`

Initializes the last layer of the Q-network with very small weights (`cold_q_scale`, default 1e-12). This is referenced in `config.py:94-95` but the initialization logic is applied when building networks.

### 9.6 Weight Reset (`weight_reset_interval`)

**Flag**: `--weight_reset_interval=N`

Periodically reinitializes all network parameters (`learning.py:463-478`):

```python
if actor_steps > 0 and self.config.weight_reset_interval > 0:
    if self._reset_counter > self.config.weight_reset_interval:
        self._state = reset_network_state()  # Fresh random params
```

### 9.7 Q-Max Action Selection (`Q_max`)

**Flag**: `--Q_max`

Instead of using the learned policy, select actions by maximizing Q over a grid (`networks.py:129-157`):

```python
def maximize_q_action(q_network, q_params, obs, grid_size=20):
    grid = jnp.linspace(-1.0, 1.0, num=grid_size)
    action_grid = jnp.array(list(itertools.product(grid, repeat=action_dim)))
    # Evaluate Q for all grid actions, pick argmax
    q_values = jnp.sum(sa_repr * g_repr, axis=-1)
    best_action = action_grid[jnp.argmax(q_values)]
```

This evaluates 20^2 = 400 candidate actions per step.

---

## 10. Tabular SGCRL Implementation

### File: `tabular_maze.ipynb`

The tabular implementation is a self-contained Jupyter notebook that demonstrates SGCRL **without neural networks**.

### 10.1 Tabular Environment

```python
HEIGHT, WIDTH = 10, 10       # 10x10 grid (NOT 11x11 like continuous)
NUM_STATES = 100
NUM_ACTIONS = 5              # stay, down, up, right, left
A_TO_DELTA = np.array([[0,0], [1,0], [-1,0], [0,1], [0,-1]])
START_STATE = 0              # (0, 0) flattened
GOAL_STATE = 99              # (9, 9) flattened
```

The maze is a **discrete** 10x10 FourRooms with horizontal and vertical walls:
- Horizontal wall at row 5 with two 2-cell doors at columns [2,3] and [7,8]
- Vertical wall at column 5 with two 2-cell doors at rows [2,3] and [7,8]

The step function is **deterministic** (no action noise):
```python
def step(state: int, action: int) -> int:
    di, dj = A_TO_DELTA[action]
    i, j = np.unravel_index(state, walls.shape)
    ni, nj = i + di, j + dj
    if 0 <= ni < HEIGHT and 0 <= nj < WIDTH and walls[ni, nj] == 0:
        return np.ravel_multi_index((ni, nj), walls.shape)
    return state  # blocked: stay in place
```

### 10.2 Tabular Agent: `SGCRLAgent`

Each state has a learnable **embedding vector** `psi[s]` of dimension `rep_dim`:

```python
self.psi = np.empty((nState, rep_dim))  # (100, 16) lookup table
```

**Initialization**:
- Goal embedding: `psi[goal] = randn(rep_dim) * 0.1`, then L2-normalized
- Other states: `psi[s] = psi_goal + randn(rep_dim) * 0.1`, then L2-normalized
- All embeddings start near the goal embedding, breaking symmetry

### 10.3 Action Selection

**Exploration (softmax)**:
```python
def select_action(self, s, g):
    for a in range(self.nAction):
        ns = step(s, a)
        sim = self.psi[ns] @ self.psi[g]  # inner product
        similarities.append(sim)

    inverse_temp = 1.0 / self.entropy_coeff  # default: 1/0.1 = 10
    logits = np.array(similarities) * inverse_temp
    probs = softmax(logits)
    return np.random.choice(self.nAction, p=probs)
```

**Evaluation (greedy)**:
```python
def eval_action(self, s, g):
    # Deterministic: pick action with max psi(s') . psi(g)
    best_a = argmax_a(psi[step(s,a)] @ psi[g])
```

### 10.4 Contrastive Update (InfoNCE Gradient)

The `update_representations()` method implements the **column-wise softmax contrastive loss** with explicit gradient computation:

```python
def update_representations(self):
    # Sample (s, s+) pairs from replay using geometric discounting
    for idx in traj_ids:
        traj = self.replay[idx]
        i = random_int(0, len(traj)-1)
        w = gamma ** np.arange(remaining)  # Geometric weights
        j = i + np.random.choice(remaining, p=w/w.sum())
        s_list.append(traj[i])
        sp_list.append(traj[j])

    psi_s = self.psi[s_batch]     # (B, rep_dim)
    psi_p = self.psi[sp_batch]    # (B, rep_dim)

    # Column-wise softmax: P[i,j] = exp(psi_s[i] . psi_p[j]) / sum_k exp(psi_s[k] . psi_p[j])
    dots = psi_s @ psi_p.T
    P = softmax_columnwise(dots)

    # NLL = -mean(log(P[i,i]))  (diagonal = positive pairs)
    nll = -np.mean(np.log(np.diag(P) + 1e-12))

    # Anchor update: delta_psi[s_j] = lr * (I - P) @ psi_p
    coeff = np.eye(B) - P
    anchor_update = lr_psi * (coeff @ psi_p)
    np.add.at(self.psi, s_batch, anchor_update)

    # Positive update: delta_psi[sp_k] = lr * (psi_s - P.T @ psi_s)
    expected_anchor = P.T @ psi_s
    pos_update = lr_psi * (psi_s - expected_anchor)
    np.add.at(self.psi, sp_batch, pos_update)

    # Re-normalize to unit sphere
    self.psi /= np.linalg.norm(self.psi, axis=1, keepdims=True) + 1e-8
```

This is the direct NumPy analog of the CPC loss used in the neural version. The key difference: gradients are computed analytically and applied directly to the lookup table embeddings (no backpropagation through a network).

### 10.5 Tabular Hyperparameters

```python
config = {
    "rep_dim": 16,
    "episodes_per_upd": 1,        # Update every episode
    "lr_psi": 1e-2,               # Learning rate for embeddings
    "replay_capacity": 1000,
    "max_steps": 50,              # Steps per episode
    "batch_size": 128,
    "max_episodes": 500,
    "gamma": 0.99,
    "entropy_coeff": 0.1,         # Softmax temperature = 1/0.1 = 10
    "plot_freq": 1,
    "seed": 42,
}
```

### 10.6 Visualization

The notebook creates an animation showing:
- **Heatmap**: `psi(s) . psi(g)` for all states (cosine similarity to goal)
- **Trajectory overlay**: The evaluation policy's path
- **Success rate curve**: Rolling average over training episodes

This animation demonstrates the "pruning" and "trace formation" mechanisms described in the paper.

---

## 11. Running Experiments

### 11.1 Basic FourRooms Experiment

```bash
python lp_contrastive.py \
    --env point_FourRooms \
    --alg contrastive_cpc \
    --num_steps 2000000 \
    --seed 31
```

### 11.2 Safety Region Experiment

```bash
python lp_contrastive.py \
    --env point_FourRooms \
    --region_bounds=0,5:5,11 \
    --seed 31
```

### 11.3 With Goal as Negative

```bash
python lp_contrastive.py \
    --env point_FourRooms \
    --goal_neg_actor_steps=100000 \
    --seed 31
```

### 11.4 Shell Script: `run_lp_contrastive.sh`

```bash
#!/bin/bash
seeds=(31)
devices=(0)
for idx in "${!seeds[@]}"; do
    CUDA_VISIBLE_DEVICES=$DEV \
    nohup python -u lp_contrastive.py \
        --env point_FourRooms \
        --seed "$SEED" \
        --num_steps 500000 \
        --region_bounds=0,5:5,11 \
        > "lp_contrastive_seed${SEED}.out" 2>&1 &
done
```

### 11.5 Output Structure

```
logs/
  contrastive_cpc_point_FourRooms_31/
    config.json                    # Run configuration (saved automatically)
    learner/                       # Checkpoints
    actor/                         # Actor logs (CSV)
    evaluator/                     # Evaluator logs (CSV)

safety_region_visits_data/
  point_FourRooms_31/
    region_visits_000000-000999.json   # Visitation counts per episode block
    region_visits_001000-001999.json
    ...
```

### 11.6 Launch Mode

Default launch mode is `current_terminal` (multi-threaded, all components in one process). For distributed execution with separate windows:
```python
lp.launch(program, terminal='tmux')
```

---

## 12. File Reference

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `point_env.py` | 2D maze environments | `PointEnv`, `WALLS` dict, `resize_walls()` |
| `env_utils.py` | Environment loading | `load()`, `SawyerBin/Box/Peg` |
| `lp_contrastive.py` | Training entry point | `main()`, `get_program()`, `get_env()`, `fixed_goal_dict` |
| `contrastive/config.py` | Hyperparameter config | `ContrastiveConfig` dataclass |
| `contrastive/networks.py` | Network architectures | `make_networks()`, `ResidualMLP`, `ContrastiveNetworks`, `maximize_q_action()` |
| `contrastive/learning.py` | Training loop | `ContrastiveLearner`, `TrainingState`, `critic_loss()`, `actor_loss()`, `update_step()` |
| `contrastive/builder.py` | Factory methods | `ContrastiveBuilder`, `make_replay_tables()`, `make_dataset_iterator()` |
| `contrastive/agents.py` | Agent assembly | `DistributedContrastive` |
| `contrastive/distributed_layout.py` | Distributed topology | `DistributedLayout`, `default_evaluator_factory()`, `CheckpointingConfig` |
| `contrastive/utils.py` | Observers & wrappers | `SuccessObserver`, `DistanceObserver`, `RegionVisitObserver`, `make_environment()`, `InitiallyRandomActor` |
| `distributional.py` | Policy distributions | `NormalTanhDistribution`, `TanhTransformedDistribution` |
| `default.py` | Logger factory | `make_default_logger()` |
| `tabular_maze.ipynb` | Tabular SGCRL (10x10 FourRooms) | `SGCRLAgent` class |
| `tabular_hanoi.ipynb` | Tabular SGCRL (Hanoi) | `SGCRLAgent` class |
| `run_lp_contrastive.sh` | Experiment launcher | Shell script |
| `requirements.txt` | Dependencies | JAX 0.4.10, dm-acme 0.4.0, Haiku 0.0.9, etc. |

---

## Appendix: Data Flow Diagram

```
                    +------------------+
                    |   Environment    |
                    |  (PointEnv)      |
                    +--------+---------+
                             |
                    obs = [x, y, gx, gy]
                             |
                    +--------v---------+
                    |     Actor(s)     |
                    |  (4 parallel)    |
                    +--------+---------+
                             |
                    episodes (full trajectories)
                             |
                    +--------v---------+
                    |  Replay Buffer   |
                    |  (Reverb)        |
                    +--------+---------+
                             |
                    HER goal relabeling:
                    sample future state as goal
                             |
                    +--------v---------+
                    |     Learner      |
                    |                  |
                    |  1. Critic loss  |
                    |     sa_encoder(s,a) -> phi  (B, 64)
                    |     g_encoder(g)    -> psi  (B, 64)
                    |     logits = phi @ psi.T    (B, B)
                    |     CPC: softmax_cross_entropy(logits, I)
                    |                  |
                    |  2. Actor loss   |
                    |     policy(obs) -> action
                    |     Q = diag(phi(s,a) @ psi(g).T)
                    |     loss = -Q + alpha * log_prob
                    |                  |
                    |  3. Alpha loss   |
                    |     SAC adaptive temperature
                    +--------+---------+
                             |
                    updated params -> Actor(s)
                             |
                    +--------v---------+
                    |    Evaluator     |
                    |  (deterministic) |
                    |  Fixed goals     |
                    |  SuccessObserver |
                    |  DistanceObserver|
                    +------------------+
```
