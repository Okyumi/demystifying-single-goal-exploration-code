"""
Self-contained environment definitions for the isolation study.

Two environment types:
1. ContinualMaze — deterministic maze with phase-changing wall layouts
2. StochasticSuccessMaze — fixed walls, direction-dependent success probability

Wall generators reused from the continual_maze experiment but fully self-contained.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict


# ---------------------------------------------------------------------------
# Wall layout generators
# ---------------------------------------------------------------------------

def _make_walls(height: int, width: int) -> np.ndarray:
    return np.zeros((height, width), dtype=int)


def _add_horizontal_wall(walls: np.ndarray, row: int,
                         doors: List[int]) -> np.ndarray:
    w = walls.copy()
    w[row, :] = 1
    for c in doors:
        if 0 <= c < w.shape[1]:
            w[row, c] = 0
    return w


def _add_vertical_wall(walls: np.ndarray, col: int,
                       doors: List[int]) -> np.ndarray:
    w = walls.copy()
    w[:, col] = 1
    for r in doors:
        if 0 <= r < w.shape[0]:
            w[r, col] = 0
    return w


def create_phase0_walls(h: int = 11, w: int = 11) -> np.ndarray:
    """FourRooms — horizontal wall row 5 (doors 2,8), vertical wall col 5 (doors 2,8)."""
    walls = _make_walls(h, w)
    walls = _add_horizontal_wall(walls, 5, doors=[2, 8])
    walls = _add_vertical_wall(walls, 5, doors=[2, 8])
    return walls


def create_phase1_walls(h: int = 11, w: int = 11) -> np.ndarray:
    """Corridor-shift — horizontal wall row 5 (door 2 only), vertical wall col 5 (door 8 only)."""
    walls = _make_walls(h, w)
    walls = _add_horizontal_wall(walls, 5, doors=[2])
    walls = _add_vertical_wall(walls, 5, doors=[8])
    return walls


def create_phase3_walls(h: int = 11, w: int = 11) -> np.ndarray:
    """L-wall — vertical wall col 3 rows 0-8 (door row 4), horizontal wall row 7 cols 3-10 (door col 8)."""
    walls = _make_walls(h, w)
    for r in range(9):
        walls[r, 3] = 1
    walls[4, 3] = 0
    for c in range(3, 11):
        walls[7, c] = 1
    walls[7, 8] = 0
    return walls


# ---------------------------------------------------------------------------
# Maze phase configuration
# ---------------------------------------------------------------------------

@dataclass
class MazePhase:
    name: str
    walls: np.ndarray
    num_episodes: int
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "num_episodes": int(self.num_episodes),
            "description": self.description,
            "walls": [[int(c) for c in row] for row in self.walls],
        }


# ---------------------------------------------------------------------------
# ContinualMaze
# ---------------------------------------------------------------------------

class ContinualMaze:
    """Tabular maze with phase-changing wall layouts."""

    NUM_ACTIONS = 5  # stay, down, up, right, left
    A_TO_DELTA = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]])

    def __init__(self, phases: List[MazePhase],
                 height: int = 11, width: int = 11,
                 start_state: Optional[int] = None,
                 goal_state: Optional[int] = None):
        assert len(phases) > 0
        self.height = height
        self.width = width
        self.num_states = height * width
        self.phases = phases
        self.start_state = start_state if start_state is not None else 0
        self.goal_state = (goal_state if goal_state is not None
                           else np.ravel_multi_index((height - 1, width - 1),
                                                     (height, width)))
        for p in phases:
            assert p.walls.shape == (height, width)
            si, sj = np.unravel_index(self.start_state, (height, width))
            gi, gj = np.unravel_index(self.goal_state, (height, width))
            assert p.walls[si, sj] == 0, f"Start walled in '{p.name}'"
            assert p.walls[gi, gj] == 0, f"Goal walled in '{p.name}'"

        self._phase_idx = 0
        self._walls = self.phases[0].walls.copy()

    @property
    def current_phase_idx(self) -> int:
        return self._phase_idx

    @property
    def current_phase(self) -> MazePhase:
        return self.phases[self._phase_idx]

    @property
    def walls(self) -> np.ndarray:
        return self._walls

    def set_phase(self, phase_idx: int):
        assert 0 <= phase_idx < len(self.phases)
        self._phase_idx = phase_idx
        self._walls = self.phases[phase_idx].walls.copy()

    def step(self, state: int, action: int) -> int:
        di, dj = self.A_TO_DELTA[action]
        i, j = np.unravel_index(state, (self.height, self.width))
        ni, nj = int(i + di), int(j + dj)
        if (0 <= ni < self.height and 0 <= nj < self.width
                and self._walls[ni, nj] == 0):
            return int(np.ravel_multi_index((ni, nj), (self.height, self.width)))
        return state

    def is_goal(self, state: int) -> bool:
        return state == self.goal_state

    def is_near_goal(self, state: int, threshold: int = 1) -> bool:
        si, sj = np.unravel_index(state, (self.height, self.width))
        gi, gj = np.unravel_index(self.goal_state, (self.height, self.width))
        return abs(int(si) - int(gi)) + abs(int(sj) - int(gj)) <= threshold

    def state_to_coord(self, state: int) -> Tuple[int, int]:
        return tuple(np.unravel_index(state, (self.height, self.width)))

    def coord_to_state(self, row: int, col: int) -> int:
        return int(np.ravel_multi_index((row, col), (self.height, self.width)))

    def get_open_states(self) -> np.ndarray:
        return np.where(self._walls.ravel() == 0)[0]

    def get_config(self) -> dict:
        return {
            "height": int(self.height),
            "width": int(self.width),
            "start_state": int(self.start_state),
            "goal_state": int(self.goal_state),
            "phases": [p.to_dict() for p in self.phases],
        }


# ---------------------------------------------------------------------------
# StochasticSuccessMaze
# ---------------------------------------------------------------------------

class StochasticSuccessMaze:
    """Tabular maze where success depends on approach direction to the goal."""

    NUM_ACTIONS = 5
    A_TO_DELTA = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]])

    _MOVE_TO_DIR = {
        (1, 0): "from_above",
        (-1, 0): "from_below",
        (0, 1): "from_left",
        (0, -1): "from_right",
        (0, 0): "stay",
    }

    def __init__(self, walls: np.ndarray,
                 height: int = 11, width: int = 11,
                 start_state: Optional[int] = None,
                 goal_state: Optional[int] = None,
                 approach_probs: Optional[Dict[str, float]] = None):
        self.height = height
        self.width = width
        self.num_states = height * width
        self.walls = walls.copy()
        self.start_state = start_state if start_state is not None else 0
        self.goal_state = (goal_state if goal_state is not None
                           else np.ravel_multi_index((height - 1, width - 1),
                                                     (height, width)))
        self.approach_probs = approach_probs or {
            "from_above": 0.10,
            "from_left":  0.25,
            "from_right": 0.50,
            "from_below": 0.80,
            "stay":       0.00,
        }
        self.phases = []

    def step(self, state: int, action: int) -> int:
        di, dj = self.A_TO_DELTA[action]
        i, j = np.unravel_index(state, (self.height, self.width))
        ni, nj = int(i + di), int(j + dj)
        if (0 <= ni < self.height and 0 <= nj < self.width
                and self.walls[ni, nj] == 0):
            return int(np.ravel_multi_index((ni, nj), (self.height, self.width)))
        return state

    def is_goal(self, state: int) -> bool:
        return state == self.goal_state

    def is_near_goal(self, state: int, threshold: int = 1) -> bool:
        si, sj = np.unravel_index(state, (self.height, self.width))
        gi, gj = np.unravel_index(self.goal_state, (self.height, self.width))
        return abs(int(si) - int(gi)) + abs(int(sj) - int(gj)) <= threshold

    def approach_direction(self, prev_state: int, cur_state: int) -> str:
        pi, pj = np.unravel_index(prev_state, (self.height, self.width))
        ci, cj = np.unravel_index(cur_state, (self.height, self.width))
        dr, dc = int(ci - pi), int(cj - pj)
        return self._MOVE_TO_DIR.get((dr, dc), "stay")

    def success_probability(self, approach_dir: str) -> float:
        return self.approach_probs.get(approach_dir, 0.0)

    def check_stochastic_success(self, prev_state: int,
                                  cur_state: int) -> Tuple[bool, str, float]:
        direction = self.approach_direction(prev_state, cur_state)
        prob = self.success_probability(direction)
        success = np.random.random() < prob
        return success, direction, prob

    def state_to_coord(self, state: int) -> Tuple[int, int]:
        return tuple(np.unravel_index(state, (self.height, self.width)))

    def coord_to_state(self, row: int, col: int) -> int:
        return int(np.ravel_multi_index((row, col), (self.height, self.width)))

    def get_open_states(self) -> np.ndarray:
        return np.where(self.walls.ravel() == 0)[0]

    def get_config(self) -> dict:
        return {
            "type": "stochastic_success",
            "height": int(self.height),
            "width": int(self.width),
            "start_state": int(self.start_state),
            "goal_state": int(self.goal_state),
            "approach_probs": self.approach_probs,
            "walls": [[int(c) for c in row] for row in self.walls],
        }


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def make_fourrooms_maze(goal_state: Optional[int] = None) -> ContinualMaze:
    """Single-phase FourRooms maze for baseline conditions."""
    walls = create_phase0_walls()
    goal = goal_state or np.ravel_multi_index((10, 10), (11, 11))
    phases = [MazePhase(name="fourrooms", walls=walls,
                        num_episodes=10000,
                        description="Standard FourRooms")]
    return ContinualMaze(phases=phases, height=11, width=11,
                         start_state=0, goal_state=goal)


def make_stochastic_fourrooms_maze(
    approach_probs: Optional[Dict[str, float]] = None,
) -> StochasticSuccessMaze:
    """FourRooms maze with stochastic direction-dependent success."""
    walls = create_phase0_walls()
    return StochasticSuccessMaze(
        walls=walls, height=11, width=11,
        start_state=0,
        goal_state=np.ravel_multi_index((10, 10), (11, 11)),
        approach_probs=approach_probs,
    )


def make_changing_dynamics_maze(episodes_per_phase: int = 5000) -> ContinualMaze:
    """3-phase changing dynamics: fourrooms -> corridor_shift -> l_wall."""
    goal = np.ravel_multi_index((10, 10), (11, 11))
    phases = [
        MazePhase(
            name="fourrooms",
            walls=create_phase0_walls(),
            num_episodes=episodes_per_phase,
            description="Standard FourRooms: doors at (2,5),(8,5),(5,2),(5,8)",
        ),
        MazePhase(
            name="corridor_shift",
            walls=create_phase1_walls(),
            num_episodes=episodes_per_phase,
            description="Corridor-shift: doors at (2,5),(5,8) only",
        ),
        MazePhase(
            name="l_wall",
            walls=create_phase3_walls(),
            num_episodes=episodes_per_phase,
            description="L-wall: bottleneck at (4,3) and (7,8)",
        ),
    ]
    return ContinualMaze(phases=phases, height=11, width=11,
                         start_state=0, goal_state=goal)


def make_static_fourrooms(goal_state: Optional[int] = None) -> ContinualMaze:
    """Static FourRooms for 15000 episodes (baseline for Study B)."""
    walls = create_phase0_walls()
    goal = goal_state or np.ravel_multi_index((10, 10), (11, 11))
    phases = [MazePhase(name="fourrooms", walls=walls,
                        num_episodes=15000,
                        description="Static FourRooms baseline")]
    return ContinualMaze(phases=phases, height=11, width=11,
                         start_state=0, goal_state=goal)
