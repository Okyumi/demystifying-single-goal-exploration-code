"""
Continual maze environments for testing SGCRL adaptation.

Two environment classes:
1. ContinualMaze — deterministic maze with phase-changing wall layouts
2. StochasticSuccessMaze — fixed walls, but success probability depends on
   the direction from which the agent approaches the goal

In both cases the goal position is FIXED across all phases.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict


# ---------------------------------------------------------------------------
# Wall layout generators  (on a height×width grid, 1=wall, 0=open)
# ---------------------------------------------------------------------------

def _make_walls(height: int, width: int) -> np.ndarray:
    """Return an all-open grid."""
    return np.zeros((height, width), dtype=int)


def _add_horizontal_wall(walls: np.ndarray, row: int,
                         doors: List[int]) -> np.ndarray:
    """Fill an entire row with wall, then punch holes at *doors* columns."""
    w = walls.copy()
    w[row, :] = 1
    for c in doors:
        if 0 <= c < w.shape[1]:
            w[row, c] = 0
    return w


def _add_vertical_wall(walls: np.ndarray, col: int,
                       doors: List[int]) -> np.ndarray:
    """Fill an entire column with wall, then punch holes at *doors* rows."""
    w = walls.copy()
    w[:, col] = 1
    for r in doors:
        if 0 <= r < w.shape[0]:
            w[r, col] = 0
    return w


# --- 6 increasingly-different FourRooms variants -------------------------

def create_phase0_walls(h: int = 11, w: int = 11) -> np.ndarray:
    """Phase 0 – Standard FourRooms.
    Horizontal wall at row 5 with doors at cols 2,8.
    Vertical wall at col 5 with doors at rows 2,8.
    Only path from (0,0) to (10,10): top-left → top-right → bottom-right."""
    walls = _make_walls(h, w)
    walls = _add_horizontal_wall(walls, 5, doors=[2, 8])
    walls = _add_vertical_wall(walls, 5, doors=[2, 8])
    return walls


def create_phase1_walls(h: int = 11, w: int = 11) -> np.ndarray:
    """Phase 1 – Corridor shift.
    Horizontal wall keeps door at col 2 only (col 8 blocked).
    Vertical wall keeps door at row 8 only (row 2 blocked).
    Forces a completely different route: top-left → bottom-left → bottom-right."""
    walls = _make_walls(h, w)
    walls = _add_horizontal_wall(walls, 5, doors=[2])
    walls = _add_vertical_wall(walls, 5, doors=[8])
    return walls


def create_phase2_walls(h: int = 11, w: int = 11) -> np.ndarray:
    """Phase 2 – Narrow spiral.
    Three horizontal barriers create a zig-zag path.
    Agent must go right→down→left→down→right to reach (10,10)."""
    walls = _make_walls(h, w)
    # Barrier at row 3: open only at col 9-10 (right side)
    walls = _add_horizontal_wall(walls, 3, doors=[9, 10])
    # Barrier at row 6: open only at col 0-1 (left side)
    walls = _add_horizontal_wall(walls, 6, doors=[0, 1])
    # Barrier at row 9: open only at col 9-10 (right side)
    walls = _add_horizontal_wall(walls, 9, doors=[9, 10])
    return walls


def create_phase3_walls(h: int = 11, w: int = 11) -> np.ndarray:
    """Phase 3 – L-wall with bottleneck.
    Vertical wall from row 0-8 at col 3; horizontal wall from col 3-10 at row 7.
    Single door at (7, 3) and (4, 3). Forces a tight L-shaped detour."""
    walls = _make_walls(h, w)
    # Vertical wall col 3, rows 0-8, doors at rows 4
    for r in range(9):
        walls[r, 3] = 1
    walls[4, 3] = 0
    # Horizontal wall row 7, cols 3-10, door at col 8
    for c in range(3, 11):
        walls[7, c] = 1
    walls[7, 8] = 0
    return walls


def create_phase4_walls(h: int = 11, w: int = 11) -> np.ndarray:
    """Phase 4 – Center block maze.
    A 3×3 block in the centre forces the agent around it.
    Plus vertical wall at col 8 with single door at row 9."""
    walls = _make_walls(h, w)
    # Center block rows 4-6, cols 4-6
    walls[4:7, 4:7] = 1
    # Vertical wall col 8 with door at row 9
    walls = _add_vertical_wall(walls, 8, doors=[9, 0, 1, 2, 3])
    return walls


def create_phase5_walls(h: int = 11, w: int = 11) -> np.ndarray:
    """Phase 5 – Double spiral.
    Two vertical walls plus one horizontal wall create a winding path.
    Agent must zig-zag extensively."""
    walls = _make_walls(h, w)
    # Vertical wall col 3, door only at row 10 (bottom)
    for r in range(10):
        walls[r, 3] = 1
    walls[10, 3] = 0
    # Vertical wall col 7, door only at row 0 (top)
    for r in range(1, 11):
        walls[r, 7] = 1
    walls[0, 7] = 0
    # Horizontal wall row 5, cols 3-7, door at col 5
    for c in range(3, 8):
        walls[5, c] = 1
    walls[5, 5] = 0
    return walls


# ---------------------------------------------------------------------------
# Maze phase configuration
# ---------------------------------------------------------------------------

@dataclass
class MazePhase:
    """One phase of a continual maze experiment."""
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
# ContinualMaze  (deterministic dynamics, changing walls)
# ---------------------------------------------------------------------------

class ContinualMaze:
    """Tabular maze with phase-changing wall layouts.

    Parameters
    ----------
    phases : list of MazePhase
    height, width : grid dimensions
    start_state, goal_state : flat indices (fixed across phases)
    """

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
            assert p.walls.shape == (height, width), \
                f"Phase '{p.name}' walls {p.walls.shape} != ({height},{width})"
            si, sj = np.unravel_index(self.start_state, (height, width))
            gi, gj = np.unravel_index(self.goal_state, (height, width))
            assert p.walls[si, sj] == 0, f"Start walled in '{p.name}'"
            assert p.walls[gi, gj] == 0, f"Goal walled in '{p.name}'"

        self._phase_idx = 0
        self._walls = self.phases[0].walls.copy()

    # -- Phase management ---------------------------------------------------

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

    def advance_phase(self) -> bool:
        if self._phase_idx + 1 < len(self.phases):
            self.set_phase(self._phase_idx + 1)
            return True
        return False

    # -- Dynamics -----------------------------------------------------------

    def step(self, state: int, action: int) -> int:
        """Deterministic one-step transition."""
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
# StochasticSuccessMaze  (fixed walls, stochastic goal success)
# ---------------------------------------------------------------------------

class StochasticSuccessMaze:
    """Tabular maze where success depends on the *approach direction* to the goal.

    The agent can physically reach the goal cell from any direction, but
    whether that counts as a "successful" episode is probabilistic:
    each approach direction has its own success probability.

    This lets us test whether SGCRL overexploits the first partially-
    successful route it discovers, even when better routes exist.

    Parameters
    ----------
    walls : np.ndarray
        Static wall layout (does not change).
    height, width : grid dimensions
    start_state, goal_state : flat indices
    approach_probs : dict mapping direction name → success probability
        Directions: "from_above" (agent arrives from row-1),
                    "from_below" (row+1), "from_left" (col-1),
                    "from_right" (col+1), "stay" (already on goal).
    """

    NUM_ACTIONS = 5  # stay, down, up, right, left
    A_TO_DELTA = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]])

    # Map (delta_row, delta_col) of the agent's movement to a direction name
    _MOVE_TO_DIR = {
        (1, 0): "from_above",   # agent moved down → arrived from above
        (-1, 0): "from_below",  # agent moved up → arrived from below
        (0, 1): "from_left",    # agent moved right → arrived from left
        (0, -1): "from_right",  # agent moved left → arrived from right
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

        # Default approach probabilities — deliberately asymmetric
        self.approach_probs = approach_probs or {
            "from_above": 0.10,   # hardest
            "from_left":  0.25,
            "from_right": 0.50,
            "from_below": 0.80,   # easiest
            "stay":       0.00,
        }

        # No phase management needed (static env), but expose the
        # same interface as ContinualMaze for agent compatibility
        self.phases = []

    # -- Dynamics -----------------------------------------------------------

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
        """Determine which direction the agent approached cur_state from."""
        pi, pj = np.unravel_index(prev_state, (self.height, self.width))
        ci, cj = np.unravel_index(cur_state, (self.height, self.width))
        dr, dc = int(ci - pi), int(cj - pj)
        return self._MOVE_TO_DIR.get((dr, dc), "stay")

    def success_probability(self, approach_dir: str) -> float:
        return self.approach_probs.get(approach_dir, 0.0)

    def check_stochastic_success(self, prev_state: int,
                                  cur_state: int) -> Tuple[bool, str, float]:
        """Check whether arriving at cur_state from prev_state counts as success.

        Returns (success_bool, direction_name, probability).
        """
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

def make_continual_maze_v2(episodes_per_phase: int = 500) -> ContinualMaze:
    """6-phase continual maze on an 11×11 grid.

    Each phase forces a substantially different optimal path from (0,0) to (10,10).
    """
    phases = [
        MazePhase(
            name="fourrooms",
            walls=create_phase0_walls(),
            num_episodes=episodes_per_phase,
            description="Standard FourRooms: route via top-right and bottom-right rooms",
        ),
        MazePhase(
            name="corridor_shift",
            walls=create_phase1_walls(),
            num_episodes=episodes_per_phase,
            description="Doors moved: forces route via bottom-left room instead",
        ),
        MazePhase(
            name="zigzag",
            walls=create_phase2_walls(),
            num_episodes=episodes_per_phase,
            description="Three horizontal barriers → right-down-left-down-right zig-zag",
        ),
        MazePhase(
            name="l_wall",
            walls=create_phase3_walls(),
            num_episodes=episodes_per_phase,
            description="L-shaped wall with bottleneck door: tight detour path",
        ),
        MazePhase(
            name="center_block",
            walls=create_phase4_walls(),
            num_episodes=episodes_per_phase,
            description="3×3 center block + right-side wall: navigate around obstacles",
        ),
        MazePhase(
            name="double_spiral",
            walls=create_phase5_walls(),
            num_episodes=episodes_per_phase,
            description="Two vertical walls + horizontal: long winding spiral path",
        ),
    ]
    return ContinualMaze(phases=phases, height=11, width=11)


def make_stochastic_maze(
    height: int = 11, width: int = 11,
    approach_probs: Optional[Dict[str, float]] = None,
) -> StochasticSuccessMaze:
    """Open maze (no internal walls) with stochastic direction-dependent success.

    The maze is intentionally open so that multiple approach routes exist
    and the agent must discover which approach direction has highest success probability.
    """
    walls = _make_walls(height, width)
    return StochasticSuccessMaze(
        walls=walls,
        height=height,
        width=width,
        start_state=0,
        goal_state=np.ravel_multi_index((height - 1, width - 1),
                                         (height, width)),
        approach_probs=approach_probs,
    )


# Keep old name for backward compatibility
def make_default_continual_maze(episodes_per_phase: int = 500) -> ContinualMaze:
    return make_continual_maze_v2(episodes_per_phase)


# Pre-built wall arrays for convenience
FOUR_ROOMS_STANDARD = create_phase0_walls()
FOUR_ROOMS_SHORTCUT = create_phase0_walls()  # placeholder
FOUR_ROOMS_REROUTE = create_phase0_walls()   # placeholder
