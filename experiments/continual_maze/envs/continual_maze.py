"""
ContinualMaze environment for testing SGCRL adaptation to changing dynamics.

The goal position is FIXED across all phases. The maze wall layout CHANGES
at configurable training step intervals. Each phase has a different maze
structure but same goal, forcing the agent to adapt its path.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Wall layout generators
# ---------------------------------------------------------------------------

def create_fourrooms_walls(height: int = 10, width: int = 10,
                           door_len: int = 2) -> np.ndarray:
    """Standard four-rooms layout with horizontal and vertical walls."""
    walls = np.zeros((height, width), dtype=int)
    # Horizontal wall at midpoint
    walls[height // 2, :] = 1
    doors_h = np.concatenate([
        width // 4 + np.arange(door_len),
        width * 3 // 4 + np.arange(door_len),
    ])
    walls[height // 2, doors_h] = 0
    # Vertical wall at midpoint
    walls[:, width // 2] = 1
    doors_v = np.concatenate([
        height // 4 + np.arange(door_len),
        height * 3 // 4 + np.arange(door_len),
    ])
    walls[doors_v, width // 2] = 0
    return walls


def create_shortcut_walls(height: int = 10, width: int = 10,
                          door_len: int = 2) -> np.ndarray:
    """Four-rooms with an extra shortcut: a gap opened in the horizontal wall
    near column 0, creating a direct path from top-left to bottom-left."""
    walls = create_fourrooms_walls(height, width, door_len)
    # Open a 2-cell gap at the left end of the horizontal wall
    walls[height // 2, 0:2] = 0
    return walls


def create_reroute_walls(height: int = 10, width: int = 10,
                         door_len: int = 2) -> np.ndarray:
    """Four-rooms where the original bottom-right door in the vertical wall
    is BLOCKED, forcing use of a new gap at bottom row of the vertical wall."""
    walls = create_fourrooms_walls(height, width, door_len)
    # Block the original lower-right door in the vertical wall (rows 7,8)
    walls[height * 3 // 4: height * 3 // 4 + door_len, width // 2] = 1
    # Open a new gap at the very bottom of the vertical wall
    walls[height - 2: height, width // 2] = 0
    return walls


# ---------------------------------------------------------------------------
# Pre-built wall configs
# ---------------------------------------------------------------------------

FOUR_ROOMS_STANDARD = create_fourrooms_walls()
FOUR_ROOMS_SHORTCUT = create_shortcut_walls()
FOUR_ROOMS_REROUTE = create_reroute_walls()


# ---------------------------------------------------------------------------
# Maze phase configuration
# ---------------------------------------------------------------------------

@dataclass
class MazePhase:
    """One phase of the continual maze experiment."""
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
# ContinualMaze environment
# ---------------------------------------------------------------------------

class ContinualMaze:
    """A tabular maze environment with phase-changing wall layouts.

    Parameters
    ----------
    phases : list of MazePhase
        Ordered list of maze configurations and their durations.
    height, width : int
        Grid dimensions (must be consistent across all phase wall arrays).
    start_state : int
        Flattened index of the start position (fixed across phases).
    goal_state : int
        Flattened index of the goal position (fixed across phases).
    """

    NUM_ACTIONS = 5  # stay, down, up, right, left
    A_TO_DELTA = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]])

    def __init__(self, phases: List[MazePhase],
                 height: int = 10, width: int = 10,
                 start_state: Optional[int] = None,
                 goal_state: Optional[int] = None):
        assert len(phases) > 0, "Need at least one phase"
        self.height = height
        self.width = width
        self.num_states = height * width
        self.phases = phases

        # Default start/goal: top-left / bottom-right
        self.start_state = start_state if start_state is not None else 0
        self.goal_state = (goal_state if goal_state is not None
                           else np.ravel_multi_index((height - 1, width - 1),
                                                     (height, width)))

        # Validate all wall arrays
        for p in phases:
            assert p.walls.shape == (height, width), \
                f"Phase '{p.name}' walls shape {p.walls.shape} != ({height}, {width})"
            # Ensure start and goal are not walled off
            si, sj = np.unravel_index(self.start_state, (height, width))
            gi, gj = np.unravel_index(self.goal_state, (height, width))
            assert p.walls[si, sj] == 0, f"Start is walled in phase '{p.name}'"
            assert p.walls[gi, gj] == 0, f"Goal is walled in phase '{p.name}'"

        # Start in phase 0
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
        """Switch to a specific phase."""
        assert 0 <= phase_idx < len(self.phases)
        self._phase_idx = phase_idx
        self._walls = self.phases[phase_idx].walls.copy()

    def advance_phase(self) -> bool:
        """Advance to the next phase. Returns True if there is a next phase."""
        if self._phase_idx + 1 < len(self.phases):
            self.set_phase(self._phase_idx + 1)
            return True
        return False

    # -- Dynamics -----------------------------------------------------------

    def step(self, state: int, action: int) -> int:
        """Deterministic step in the current wall layout."""
        di, dj = self.A_TO_DELTA[action]
        i, j = np.unravel_index(state, (self.height, self.width))
        ni, nj = int(i + di), int(j + dj)
        if 0 <= ni < self.height and 0 <= nj < self.width and self._walls[ni, nj] == 0:
            return int(np.ravel_multi_index((ni, nj), (self.height, self.width)))
        return state  # blocked

    def is_goal(self, state: int) -> bool:
        """Exact goal check."""
        return state == self.goal_state

    def is_near_goal(self, state: int, threshold: int = 1) -> bool:
        """Manhattan-distance goal proximity check."""
        si, sj = np.unravel_index(state, (self.height, self.width))
        gi, gj = np.unravel_index(self.goal_state, (self.height, self.width))
        return abs(int(si) - int(gi)) + abs(int(sj) - int(gj)) <= threshold

    def state_to_coord(self, state: int) -> Tuple[int, int]:
        return tuple(np.unravel_index(state, (self.height, self.width)))

    def coord_to_state(self, row: int, col: int) -> int:
        return int(np.ravel_multi_index((row, col), (self.height, self.width)))

    # -- Utility ------------------------------------------------------------

    def get_open_states(self) -> np.ndarray:
        """Return flat indices of all non-wall states in current phase."""
        return np.where(self._walls.ravel() == 0)[0]

    def get_config(self) -> dict:
        """Serialisable configuration for reproducibility."""
        return {
            "height": int(self.height),
            "width": int(self.width),
            "start_state": int(self.start_state),
            "goal_state": int(self.goal_state),
            "phases": [p.to_dict() for p in self.phases],
        }


# ---------------------------------------------------------------------------
# Convenience: default 3-phase continual maze
# ---------------------------------------------------------------------------

def make_default_continual_maze(episodes_per_phase: int = 500) -> ContinualMaze:
    """Create the default 3-phase continual maze experiment.

    Phase 1 (Standard FourRooms):
        Agent must navigate through the canonical door sequence.
    Phase 2 (Shortcut):
        A new gap opens in the horizontal wall near column 0,
        creating a shorter path. Does the agent switch?
    Phase 3 (Reroute):
        The lower-right vertical-wall door is blocked and a new
        gap opens at the bottom. The agent must re-explore.
    """
    phases = [
        MazePhase(
            name="standard",
            walls=FOUR_ROOMS_STANDARD,
            num_episodes=episodes_per_phase,
            description="Standard FourRooms — canonical door sequence",
        ),
        MazePhase(
            name="shortcut",
            walls=FOUR_ROOMS_SHORTCUT,
            num_episodes=episodes_per_phase,
            description="Shortcut opened at left side of horizontal wall",
        ),
        MazePhase(
            name="reroute",
            walls=FOUR_ROOMS_REROUTE,
            num_episodes=episodes_per_phase,
            description="Lower-right door blocked, new gap at bottom of vertical wall",
        ),
    ]
    return ContinualMaze(phases=phases)
