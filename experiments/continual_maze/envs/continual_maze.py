"""
ContinualMaze environment for testing SGCRL adaptation.

The maze keeps a FIXED goal but CHANGES wall layout at configurable
training-step intervals.  Each phase has a different maze structure,
forcing the agent to adapt its path.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Maze construction helpers
# ---------------------------------------------------------------------------

def build_fourrooms_walls(height: int = 10, width: int = 10,
                          door_len: int = 2) -> np.ndarray:
    """Standard FourRooms wall layout (Phase 1).

    Horizontal wall at row height//2, vertical wall at col width//2,
    with two door openings in each wall.
    """
    walls = np.zeros((height, width), dtype=int)

    # Horizontal wall
    walls[height // 2, :] = 1
    doors_h = np.concatenate([
        width // 4 + np.arange(door_len),
        width * 3 // 4 + np.arange(door_len),
    ]).astype(int)
    walls[height // 2, doors_h] = 0

    # Vertical wall
    walls[:, width // 2] = 1
    doors_v = np.concatenate([
        height // 4 + np.arange(door_len),
        height * 3 // 4 + np.arange(door_len),
    ]).astype(int)
    walls[doors_v, width // 2] = 0

    return walls


def build_shortcut_walls(height: int = 10, width: int = 10,
                         door_len: int = 2) -> np.ndarray:
    """Phase 2: FourRooms + remove part of the vertical wall in the
    bottom-right quadrant, creating a shorter path from start to goal.

    Opens a direct corridor through the vertical wall at rows 6-8
    (below horizontal wall), allowing the agent to go right from the
    bottom-left room directly into the bottom-right room without
    needing the existing doors.
    """
    walls = build_fourrooms_walls(height, width, door_len)
    # Open a wide passage in the vertical wall for rows 6,7,8
    # (the bottom half, below horizontal wall at row 5)
    walls[6, width // 2] = 0
    walls[7, width // 2] = 0
    walls[8, width // 2] = 0
    return walls


def build_blocked_walls(height: int = 10, width: int = 10,
                        door_len: int = 2) -> np.ndarray:
    """Phase 3: Block the doors in the bottom half of the vertical wall
    and open a new path through the top half.

    This forces the agent to find a path through the top rooms
    instead of the bottom rooms.
    """
    walls = build_fourrooms_walls(height, width, door_len)

    # Block the bottom-half vertical-wall doors (rows 7,8)
    walls[7, width // 2] = 1
    walls[8, width // 2] = 1

    # Block the bottom horizontal-wall doors (cols 7,8)
    walls[height // 2, 7] = 1
    walls[height // 2, 8] = 1

    # Open a wide passage in the vertical wall for rows 0-3 (top half)
    walls[0, width // 2] = 0
    walls[1, width // 2] = 0

    # Open a wide passage in the horizontal wall near the right side
    # so agent can descend from top-right to bottom-right room
    walls[height // 2, 8] = 0
    walls[height // 2, 9] = 0

    return walls


# ---------------------------------------------------------------------------
# Phase description
# ---------------------------------------------------------------------------

@dataclass
class MazePhase:
    """One phase of the continual maze experiment."""
    name: str
    walls: np.ndarray
    num_episodes: int
    description: str = ""


# ---------------------------------------------------------------------------
# ContinualMaze environment
# ---------------------------------------------------------------------------

class ContinualMaze:
    """Tabular maze whose wall layout changes across phases.

    Parameters
    ----------
    height, width : int
        Grid dimensions.
    start_state : int
        Flat index of the start position.
    goal_state : int
        Flat index of the goal position (fixed across all phases).
    phases : list[MazePhase]
        Ordered list of phases.  Each carries its own wall array and
        a duration in training episodes.
    """

    NUM_ACTIONS = 5
    A_TO_DELTA = np.array([[0, 0],   # stay
                           [1, 0],   # down
                           [-1, 0],  # up
                           [0, 1],   # right
                           [0, -1]]) # left

    def __init__(
        self,
        height: int,
        width: int,
        start_state: int,
        goal_state: int,
        phases: List[MazePhase],
    ):
        self.height = height
        self.width = width
        self.start_state = start_state
        self.goal_state = goal_state
        self.phases = phases
        self.num_states = height * width
        self.shape = (height, width)

        # Current phase bookkeeping
        self._phase_idx = 0
        self._episodes_in_phase = 0
        self._total_episodes = 0
        self.walls = self.phases[0].walls.copy()

        # Log of phase transitions: list of (episode, phase_idx)
        self.phase_log: List[Tuple[int, int]] = [(0, 0)]

    # ------------------------------------------------------------------
    # Phase management
    # ------------------------------------------------------------------

    @property
    def current_phase(self) -> MazePhase:
        return self.phases[self._phase_idx]

    @property
    def phase_idx(self) -> int:
        return self._phase_idx

    @property
    def total_episodes(self) -> int:
        return self._total_episodes

    def advance_episode(self) -> bool:
        """Call after each training episode.

        Returns True if a phase transition occurred.
        """
        self._total_episodes += 1
        self._episodes_in_phase += 1

        if (self._episodes_in_phase >= self.current_phase.num_episodes
                and self._phase_idx < len(self.phases) - 1):
            self._phase_idx += 1
            self._episodes_in_phase = 0
            self.walls = self.phases[self._phase_idx].walls.copy()
            self.phase_log.append((self._total_episodes, self._phase_idx))
            return True
        return False

    def total_num_episodes(self) -> int:
        return sum(p.num_episodes for p in self.phases)

    # ------------------------------------------------------------------
    # Transition dynamics
    # ------------------------------------------------------------------

    def step(self, state: int, action: int) -> int:
        """Deterministic transition."""
        di, dj = self.A_TO_DELTA[action]
        i, j = np.unravel_index(state, self.shape)
        ni, nj = int(i + di), int(j + dj)
        if 0 <= ni < self.height and 0 <= nj < self.width and self.walls[ni, nj] == 0:
            return int(np.ravel_multi_index((ni, nj), self.shape))
        return state

    def is_near_goal(self, state: int, dist: int = 1) -> bool:
        si, sj = np.unravel_index(state, self.shape)
        gi, gj = np.unravel_index(self.goal_state, self.shape)
        return abs(int(si) - int(gi)) + abs(int(sj) - int(gj)) <= dist

    def state_to_coord(self, state: int) -> Tuple[int, int]:
        return tuple(np.unravel_index(state, self.shape))

    def coord_to_state(self, row: int, col: int) -> int:
        return int(np.ravel_multi_index((row, col), self.shape))

    # ------------------------------------------------------------------
    # Convenience builders
    # ------------------------------------------------------------------

    @classmethod
    def make_default(
        cls,
        episodes_per_phase: int = 500,
        height: int = 10,
        width: int = 10,
    ) -> "ContinualMaze":
        """Build the standard 3-phase continual maze."""
        start = int(np.ravel_multi_index((0, 0), (height, width)))
        goal = int(np.ravel_multi_index((9, 9), (height, width)))

        phases = [
            MazePhase(
                name="fourrooms",
                walls=build_fourrooms_walls(height, width),
                num_episodes=episodes_per_phase,
                description="Standard FourRooms layout",
            ),
            MazePhase(
                name="shortcut",
                walls=build_shortcut_walls(height, width),
                num_episodes=episodes_per_phase,
                description="Shortcut opened in vertical wall (bottom half)",
            ),
            MazePhase(
                name="blocked",
                walls=build_blocked_walls(height, width),
                num_episodes=episodes_per_phase,
                description="Bottom path blocked, top path opened",
            ),
        ]

        return cls(
            height=height,
            width=width,
            start_state=start,
            goal_state=goal,
            phases=phases,
        )
