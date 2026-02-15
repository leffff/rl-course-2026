import numpy as np
import matplotlib.pyplot as plt

class MazeEnv:
    """
    maze : np.ndarray (H, W), dtype int
        1 = wall/obstacle, 0 = free cell.
    start : tuple (row, col)
        Starting position. Must be a free cell.
    goal : tuple (row, col)
        Goal (terminal) position. Must be a free cell.
    max_steps : int
        Episode is truncated after this many steps.
    step_reward : float
        Reward given at every non-terminal step (typically -1).
    goal_reward : float
        Bonus reward for reaching the goal.
    """

    ACTIONS = {0: (-1, 0),   # up
               1: ( 1, 0),   # down
               2: ( 0, -1),  # left
               3: ( 0,  1)}  # right

    def __init__(
        self,
        maze: np.ndarray,
        start: tuple, goal: tuple,
        max_steps: int = 200,
        step_reward: float = -1.0,
        goal_reward: float = 0.0
    ):
        self.maze = maze.copy()
        self.H, self.W = maze.shape
        self.start = start
        self.goal = goal
        self.max_steps = max_steps
        self.step_reward = step_reward
        self.goal_reward = goal_reward

        assert maze[start] == 0, "start must be a free cell"
        assert maze[goal] == 0, "goal must be a free cell"

        # free cells → state indices (only free cells get an index)
        self._pos_to_state = {}
        self._state_to_pos = {}
        idx = 0
        for r in range(self.H):
            for c in range(self.W):
                if maze[r, c] == 0:
                    self._pos_to_state[(r, c)] = idx
                    self._state_to_pos[idx] = (r, c)
                    idx += 1

        self.n_states = idx
        self.n_actions = len(self.ACTIONS)
        self.terminal_state = self._pos_to_state[goal]

        self._agent_pos = None
        self._step_count = 0

    def reset(self) -> int:
        """Reset env, return initial state index."""
        self._agent_pos = self.start
        self._step_count = 0
        return self._pos_to_state[self.start]

    def step(self, action: int) -> tuple:
        """
        Execute action.
        Returns (next_state, reward, done).
        """
        r, c = self._agent_pos
        dr, dc = self.ACTIONS[action]
        nr, nc = r + dr, c + dc

        # stay in place if hitting wall or boundary
        if not (0 <= nr < self.H and 0 <= nc < self.W) or self.maze[nr, nc] == 1:
            nr, nc = r, c

        self._agent_pos = (nr, nc)
        self._step_count += 1
        state = self._pos_to_state[(nr, nc)]

        if (nr, nc) == self.goal:
            return state, self.goal_reward, True

        if self._step_count >= self.max_steps:
            return state, self.step_reward, True

        return state, self.step_reward, False

    def state_to_pos(self, state: int) -> tuple:
        return self._state_to_pos[state]

    def pos_to_state(self, pos: tuple) -> int:
        return self._pos_to_state[pos]

def plot_maze(env):
    fig, ax = plt.subplots(figsize=(env.W, env.H))

    ax.imshow(env.maze, cmap="Greys", vmin=0, vmax=1)

    for x in range(env.W + 1):
        ax.axvline(x - 0.5, color="grey", linewidth=0.5)
    for y in range(env.H + 1):
        ax.axhline(y - 0.5, color="grey", linewidth=0.5)

    sr, sc = env.start
    gr, gc = env.goal
    ax.plot(sc, sr, "gs", markersize=14, label="Start")
    ax.plot(gc, gr, "r*", markersize=18, label="Goal")

    ax.set_xticks(range(env.W))
    ax.set_yticks(range(env.H))

    return ax


class MazeEnvWithDistanceReward(MazeEnv):
    def __init__(self, maze, start, goal, max_steps = 200, step_reward = -1, goal_reward = 0):
        super().__init__(maze, start, goal, max_steps, step_reward, goal_reward)

        self.start_goal_distance = np.abs(self.start[0] - self.goal[0]) + np.abs(self.start[1] - self.goal[1])

    def step(self, action: int) -> tuple:
        """
        Execute action.
        Returns (next_state, reward, done).
        """
        r, c = self._agent_pos
        dr, dc = self.ACTIONS[action]
        nr, nc = r + dr, c + dc

        # stay in place if hitting wall or boundary
        if not (0 <= nr < self.H and 0 <= nc < self.W) or self.maze[nr, nc] == 1:
            nr, nc = r, c

        self._agent_pos = (nr, nc)
        self._step_count += 1
        state = self._pos_to_state[(nr, nc)]

        distance = np.abs(nr - self.goal[0]) + np.abs(nc - self.goal[1])

        if (nr, nc) == self.goal:
            return state, self.goal_reward, True

        if self._step_count >= self.max_steps:
            return state, self.step_reward * distance / self.start_goal_distance, True

        return state, self.step_reward * distance / self.start_goal_distance, False

    def state_to_pos(self, state: int) -> tuple:
        return self._state_to_pos[state]

    def pos_to_state(self, pos: tuple) -> int:
        return self._pos_to_state[pos]