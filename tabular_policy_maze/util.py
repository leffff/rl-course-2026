from collections import deque
import numpy as np

from tabular_policy_maze.maze_env import MazeEnv

def _bfs_reachable(maze, start, goal):
    H, W = maze.shape
    visited = set()
    queue = deque([start])
    visited.add(start)
    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            return True
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and maze[nr, nc] == 0 and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append((nr, nc))
    return False

def generate_maze_env(
    size,
    obstacle_pct=0.25,
    max_steps=200,
    step_reward=-1.0,
    goal_reward=0.0,
    seed=None
):
    """
    Generate a random maze and return a MazeEnv with valid start/goal.
    Start = top-left free cell, goal = bottom-right free cell.
    Guarantees a path exists between them (via BFS check + retry).
    """
    rng = np.random.RandomState(seed)
    H, W = (size, size) if isinstance(size, int) else size

    for _ in range(1000):
        maze = (rng.rand(H, W) < obstacle_pct).astype(int)
        # force start and goal free
        maze[0, 0] = 0
        maze[H - 1, W - 1] = 0

        if _bfs_reachable(maze, (0, 0), (H - 1, W - 1)):
            return MazeEnv(maze, start=(0, 0), goal=(H - 1, W - 1),
                           max_steps=max_steps, step_reward=step_reward,
                           goal_reward=goal_reward)

    raise RuntimeError("Could not generate a solvable maze after 1000 attempts")

def build_maze_env(
    maze_env_class,
    size,
    obstacle_pct=0.25,
    max_steps=200,
    step_reward=-1.0,
    goal_reward=0.0,
    seed=None
):
    """
    Generate a random maze and return a MazeEnv with valid start/goal.
    Start = top-left free cell, goal = bottom-right free cell.
    Guarantees a path exists between them (via BFS check + retry).
    """
    rng = np.random.RandomState(seed)
    H, W = (size, size) if isinstance(size, int) else size

    for _ in range(1000):
        maze = (rng.rand(H, W) < obstacle_pct).astype(int)
        # force start and goal free
        maze[0, 0] = 0
        maze[H - 1, W - 1] = 0

        if _bfs_reachable(maze, (0, 0), (H - 1, W - 1)):
            return maze_env_class(maze, start=(0, 0), goal=(H - 1, W - 1),
                           max_steps=max_steps, step_reward=step_reward,
                           goal_reward=goal_reward)

    raise RuntimeError("Could not generate a solvable maze after 1000 attempts")