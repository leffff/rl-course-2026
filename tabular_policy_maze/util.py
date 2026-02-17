from collections import deque
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import imageio

from tabular_policy_maze.maze_env import MazeEnv, plot_maze
from tabular_policy_maze.reinforce import (
    sample_action,
    sample_trajectory,
    sample_trajectory_deterministic,
)


def plot_maze_with_trajectory(env, theta, title="Trajectory"):
    ax = plot_maze(env)
    positions, reached_goal = sample_trajectory(env, theta)

    rows = [p[0] for p in positions]
    cols = [p[1] for p in positions]

    jitter = np.random.uniform(-0.2, 0.2, size=(len(rows), 2))
    rows_j = np.array(rows, dtype=float) + jitter[:, 0]
    cols_j = np.array(cols, dtype=float) + jitter[:, 1]

    color = "b" if reached_goal else "r"
    ax.plot(cols_j, rows_j, f"{color}-", linewidth=1.5, alpha=0.6)
    ax.plot(cols_j[0], rows_j[0], "go", markersize=10, zorder=5)
    ax.plot(cols_j[-1], rows_j[-1], "ro", markersize=10, zorder=5)

    status = "reached goal" if reached_goal else "TRUNCATED"
    ax.set_title(f"{title}  (steps: {len(positions) - 1}, {status})")
    return ax

def plot_steps_distribution(env, theta, n_trajectories=500, figsize=(10, 5)):
    steps_success, steps_fail = [], []
    for _ in tqdm(range(n_trajectories), desc="Sampling trajectories"):
        s = env.reset()
        done, t = False, 0
        while not done:
            a = sample_action(s, theta)
            s, _, done = env.step(a)
            t += 1
        if env._agent_pos == env.goal:
            steps_success.append(t)
        else:
            steps_fail.append(t)

    fig, ax = plt.subplots(figsize=figsize)

    all_steps = steps_success + steps_fail
    bins = range(min(all_steps), max(all_steps) + 2)

    if steps_success:
        ax.hist(steps_success, bins=bins, edgecolor="black", alpha=0.7,
                color="steelblue", label=f"goal ({len(steps_success)})")
    if steps_fail:
        ax.hist(steps_fail, bins=bins, edgecolor="black", alpha=0.7,
                color="salmon", label=f"truncated ({len(steps_fail)})")

    ax.axvline(np.mean(all_steps), color="red", linestyle="--",
               label=f"mean={np.mean(all_steps):.1f}")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Count")
    success_rate = len(steps_success) / n_trajectories * 100
    ax.set_title(f"Steps distribution — success rate: {success_rate:.1f}%")
    ax.legend()
    return ax


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

def evaluate_policy(env, theta, n_eval=200):
    """Run n_eval episodes, return success rate and mean steps (goal-reaching only)."""
    success, goal_steps = 0, []
    for _ in range(n_eval):
        s = env.reset()
        done, t = False, 0
        while not done:
            a = sample_action(s, theta)
            s, _, done = env.step(a)
            t += 1
        if env._agent_pos == env.goal:
            success += 1
            goal_steps.append(t)
    rate = success / n_eval
    mean_steps = np.mean(goal_steps) if goal_steps else float('inf')
    return rate, mean_steps


def benchmark(
    train_method,
    train_arguments,
    maze_class,
    maze_arguments,
    n_seeds,
    n_eval
):
    """
    For each seed: generate maze → train REINFORCE → evaluate.
    Returns dict with per-seed and aggregate results.
    """
    results = []

    for seed in tqdm(range(n_seeds), desc="Seeds"):
        env = build_maze_env(
            maze_class,
            **maze_arguments,
            seed=seed
            )
        
        theta, mean_returns = train_method(
            env, **train_arguments
        )
        rate, mean_steps = evaluate_policy(env, theta, n_eval=n_eval)
        results.append({
            'seed': seed,
            'n_states': env.n_states,
            'success_rate': rate,
            'mean_steps': mean_steps,
            'final_return': mean_returns[-1],
        })

    rates = [r['success_rate'] for r in results]
    steps = [r['mean_steps'] for r in results if r['mean_steps'] != float('inf')]

    summary = {
        'n_seeds': n_seeds,
        'avg_success_rate': np.mean(rates),
        'std_success_rate': np.std(rates),
        'avg_steps': np.mean(steps) if steps else float('inf'),
        'std_steps': np.std(steps) if steps else float('inf'),
        'per_seed': results,
    }

    summary.update(**maze_arguments)
    summary.update(**train_arguments)

    print(f"\n{'='*50}")

    size = maze_arguments['size']
    obstacle_pct = maze_arguments['obstacle_pct']
    if size:
        print(f"Maze {size}x{size}, obstacles={obstacle_pct*100:.0f}%, seeds={n_seeds}")
    print(f"Success rate: {summary['avg_success_rate']*100:.1f}% ± {summary['std_success_rate']*100:.1f}%")
    print(f"Mean steps (goal only): {summary['avg_steps']:.1f} ± {summary['std_steps']:.1f}")
    print(f"{'='*50}")

    return summary


def render_frame(env, position, path_so_far=None):
    """
    Render one frame: maze + optional path so far + agent at position.
    position: (row, col). path_so_far: list of (row, col).
    Returns RGB array (H, W, 3) uint8.
    """
    ax = plot_maze(env)
    if path_so_far and len(path_so_far) >= 2:
        cols = [p[1] for p in path_so_far]
        rows = [p[0] for p in path_so_far]
        ax.plot(cols, rows, "b-", linewidth=2, alpha=0.6)
    r, c = position
    ax.plot(c, r, "o", color="blue", markersize=14, zorder=5)
    fig = ax.figure
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    img = imageio.imread(buf)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[-1] == 4:
        img = img[..., :3]
    return img


def frames_to_gif(frames, path, fps=4):
    """Write list of RGB arrays (H, W, 3) uint8 to GIF at path."""
    imageio.mimsave(path, frames, fps=fps, loop=0)


def create_gif(
    theta,
    env,
    output_path="maze_walk.gif",
    *,
    fps=4,
    show_path=True,
    deterministic=False,
    last_frame_hold=5,
):
    """
    Sample one trajectory, render frames, save GIF.

    Returns (positions, reached_goal).
    """

    if deterministic:
        positions, reached_goal = sample_trajectory_deterministic(env, theta)
    else:
        positions, reached_goal = sample_trajectory(env, theta)

    if not positions:
        raise RuntimeError("Trajectory is empty, cannot create GIF")

    frames = []
    for i, pos in enumerate(positions):
        path_so_far = positions[: i + 1] if show_path else None
        frames.append(render_frame(env, pos, path_so_far))

    for _ in range(last_frame_hold):
        frames.append(render_frame(env, positions[-1], positions if show_path else None))

    frames_to_gif(frames, output_path, fps=fps)
    return positions, reached_goal