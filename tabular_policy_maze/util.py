from collections import deque
import numpy as np
from tqdm.auto import tqdm

from tabular_policy_maze.maze_env import MazeEnv
from tabular_policy_maze.reinforce import sample_action

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