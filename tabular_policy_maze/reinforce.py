import sys
sys.path.append(".")

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from tabular_policy_maze.maze_env import plot_maze

def softmax_policy(state, theta):
    """theta shape: (n_states, n_actions). Returns action probabilities."""
    logits = theta[state]
    exp_l = np.exp(logits - logits.max())
    return exp_l / exp_l.sum()

def sample_action(state, theta):
    probs = softmax_policy(state, theta)
    return np.random.choice(len(probs), p=probs)

def log_policy_gradient(state, action, theta):
    """d/d(theta[s,:]) of log pi(a|s). Returns (n_states, n_actions) grad."""
    grad = np.zeros_like(theta)
    probs = softmax_policy(state, theta)
    grad[state] = -probs
    grad[state, action] += 1.0  # one-hot minus probs (softmax grad)
    return grad

def train_reinforce(env, n_iter=300, n_episodes=32, alpha=0.05, gamma=1.0):
    theta = np.zeros((env.n_states, env.n_actions))
    mean_returns = []

    for _ in tqdm(range(n_iter), desc="REINFORCE Iteration"):
        grad_total = np.zeros_like(theta)
        returns = []

        for _ in range(n_episodes):
            s = env.reset()
            traj, G, discount = [], 0.0, 1.0
            done = False
            while not done:
                a = sample_action(s, theta)
                s_next, r, done = env.step(a)
                traj.append((s, a, r))
                G += discount * r
                discount *= gamma
                s = s_next
            for (s, a, r) in traj:
                grad_total += G * log_policy_gradient(s, a, theta)
            returns.append(G)

        theta += alpha * grad_total / n_episodes
        mean_returns.append(np.mean(returns))

    return theta, mean_returns

def sample_trajectory(env, theta):
    """Roll out the current policy. Returns list of (row, col) positions."""
    s = env.reset()
    positions = [env.state_to_pos(s)]
    done = False
    while not done:
        a = sample_action(s, theta)
        s, _, done = env.step(a)
        positions.append(env.state_to_pos(s))
    return positions


def plot_maze_with_trajectory(env, theta, title="Trajectory"):
    ax = plot_maze(env)
    positions = sample_trajectory(env, theta)

    rows = [p[0] for p in positions]
    cols = [p[1] for p in positions]

    # slight jitter so overlapping segments are visible
    jitter = np.random.uniform(-0.2, 0.2, size=(len(rows), 2))
    rows_j = np.array(rows, dtype=float) + jitter[:, 0]
    cols_j = np.array(cols, dtype=float) + jitter[:, 1]

    ax.plot(cols_j, rows_j, "b-", linewidth=1.5, alpha=0.6)
    ax.plot(cols_j[0], rows_j[0], "go", markersize=10, zorder=5)
    ax.plot(cols_j[-1], rows_j[-1], "ro", markersize=10, zorder=5)

    ax.set_title(f"{title}  (steps: {len(positions) - 1})")
    return ax

def plot_steps_distribution(env, theta, n_trajectories=500, figsize=(10, 8)):
    steps = []
    for _ in tqdm(range(n_trajectories), desc="Sampling trajectories"):
        s = env.reset()
        done, t = False, 0
        while not done:
            a = sample_action(s, theta)
            s, _, done = env.step(a)
            t += 1
        steps.append(t)

    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(steps, bins=range(min(steps), max(steps) + 2), edgecolor="black", alpha=0.7)
    ax.axvline(np.mean(steps), color="red", linestyle="--", label=f"mean={np.mean(steps):.1f}, std={np.std(steps):.1f}")
    ax.set_xlabel("Steps to goal")
    ax.set_ylabel("Count")
    ax.set_title(f"Steps distribution ({n_trajectories} episodes)")

    ax.legend()
    return ax