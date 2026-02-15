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

def policy_entropy_gradient(state, theta, eps=1e-8):
    grad = np.zeros_like(theta)
    probs = softmax_policy(state, theta)

    for a in range(len(probs)):
        weight = -probs[a] * (np.log(probs[a] + eps) + 1.0)
        grad += weight * log_policy_gradient(state, a, theta)
    
    return grad

def train_reinforce(env, n_iter=300, n_episodes=32, alpha=0.05, gamma=1.0):
    theta = np.zeros((env.n_states, env.n_actions))
    mean_returns = []

    pbar = tqdm(range(n_iter), desc="REINFORCE Iteration")
    for _ in pbar:
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
        mean_ret = float(np.mean(returns))
        mean_returns.append(mean_ret)
        pbar.set_postfix(mean_return=f"{mean_ret:.3f}")

    return theta, mean_returns


def train_reinforce_with_baseline(env, n_iter=300, n_episodes=32, alpha=0.05, gamma=1.0, alpha_v=0.01):
    theta = np.zeros((env.n_states, env.n_actions))
    V = np.zeros(env.n_states)

    mean_returns = []

    pbar = tqdm(range(n_iter), desc="REINFORCE Iteration")
    for _ in pbar:
        grad_total = np.zeros_like(theta)
        delta_V = np.zeros_like(V)
        count_V = np.zeros_like(V)
        returns = []

        for _ in range(n_episodes):
            s = env.reset()

            states = []
            actions = []
            rewards = []

            done = False
            while not done:
                a = sample_action(s, theta)
                s_next, r, done = env.step(a)
                
                states.append(s)
                actions.append(a)
                rewards.append(r)

                s = s_next
            
            T = len(states)
            G_t = np.zeros(T)
            G = 0.0
            for t in reversed(range(T)):
                G = rewards[t] + gamma * G
                G_t[t] = G
            
            returns.append(G_t[0])
            
            for t in range(T):
                s = states[t]
                a = actions[t]
                advantage = G_t[t] - V[s]
                
                grad_total += log_policy_gradient(s, a, theta) * advantage

                delta_V[s] += advantage
                count_V[s] += 1

        theta += alpha * grad_total / n_episodes

        mask = count_V > 0
        V[mask] += alpha_v * delta_V[mask] / count_V[mask]

        mean_ret = float(np.mean(returns))
        mean_returns.append(mean_ret)
        pbar.set_postfix(mean_return=f"{mean_ret:.3f}")

    return theta, mean_returns


def train_reinforce_with_advantage(
    env, n_iter=300, n_episodes=32, alpha=0.05, gamma=1.0, alpha_v=0.01, adv_eps=1e-8
):
    theta = np.zeros((env.n_states, env.n_actions), dtype=np.float64)
    V = np.zeros(env.n_states, dtype=np.float64)

    mean_returns = []

    pbar = tqdm(range(n_iter), desc="REINFORCE Iteration")
    for _ in pbar:
        returns = []
        delta_V = np.zeros_like(V)
        count_V = np.zeros_like(V)

        batch_states = []
        batch_actions = []
        batch_advantages = []

        for _ in range(n_episodes):
            s = env.reset()

            states, actions, rewards = [], [], []
            done = False
            while not done:
                a = sample_action(s, theta)
                s_next, r, done = env.step(a)

                states.append(s)
                actions.append(a)
                rewards.append(r)

                s = s_next

            T = len(states)
            G_t = np.zeros(T, dtype=np.float64)
            G = 0.0
            for t in reversed(range(T)):
                G = rewards[t] + gamma * G
                G_t[t] = G

            returns.append(G_t[0])

            for t in range(T):
                s = states[t]
                a = actions[t]
                adv = G_t[t] - V[s]

                batch_states.append(s)
                batch_actions.append(a)
                batch_advantages.append(adv)

                delta_V[s] += adv
                count_V[s] += 1.0

        adv_arr = np.asarray(batch_advantages)
        adv_mean = adv_arr.mean()
        adv_std = adv_arr.std()
        adv_norm = (adv_arr - adv_mean) / (adv_std + adv_eps)

        grad_total = np.zeros_like(theta)
        for s, a, adv_n in zip(batch_states, batch_actions, adv_norm):
            grad_total += log_policy_gradient(s, a, theta) * adv_n

        theta += alpha * grad_total / n_episodes

        mask = count_V > 0
        V[mask] += alpha_v * (delta_V[mask] / count_V[mask])

        mean_ret = float(np.mean(returns))
        mean_returns.append(mean_ret)
        pbar.set_postfix(mean_return=f"{mean_ret:.3f}")

    return theta, mean_returns


def train_reinforce_with_advantage_entropy(
    env,
    n_iter=300,
    n_episodes=32,
    alpha=0.05,
    gamma=1.0,
    alpha_v=0.01,
    entropy_beta=0.05,
    adv_eps=1e-8
):
    theta = np.zeros((env.n_states, env.n_actions), dtype=np.float64)
    V = np.zeros(env.n_states, dtype=np.float64)

    mean_returns = []

    pbar = tqdm(range(n_iter), desc="REINFORCE Iteration")
    for _ in pbar:
        returns = []
        delta_V = np.zeros_like(V)
        count_V = np.zeros_like(V)

        batch_states = []
        batch_actions = []
        batch_advantages = []

        for _ in range(n_episodes):
            s = env.reset()

            states, actions, rewards = [], [], []
            done = False
            while not done:
                a = sample_action(s, theta)
                s_next, r, done = env.step(a)

                states.append(s)
                actions.append(a)
                rewards.append(r)

                s = s_next

            T = len(states)
            G_t = np.zeros(T, dtype=np.float64)
            G = 0.0
            for t in reversed(range(T)):
                G = rewards[t] + gamma * G
                G_t[t] = G

            returns.append(G_t[0])

            for t in range(T):
                s = states[t]
                a = actions[t]
                adv = G_t[t] - V[s]

                batch_states.append(s)
                batch_actions.append(a)
                batch_advantages.append(adv)

                delta_V[s] += adv
                count_V[s] += 1.0

        adv_arr = np.asarray(batch_advantages)
        adv_norm = (adv_arr - adv_arr.mean()) / (adv_arr.std() + adv_eps)

        grad_total = np.zeros_like(theta)
        for s, a, adv_n in zip(batch_states, batch_actions, adv_norm):
            grad_total += log_policy_gradient(s, a, theta) * adv_n + entropy_beta * policy_entropy_gradient(s, theta)

        theta += alpha * grad_total / len(batch_states)

        mask = count_V > 0
        V[mask] += alpha_v * (delta_V[mask] / count_V[mask])

        mean_ret = float(np.mean(returns))
        mean_returns.append(mean_ret)
        pbar.set_postfix(mean_return=f"{mean_ret:.3f}")

    return theta, mean_returns


def train_reinforce_with_gae_entropy(
    env,
    n_iter=300,
    n_episodes=32,
    alpha=0.05,
    gamma=1.0,
    lam=0.95,
    entropy_beta=0.05,
    adv_eps=1e-8
):
    theta = np.zeros((env.n_states, env.n_actions), dtype=np.float64)
    V = np.zeros(env.n_states, dtype=np.float64)

    mean_returns = []

    pbar = tqdm(range(n_iter), desc="REINFORCE+GAE Iteration")
    for _ in pbar:
        returns = []
        batch_states = []
        batch_actions = []
        batch_advantages = []

        for _ in range(n_episodes):
            s = env.reset()

            states, actions, rewards, next_states, dones = [], [], [], [], []
            done = False
            G = 0.0
            discount = 1.0
            while not done:
                a = sample_action(s, theta)
                s_next, r, done = env.step(a)

                G += discount * r
                discount *= gamma

                states.append(s)
                actions.append(a)
                rewards.append(r)
                next_states.append(s_next)
                dones.append(done)

                s = s_next

            # --------- GAE advantages ----------
            T = len(states)
            adv_ep = np.zeros(T, dtype=np.float64)
            gae = 0.0
            for t in reversed(range(T)):
                s = states[t]
                s_next = next_states[t]

                v_next = 0.0 if dones[t] else V[s_next]

                delta = rewards[t] + gamma * v_next - V[s]
                gae = delta + gamma * lam * gae
                adv_ep[t] = gae

            for t in range(T):
                s = states[t]
                a = actions[t]
                adv = adv_ep[t]

                batch_states.append(s)
                batch_actions.append(a)
                batch_advantages.append(adv)

        adv_arr = np.asarray(batch_advantages, dtype=np.float64)
        adv_norm = (adv_arr - adv_arr.mean()) / (adv_arr.std() + adv_eps)

        grad_total = np.zeros_like(theta)
        for s, a, adv_n in zip(batch_states, batch_actions, adv_norm):
            # NOTE: assumes policy_entropy(s, theta) returns gradient of entropy wrt theta at state s.
            grad_total += log_policy_gradient(s, a, theta) * adv_n + entropy_beta * policy_entropy_gradient(s, theta)

        theta += alpha * grad_total / len(batch_states)

        mean_ret = float(np.mean(returns))
        mean_returns.append(mean_ret)
        pbar.set_postfix(mean_return=f"{mean_ret:.3f}")

    return theta, mean_returns



def sample_trajectory(env, theta):
    s = env.reset()
    positions = [env.state_to_pos(s)]
    done = False
    while not done:
        a = sample_action(s, theta)
        s, _, done = env.step(a)
        positions.append(env.state_to_pos(s))
    reached_goal = (positions[-1] == env.goal)
    return positions, reached_goal


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