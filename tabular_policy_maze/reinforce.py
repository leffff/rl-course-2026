import sys
sys.path.append(".")

import numpy as np
from tqdm.auto import tqdm

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

        theta += alpha * grad_total / n_episodes

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
            
            returns.append(G)

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
            grad_total += log_policy_gradient(s, a, theta) * adv_n + entropy_beta * policy_entropy_gradient(s, theta)

        theta += alpha * grad_total / n_episodes

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


def sample_trajectory_deterministic(env, theta):
    """Roll out with greedy (argmax) policy for reproducible trajectories."""
    s = env.reset()
    positions = [env.state_to_pos(s)]
    done = False
    while not done:
        a = int(np.argmax(softmax_policy(s, theta)))
        s, _, done = env.step(a)
        positions.append(env.state_to_pos(s))
    reached_goal = (positions[-1] == env.goal)
    return positions, reached_goal
