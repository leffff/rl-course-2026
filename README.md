# How big of a maze can REINFORCE solve?

This project explores hyperparameters and algorithm modifications for the REINFORCE policy-gradient method applied to shortest-path navigation in 2D mazes. The goal is to solve larger mazes, in fewer steps, with faster and more stable training.

We implement **tabular** (state–action) policies and compare:

- Vanilla REINFORCE  
- REINFORCE with value-function baseline  
- Advantage normalization  
- Entropy regularization  
- Distance-dependent rewards  

We evaluate all methods on **success rate** and **average path length** across different maze sizes. See the notebooks (`exps.ipynb`, `maze_tabular.ipynb`) for experiments and plots.

## Clear Problem Definition

The task is shortest-path navigation in a randomly generated 2D grid maze. The agent starts at the top-left corner and must reach the bottom-right corner while avoiding obstacles, in as few steps as possible.

**Environment dynamics:** The agent occupies a single cell and chooses one of four directions each step. If the chosen move leads into a wall or outside the grid, the agent stays in place. The maze layout (obstacle positions, start, goal) is fixed for the duration of an episode.

**Transitions are deterministic:** given a state and action, the next state is uniquely determined. Stochasticity arises only from the policy (action sampling) and from maze generation across different seeds.

---

## Installation

```bash
git clone <repo-url>
cd rl-course-2026/
pip install -r requirements.txt
```

**Requirements:** `numpy`, `tqdm`, `matplotlib`, `imageio`.

---

## Usage

### Train and evaluate

Build a maze environment, pick a training method, and run:

```python
from tabular_policy_maze.maze_env import MazeEnv, MazeEnvWithDistanceReward
from tabular_policy_maze.reinforce import (
    train_reinforce,
    train_reinforce_with_baseline,
    train_reinforce_with_advantage,
    train_reinforce_with_advantage_entropy,
)
from tabular_policy_maze.util import build_maze_env, benchmark

# Single run: build env, train, get policy (theta) and learning curve (mean_returns)
env = build_maze_env(MazeEnv, size=12, obstacle_pct=0.25, seed=42, max_steps=300)
theta, mean_returns = train_reinforce_with_advantage_entropy(
    env, n_iter=100, n_episodes=64, alpha=0.1, gamma=1.0, alpha_v=0.05, entropy_beta=0.01
)

# Benchmark: multiple seeds, report success rate and mean steps to goal
summary = benchmark(
    train_method=train_reinforce_with_advantage_entropy,
    train_arguments={"n_iter": 100, "n_episodes": 64, "alpha": 0.1, "gamma": 1.0, "alpha_v": 0.05, "entropy_beta": 0.01},
    maze_class=MazeEnv,
    maze_arguments={"size": 12, "obstacle_pct": 0.25, "max_steps": 300},
    n_seeds=5,
    n_eval=200,
)
```

### Visualize trajectory and save GIF

After training, you can plot a single trajectory or render the agent’s walk as a GIF:

```python
from tabular_policy_maze.util import plot_maze_with_trajectory, create_gif

# Static plot (one sampled trajectory)
plot_maze_with_trajectory(env, theta, title="Policy trajectory")
plt.show()

# Animated GIF: agent walking through the maze (deterministic = greedy path)
positions, reached_goal = create_gif(
    theta, env,
    output_path="maze_walk.gif",
    fps=4,
    show_path=True,
    deterministic=True,
    last_frame_hold=5,
)
```

---

## Environment

### Informal description

- **State space:** 2D grid of cells. Each cell is either **free (0)** or **obstacle (1)**. Only free cells are valid states; the agent cannot enter walls.
- **Actions:** 4 discrete — up, down, left, right. Invalid moves (into wall or boundary) leave the agent in place.
- **Start / goal:** Fixed at top-left and bottom-right free cells. Mazes are generated so a path between them exists (checked by BFS).
- **Rewards (default):**
  - **Step:** −1 per non-terminal step (encourages shorter paths).
  - **Goal:** 0 on reaching the goal (episode terminates).
- **Episode end:** Reaching the goal or hitting `max_steps` (truncation).
- **Transition function:** Deterministic: agent moves in the direction of the selected action, if it is not wall. If wall: stays in the same state.

### Formal definition (MDP)

We define the maze as a finite-horizon Markov decision process.

**Grid and layout.** Let the grid be $\mathcal{G} = \{0,\ldots,H-1\} \times \{0,\ldots,W-1\}$ with $H, W \in \mathbb{N}$. A **maze** is a function $M: \mathcal{G} \to \{0,1\}$ where $M(r,c)=0$ means free and $M(r,c)=1$ means obstacle. The set of **free cells** is $\mathcal{F} = \{(r,c) \in \mathcal{G} : M(r,c)=0\}$. We fix **start** $s_0 \in \mathcal{F}$ and **goal** $g \in \mathcal{F}$ (in the code: $s_0 = (0,0)$, $g = (H-1,W-1)$). Mazes are generated so that $g$ is reachable from $s_0$ via moves in $\mathcal{F}$ (checked by BFS).

**State space.** The state space is $\mathcal{S} = \mathcal{F}$, with size $|\mathcal{S}|$ equal to the number of free cells. States are identified with cell positions $(r,c)$ (or with an index in $\{0,\ldots,|\mathcal{S}|-1\}$ in the implementation).

**Action space.** $\mathcal{A} = \{0,1,2,3\}$ corresponding to **up**, **down**, **left**, **right** with displacement vectors $d_0=(-1,0)$, $d_1=(1,0)$, $d_2=(0,-1)$, $d_3=(0,1)$.

**Transition.** Transitions are deterministic. From state $s = (r,c)$ and action $a$:
- Let $(r',c') = (r,c) + d_a$. If $(r',c') \in \mathcal{G}$, $M(r',c')=0$, then the next state is $s' = (r',c')$; otherwise $s' = s$ (stay in place).

**Reward and termination (MazeEnv).** Let $T_{\max}$ be the maximum step count per episode (e.g. 200). Step reward $r_{\text{step}} \in \mathbb{R}$ (e.g. $-1$) and goal reward $r_{\text{goal}} \in \mathbb{R}$ (e.g. $0$).

- If $s' = g$: reward $r_{\text{goal}}$, episode **done**.
- Else if the number of steps in the episode has reached $T_{\max}$: reward $r_{\text{step}}$, episode **done** (truncation).
- Else: reward $r_{\text{step}}$, not done.

**Reward and termination (MazeEnvWithDistanceReward).** Same transition and termination. Reward is distance-dependent: let $d(s) = |r - r_g| + |c - c_g|$ (Manhattan distance from $s$ to $g$) and $D = d(s_0)$. For non-goal, non-truncation steps the reward is $r_{\text{step}} \cdot d(s')/D$; at truncation the reward is $r_{\text{step}} \cdot d(s')/D$; at the goal the reward is $r_{\text{goal}}$. So the agent gets a less negative (or zero) signal when closer to the goal.

**Initial state.** The initial state is always $s_0$. So we have a fixed start and an episodic task with horizon at most $T_{\max}$ and terminal events (goal or truncation).

---

Two environment variants (summary):

| Class | Description |
|-------|-------------|
| `MazeEnv` | Constant step reward (−1) and goal reward (0). |
| `MazeEnvWithDistanceReward` | Step reward scaled by Manhattan distance to goal (closer to goal → less negative). Speeds up learning by giving a denser learning signal. |

Environments are built with `build_maze_env(maze_env_class, size, obstacle_pct=0.25, max_steps=200, step_reward=-1.0, goal_reward=0.0, seed=None)`.

---

## Algorithms

All methods use a **tabular softmax policy**:

$$\pi_\theta(a|s) = \frac{\exp(\theta_{s,a})}{\sum_{a'} \exp(\theta_{s,a'})}$$

Parameters $\theta \in \mathbb{R}^{|S| \times |A|}$ are updated by policy gradient. Each iteration we collect $N$ episodes, then apply one gradient step.

**Notation:** $s_t$, $a_t$, $r_t$ = state, action, reward at step $t$. $G_t = \sum_{k \geq t} \gamma^{k-t} r_k$ = return from $t$. $\gamma$ = discount factor.

---

### 1. REINFORCE (vanilla)

**Policy gradient (no baseline):**

$$\nabla_\theta J \approx \frac{1}{N} \sum_{\text{episodes}} \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \, G_t$$

**Update:**

$$\theta \leftarrow \theta + \alpha \, \frac{1}{N} \sum_{\text{episodes}} \sum_t G_t \, \nabla_\theta \log \pi_\theta(a_t|s_t)$$

- For each episode we compute the full return $G$ from the start (or $G_t$ per step). We use the same $G$ for every $(s_t, a_t)$ in that episode (or $G_t$ for each $t$).
- No value function. High variance; learning is slow and unstable on larger mazes.

---

### 2. REINFORCE with baseline

We learn a state-value function $V(s)$ and use the **advantage** $A_t = G_t - V(s_t)$ in place of $G_t$.

**Policy gradient:**

$$\nabla_\theta J \approx \frac{1}{N} \sum_{\text{episodes}} \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \, (G_t - V(s_t))$$

**Updates:**

- **Policy:** $\theta \leftarrow \theta + \alpha \, \frac{1}{N} \sum_{\text{episodes}} \sum_t (G_t - V(s_t)) \, \nabla_\theta \log \pi_\theta(a_t|s_t)$
- **Value:** $V(s)$ is updated to match the returns observed at $s$ (e.g. running average of $G_t$ for visits to $s$, or a small step toward $G_t$).

The baseline $V(s_t)$ reduces variance without biasing the gradient ($\mathbb{E}[V(s_t) \nabla \log \pi] = 0$), improving stability and convergence.

---

### 3. REINFORCE with advantage normalization

Same as REINFORCE with baseline, but we **normalize the advantages** over the batch before the policy update.

**Computation:**

- Compute advantages $A_t = G_t - V(s_t)$ for all $(s_t, a_t)$ in the batch.
- Normalize: $\tilde{A}_t = \frac{A_t - \mu_A}{\sigma_A + \epsilon}$, where $\mu_A$, $\sigma_A$ are the mean and standard deviation of the $A_t$ in the batch, and $\epsilon > 0$ is a small constant.

**Policy gradient:**

$$\nabla_\theta J \approx \frac{1}{N} \sum \nabla_\theta \log \pi_\theta(a_t|s_t) \, \tilde{A}_t$$

**Updates:**

- **Policy:** $\theta \leftarrow \theta + \alpha \, \frac{1}{N} \sum \tilde{A}_t \, \nabla_\theta \log \pi_\theta(a_t|s_t)$
- **Value:** $V(s)$ updated as in REINFORCE with baseline (e.g. toward $G_t$ for visited states).

Normalizing stabilizes the scale of the gradient and often improves performance compared to the raw baseline.

---

### 4. REINFORCE with advantage normalization and entropy regularization

We combine **advantage normalization** (as above) with an **entropy bonus** to encourage exploration.

**Policy gradient:**

$$\nabla_\theta J \approx \frac{1}{N} \sum \Big[ \nabla_\theta \log \pi_\theta(a_t|s_t) \, \tilde{A}_t + \beta \, \nabla_\theta \mathcal{H}\big(\pi_\theta(\cdot|s_t)\big) \Big]$$

where $\mathcal{H}(\pi) = -\sum_a \pi(a) \log \pi(a)$ is the entropy of the policy at $s_t$, and $\beta > 0$ is the entropy coefficient.

**Update:**

$$\theta \leftarrow \theta + \alpha \, \frac{1}{N} \sum \Big[ \tilde{A}_t \, \nabla_\theta \log \pi_\theta(a_t|s_t) + \beta \, \nabla_\theta \mathcal{H}\big(\pi_\theta(\cdot|s_t)\big) \Big]$$

- $\tilde{A}_t$: normalized advantage as in the previous section.
- $V(s)$ is updated as before (e.g. toward observed returns at $s$).

The entropy term discourages the policy from becoming too deterministic too early, which helps avoid suboptimal policies and can improve success rate and path quality.

---

## Results

### Reproducibility

Hyperparameters for all reported experiments are stored in the `.json` files in the `results` folder.

### GIF: agent walking the maze

![Maze walk](assets/maze_walk.gif)


### Mean steps vs maze size

![Mean steps](assets/mean_steps.png)


### Success rate vs maze size

![Mean steps](assets/success_rate.png)

---

## Project structure

```
rl-course-2026/
├── README.md
├── requirements.txt
├── exps.ipynb          # Main experiments and benchmarks
├── maze_tabular.ipynb  # Additional maze/tabular experiments
└── tabular_policy_maze/
    ├── maze_env.py     # MazeEnv, MazeEnvWithDistanceReward, plot_maze
    ├── reinforce.py    # Policy + all REINFORCE training variants
    └── util.py        # build_maze_env, benchmark, plot_maze_with_trajectory, create_gif
```