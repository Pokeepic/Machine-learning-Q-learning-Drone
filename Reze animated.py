import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import animation

# -----------------------------
# 1) GRID (matches your screenshot)
# 0=Empty, 1=Obstacle(O), 2=Safe(R), 3=Goal(G), 4=Start(S)
# -----------------------------
grid = np.array([
    [4, 0, 2, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [1, 0, 0, 3]
])

n_rows, n_cols = grid.shape
n_states = n_rows * n_cols
n_actions = 4  # 0=Left, 1=Down, 2=Right, 3=Up

def to_state(r, c):
    return r * n_cols + c

def find_cell(val):
    return tuple(np.argwhere(grid == val)[0])

start_pos = find_cell(4)

# -----------------------------
# 2) REWARDS (delivery drone style)
# -----------------------------
STEP_PENALTY = -1
SAFE_REWARD = +2
OBSTACLE_PENALTY = -5
GOAL_REWARD = +50

def step_env(r, c, action):
    """Takes an action and returns (new_r, new_c, reward, done)."""
    nr, nc = r, c
    if action == 0:      # Left
        nc -= 1
    elif action == 1:    # Down
        nr += 1
    elif action == 2:    # Right
        nc += 1
    elif action == 3:    # Up
        nr -= 1

    # Wall -> stay with penalty
    if nr < 0 or nr >= n_rows or nc < 0 or nc >= n_cols:
        return r, c, STEP_PENALTY, False

    cell = grid[nr, nc]

    # Obstacle -> terminal
    if cell == 1:
        return nr, nc, OBSTACLE_PENALTY, True

    # Safe zone
    if cell == 2:
        return nr, nc, SAFE_REWARD, False

    # Goal -> terminal
    if cell == 3:
        return nr, nc, GOAL_REWARD, True

    # Empty
    return nr, nc, STEP_PENALTY, False


# -----------------------------
# 3) Q-LEARNING TRAINING
# -----------------------------
Q = np.zeros((n_states, n_actions))

episodes = 400
max_steps = 60
alpha = 0.8
gamma = 0.93

epsilon = 1.0
min_epsilon = 0.05
decay = 0.01

paths = []           # store path each episode for animation
episode_rewards = [] # store total reward each episode

for ep in range(episodes):
    r, c = start_pos
    done = False
    path = [(r, c)]
    total_r = 0

    for _ in range(max_steps):
        s = to_state(r, c)

        # epsilon-greedy action
        if random.random() > epsilon:
            a = int(np.argmax(Q[s]))
        else:
            a = random.randint(0, n_actions - 1)

        nr, nc, reward, done = step_env(r, c, a)
        ns = to_state(nr, nc)

        # Q update
        Q[s, a] += alpha * (reward + gamma * np.max(Q[ns]) - Q[s, a])

        r, c = nr, nc
        total_r += reward
        path.append((r, c))

        if done:
            break

    # decay epsilon after each episode
    epsilon = max(min_epsilon, epsilon * np.exp(-decay))

    paths.append(path)
    episode_rewards.append(total_r)

print("Learned Q-table:")
print(Q)

# -----------------------------
# 4) PRINT POLICY ARROWS (optional)
# -----------------------------
arrows = {0: "←", 1: "↓", 2: "→", 3: "↑"}
policy = np.array([np.argmax(Q[s]) for s in range(n_states)]).reshape(n_rows, n_cols)

print("\nGreedy Policy (S/R/O/G are tiles, arrows are actions):")
for i in range(n_rows):
    row = []
    for j in range(n_cols):
        if grid[i, j] == 4: row.append("S")
        elif grid[i, j] == 2: row.append("R")
        elif grid[i, j] == 1: row.append("O")
        elif grid[i, j] == 3: row.append("G")
        else: row.append(arrows[int(policy[i, j])])
    print(" ".join(row))

# -----------------------------
# 5) ANIMATION (agent improves over episodes)
# -----------------------------
fig, ax = plt.subplots(figsize=(5, 5))

# Light background colors to make tiles easier to see
bg = np.zeros_like(grid, dtype=float)
bg[grid == 1] = 1  # obstacles
bg[grid == 2] = 2  # safe zone
bg[grid == 3] = 3  # goal
bg[grid == 4] = 4  # start
ax.imshow(bg, alpha=0.25)

# Draw grid lines
ax.set_xticks(np.arange(n_cols + 1) - 0.5)
ax.set_yticks(np.arange(n_rows + 1) - 0.5)
ax.grid(True)
ax.set_xlim(-0.5, n_cols - 0.5)
ax.set_ylim(n_rows - 0.5, -0.5)

# Draw tile letters
tile_text = {0: "", 1: "O", 2: "R", 3: "G", 4: "S"}
for i in range(n_rows):
    for j in range(n_cols):
        ax.text(j, i, tile_text[grid[i, j]], ha="center", va="center", fontsize=14)

# Animated objects
agent_dot, = ax.plot([], [], "o", markersize=14)
trail_line, = ax.plot([], [], "-", linewidth=2)
title = ax.text(0.5, 1.05, "", transform=ax.transAxes, ha="center")

def init():
    agent_dot.set_data([], [])
    trail_line.set_data([], [])
    title.set_text("")
    return agent_dot, trail_line, title

def update(frame):
    path = paths[frame]
    ys = [p[0] for p in path]
    xs = [p[1] for p in path]

    # IMPORTANT FIX: set_data needs sequences, not single numbers
    agent_dot.set_data([xs[-1]], [ys[-1]])
    trail_line.set_data(xs, ys)
    title.set_text(f"Episode {frame+1}/{episodes} | Total Reward: {episode_rewards[frame]:.0f}")
    return agent_dot, trail_line, title

ani = animation.FuncAnimation(
    fig, update, frames=episodes, init_func=init,
    interval=60, blit=False  # blit=False works more reliably in VS Code
)

plt.show()

# -----------------------------
# 6) OPTIONAL: SAVE AS GIF (for report)
# Uncomment if you want a gif output
# Requires: pip install pillow
# -----------------------------
ani.save("q_learning_drone.gif", writer="pillow", fps=10)
print("Saved: q_learning_drone.gif")
