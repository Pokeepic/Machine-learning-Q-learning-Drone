import numpy as np
import random
import helper, maps

def to_state(r, c, n_cols):
    return r * n_cols + c

def find_cell(grid, val):
    return tuple(np.argwhere(grid == val)[0])

def step_env(grid, r, c, action, rewards):
    n_rows, n_cols = grid.shape
    nr, nc = r, c

    if action == 0:      # Left
        nc -= 1
    elif action == 1:    # Down
        nr += 1
    elif action == 2:    # Right
        nc += 1
    elif action == 3:    # Up
        nr -= 1

    # Wall → stay, no reward
    if nr < 0 or nr >= n_rows or nc < 0 or nc >= n_cols:
        return r, c, 0, False

    cell = grid[nr, nc]

    if cell == 1:   # Obstacle
        return nr, nc, rewards["OBSTACLE_PENALTY"], True
    if cell == 2:   # Safe
        return nr, nc, rewards["SAFE_REWARD"], False
    if cell == 3:   # Goal
        return nr, nc, rewards["GOAL_REWARD"], True

    return nr, nc, 0, False

def curr_params(parameters, original=None, display=True):
    print("Current Training Hyperparameters:")
    keys = list(parameters.keys())

    if display:
        for i, key in enumerate(keys, start=1):
            value = parameters[key]
            if original and key in original and value != original[key]:
                print(f"({i}) {key}: {value}  **")
            else:
                print(f"({i}) {key}: {value}")

    return keys


def update_params(parameters):
    updated_params = {}
    if not parameters:
        default = {
            "episodes": 400,
            "max_steps": 60,
            "alpha": 0.8,
            "gamma": 0.93,
            "epsilon": 1.0,
            "min_epsilon": 0.05,
            "decay": 0.01
        }
        parameters = default.copy()
        curr_params(parameters)
        print("Default parameters has been set.\n")
    
    while True:
        print("Hyperparameters: ")
        print("(1): Change values")
        print("(2): Save changes")

        prompt = input("\n: ").lower().strip()
        
        if prompt == "1":
            helper.clear_screen()

            if not updated_params:
                updated_params = parameters.copy()

            keys = curr_params(updated_params, original=parameters)

            print("\nUpdate")
            choice = int(input(": "))
            if 1 <= choice <= len(keys):
                key = keys[choice - 1]
                new_value = input(f"New value for {key}: ")
                updated_params[key] = type(updated_params[key])(new_value)
            else:
                print("Human error detected...")

        elif prompt == "2":
            print("Hyperparameters updated successfully.")
            return updated_params if updated_params else parameters
        
        else:
            print("Failed to update hyperparameters.")
            return parameters


def train(map, rewards, params, save_path="qtable.npy"):

    if map is None or rewards is None or params is None:
        print("Please build environment and configs first")
        return
    
    if map:
        print("Current map: ",end="")
        maps.display(map, enum=False)

    if rewards:
        print("Current rewards: ")
        for r, v in rewards.items():
            print(f"{r}: {v}")
        print("\n")
    
    if params:
        curr_params(params)

    _, grid = next(iter(map.items()))

    grid = np.array(grid)
    n_rows, n_cols = grid.shape
    n_states = n_rows * n_cols
    n_actions = 4

    Q = np.zeros((n_states, n_actions))
    start_pos = find_cell(grid, 4)

    epsilon = params["epsilon"]
    paths = []
    episode_rewards = []

    for ep in range(params["episodes"]):
        r, c = start_pos
        done = False
        path = [(r, c)]
        total_r = 0
        
        # Track safe cells visited in THIS episode
        visited_bonuses = set()

        for _ in range(params["max_steps"]):
            s = to_state(r, c, n_cols)

            if random.random() > epsilon:
                a = int(np.argmax(Q[s]))
            else:
                a = random.randint(0, n_actions - 1)

            nr, nc, reward, done = step_env(grid, r, c, a, rewards)
            
            if grid[nr, nc] == 2:
                if (nr, nc) in visited_bonuses:
                    reward = 0  # Already visited? No reward (treat as normal floor)
                else:
                    visited_bonuses.add((nr, nc)) # Mark as visited

            ns = to_state(nr, nc, n_cols)

            Q[s, a] += params["alpha"] * (
                reward + params["gamma"] * np.max(Q[ns]) - Q[s, a]
            )

            r, c = nr, nc
            total_r += reward
            path.append((r, c))

            if done:
                break

        epsilon = max(params["min_epsilon"], epsilon * np.exp(-params["decay"]))
        paths.append(path)
        episode_rewards.append(total_r)

    np.save(save_path, Q)
    print(f"Training complete. Q-table saved to {save_path}")

    return Q, paths, episode_rewards

def simulate_training(map, rewards, params, qtable_path="qtable.npy", epsilon=0.1):
    """
    Simulates one training episode using a loaded Q-table.
    Returns the path taken and robot commands.
    """
    if map is None or rewards is None or params is None:
        print("Please build environment and configs first")
        return None, None
    
    # Load Q-table
    try:
        Q = np.load(qtable_path)
        print(f"Q-table loaded from {qtable_path}")
    except FileNotFoundError:
        print(f"Q-table not found at {qtable_path}")
        return None, None
    
    _, grid = next(iter(map.items()))
    grid = np.array(grid)
    n_rows, n_cols = grid.shape
    n_actions = 4
    
    start_pos = find_cell(grid, 4)
    r, c = start_pos
    
    path = [(r, c)]
    actions = []  # Store action directions for command conversion
    total_reward = 0
    visited_bonuses = set()
    done = False
    
    print(f"\n{'='*50}")
    print(f"SIMULATING TRAINING EPISODE (epsilon={epsilon})")
    print(f"{'='*50}\n")
    print(f"Starting position: ({r}, {c})")
    
    for step in range(params["max_steps"]):
        s = to_state(r, c, n_cols)
        
        # Epsilon-greedy action selection
        if random.random() > epsilon:
            a = int(np.argmax(Q[s]))
            action_type = "exploit"
        else:
            a = random.randint(0, n_actions - 1)
            action_type = "explore"
        
        # Take action
        nr, nc, reward, done = step_env(grid, r, c, a, rewards)
        
        # Handle safe cell bonuses (same logic as training)
        if grid[nr, nc] == 2:
            if (nr, nc) in visited_bonuses:
                reward = 0
            else:
                visited_bonuses.add((nr, nc))
        
        # Print step info
        action_names = ['Up', 'Right', 'Down', 'Left']
        print(f"Step {step + 1}: ({r},{c}) -> {action_names[a]} ({action_type}) -> ({nr},{nc}) | Reward: {reward}")
        
        # Update state
        r, c = nr, nc
        total_reward += reward
        path.append((r, c))
        actions.append(a)
        
        if done:
            print(f"\n{'='*20} EPISODE COMPLETE {'='*20}")
            print(f"Total reward: {total_reward}")
            print(f"Steps taken: {len(actions)}")
            break

    buffer = input("Press any to continue...")
    
    if not done:
        print(f"\nMax steps ({params['max_steps']}) reached without completion")
    
    return actions