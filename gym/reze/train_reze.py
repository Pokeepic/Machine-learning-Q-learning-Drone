import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import custom_grid_env
from training_plots import plot_training
from simple_heatmap_anim import save_heatmap_gif

def run(episodes, render=False, heatmap_gif=False, gif_interval=10):
    # Define your grid layout
    grid_layout = [
        ['S', ' ', 'R', ' '],
        [' ', 'O', ' ', ' '],
        [' ', ' ', ' ', 'O'],
        ['O', ' ', ' ', 'G']
    ]
    
    env = gym.make('custom-grid-v0', 
                   grid_layout=grid_layout,
                   render_mode='human' if render else None,
                   cell_size=80)
    
    # Access unwrapped environment for custom attributes
    unwrapped_env = env.unwrapped
    num_states = unwrapped_env.grid_rows * unwrapped_env.grid_cols
    num_actions = env.action_space.n

    # Store Q-tables for animation
    q_snapshots = []
    snapshot_episodes = []
    
    q = np.zeros((num_states, num_actions))

    learning_rate = 0.9  # alpha
    discount_factor = 0.9  # gamma

    # EPSILON GREEDY POLICY
    epsilon = 1  # randomness
    epsilon_decay_rate = 0.0001  # 1/0.0001 = 10000
    rng = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        obs = env.reset()[0]  # Get observation [agent_row, agent_col, goal_row, goal_col]
        
        # Convert observation to state
        state = int(obs[0] * unwrapped_env.grid_cols + obs[1])
        
        terminated = False
        truncated = False
        episode_reward = 0

        while (not terminated and not truncated):
            # Epsilon-greedy action selection
            if rng.random() < epsilon:
                action = env.action_space.sample()  # actions: {0:left, 1:down, 2:right, 3:up}
            else:
                action = np.argmax(q[state, :])

            new_obs, reward, terminated, truncated, info = env.step(action)
            
            # Convert new observation to state
            new_state = int(new_obs[0] * unwrapped_env.grid_cols + new_obs[1])
            
            # Update Q-values
            q[state, action] = q[state, action] + learning_rate * (
                reward + discount_factor * np.max(q[new_state, :]) - q[state, action]
            )
  
            # Update current state
            state = new_state
            episode_reward += reward

        # Decay epsilon
        epsilon = max(epsilon - epsilon_decay_rate, 0)

        # Reduce learning rate after exploration phase
        if epsilon == 0:
            learning_rate = 0.0001

        # Store episode reward
        rewards_per_episode[i] = episode_reward

        # Save Q-table snapshot for animation
        if heatmap_gif and (i % gif_interval == 0 or i == episodes - 1):
            q_snapshots.append(q.copy())
            snapshot_episodes.append(i + 1)
        
        # Print progress every 1000 episodes
        if (i + 1) % 1000 == 0:
            avg_reward = np.mean(rewards_per_episode[max(0, i-99):i+1])
            print(f"Episode {i+1}/{episodes} - Avg Reward (last 100): {avg_reward:.2f} - Epsilon: {epsilon:.4f}")

    env.close()

    # Calculate rolling average of rewards
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.mean(rewards_per_episode[max(0, t-100):(t+1)])

    success_per_episode = (rewards_per_episode >= 10).astype(int)
    success_rate = np.zeros(episodes)
    for t in range(episodes):
        success_rate[t] = np.sum(success_per_episode[max(0, t-100):(t+1)])

    heatmap_gif_loc = "gifs/q_learning_heatmap.gif"
    plot_training_loc = "plots/training_plots.png"
    model_save_loc = "pkls/reze.pkl"

    plot_training(rewards_per_episode, sum_rewards, success_rate, plot_training_loc)

    # Create animated heatmap GIF
    if heatmap_gif and len(q_snapshots) > 0:
        save_heatmap_gif(q_snapshots, snapshot_episodes, unwrapped_env.grid_rows, unwrapped_env.grid_cols, episodes, heatmap_gif_loc)

    # Save Q-table
    with open(model_save_loc, "wb") as f:
        pickle.dump(q, f)

    print(f"TRAINING COMPLETED!")
    print(f"Total episodes: {episodes}")
    print(f"Final average reward (last 100): {sum_rewards[-1]:.2f}")
    print(f"Final success rate (last 100): {success_rate[-1]:.0f}%")
    print(f"Best average reward: {np.max(sum_rewards):.2f}")
    print(f"Q-table saved to: {model_save_loc}")
    print(f"Training plot saved to: {plot_training_loc}")
    
    return q

if __name__ == "__main__":
    q_table = run(100, render=False, heatmap_gif=True, gif_interval=1,)