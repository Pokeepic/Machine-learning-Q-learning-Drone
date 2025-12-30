import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import custom_grid_env
from training_plots import plot_training
from simple_heatmap_anim import save_heatmap_gif

def run(episodes, render=False, heatmap_gif=False, gif_interval=10):
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
    
    unwrapped_env = env.unwrapped
    num_states = unwrapped_env.grid_rows * unwrapped_env.grid_cols
    num_actions = env.action_space.n

    q_snapshots = []
    snapshot_episodes = []
    
    q = np.zeros((num_states, num_actions))

    # Q-Learning hyperparameters
    alpha = 0.8  # learning rate
    gamma = 0.93  # discount factor

    # Epsilon-greedy policy parameters
    epsilon = 1.0
    min_epsilon = 0.05
    decay = 0.01

    rng = np.random.default_rng()
    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        obs = env.reset()[0]
        state = int(obs[0] * unwrapped_env.grid_cols + obs[1])
        
        terminated = False
        truncated = False
        episode_reward = 0

        while (not terminated and not truncated):
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state, :])

            new_obs, reward, terminated, truncated, info = env.step(action)
            new_state = int(new_obs[0] * unwrapped_env.grid_cols + new_obs[1])
            
            q[state, action] = q[state, action] + alpha * (
                reward + gamma * np.max(q[new_state, :]) - q[state, action]
            )
  
            state = new_state
            episode_reward += reward

        # Exponential decay of epsilon
        epsilon = max(min_epsilon, epsilon * np.exp(-decay))

        if epsilon == 0:
            alpha = 0.0001

        rewards_per_episode[i] = episode_reward

        if heatmap_gif and (i % gif_interval == 0 or i == episodes - 1):
            q_snapshots.append(q.copy())
            snapshot_episodes.append(i + 1)
        
        if (i + 1) % 1000 == 0:
            avg_reward = np.mean(rewards_per_episode[max(0, i-99):i+1])
            print(f"Episode {i+1}/{episodes} - Avg Reward (last 100): {avg_reward:.2f} - Epsilon: {epsilon:.4f}")

    env.close()

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.mean(rewards_per_episode[max(0, t-100):(t+1)])

    success_per_episode = (rewards_per_episode >= 10).astype(int)
    success_rate = np.zeros(episodes)
    for t in range(episodes):
        success_rate[t] = np.sum(success_per_episode[max(0, t-100):(t+1)])

    heatmap_gif_loc = "gifs/raqib_heatmap.gif"
    plot_training_loc = "plots/training_plots_raqib.png"
    model_save_loc = "pkls/reze_raqib_config.pkl"

    plot_training(rewards_per_episode, sum_rewards, success_rate, plot_training_loc)

    if heatmap_gif and len(q_snapshots) > 0:
        save_heatmap_gif(q_snapshots, snapshot_episodes, unwrapped_env.grid_rows, unwrapped_env.grid_cols, episodes, heatmap_gif_loc)

    # Model save
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
    q_table = run(100, render=False, heatmap_gif=True, gif_interval=1)