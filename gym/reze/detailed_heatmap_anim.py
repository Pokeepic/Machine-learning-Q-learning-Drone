import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import custom_grid_env
from matplotlib.animation import PillowWriter

def run(episodes, render=False, save_gif=True, gif_interval=10):
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
                   cell_size=80,
                   max_episode_steps=60)  # Built-in truncation at 60 steps
    
    # Access unwrapped environment for custom attributes
    unwrapped_env = env.unwrapped
    num_states = unwrapped_env.grid_rows * unwrapped_env.grid_cols
    num_actions = env.action_space.n
    grid_rows = unwrapped_env.grid_rows
    grid_cols = unwrapped_env.grid_cols
    
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
    
    # Store Q-tables for animation
    q_snapshots = []
    snapshot_episodes = []

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
            q[state, action] = q[state, action] + alpha * (
                reward + gamma * np.max(q[new_state, :]) - q[state, action]
            )
  
            # Update current state
            state = new_state
            episode_reward += reward

        # Exponential decay of epsilon
        epsilon = max(min_epsilon, epsilon * np.exp(-decay))

        # Store episode reward
        rewards_per_episode[i] = episode_reward
        
        # Save Q-table snapshot for animation
        if save_gif and (i % gif_interval == 0 or i == episodes - 1):
            q_snapshots.append(q.copy())
            snapshot_episodes.append(i + 1)
        
        # Print progress every 10 episodes
        if (i + 1) % 10 == 0:
            avg_reward = np.mean(rewards_per_episode[max(0, i-99):i+1])
            print(f"Episode {i+1}/{episodes} - Avg Reward (last 100): {avg_reward:.2f} - Epsilon: {epsilon:.4f}")

    env.close()

    # Calculate rolling average of rewards
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.mean(rewards_per_episode[max(0, t-100):(t+1)])

    # Create plots
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Total reward per episode
    plt.subplot(1, 3, 1)
    plt.plot(rewards_per_episode, alpha=0.3)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Reward per Episode')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Rolling average
    plt.subplot(1, 3, 2)
    plt.plot(sum_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward (Last 100 Episodes)')
    plt.title('Training Progress (Rolling Average)')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Success rate (for goal reaching)
    plt.subplot(1, 3, 3)
    success_per_episode = (rewards_per_episode >= 10).astype(int)  # Goal gives reward of 10
    success_rate = np.zeros(episodes)
    for t in range(episodes):
        success_rate[t] = np.sum(success_per_episode[max(0, t-100):(t+1)])
    plt.plot(success_rate)
    plt.xlabel('Episode')
    plt.ylabel('Successes (Last 100 Episodes)')
    plt.title('Goal Reach Rate')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('robot_5_episode.png', dpi=150)
    plt.close()
    
    # Create animated heatmap GIF
    if save_gif and len(q_snapshots) > 0:
        print(f"\nCreating heatmap animation with {len(q_snapshots)} frames...")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        action_names = ['LEFT', 'DOWN', 'RIGHT', 'UP']
        
        # Set up the writer
        writer = PillowWriter(fps=2)
        
        with writer.saving(fig, "q_learning_heatmap.gif", dpi=100):
            for idx, (q_snap, ep) in enumerate(zip(q_snapshots, snapshot_episodes)):
                # Clear all axes
                for ax in axes.flat:
                    ax.clear()
                
                # Reshape Q-values for each action
                for action_idx, ax in enumerate(axes.flat):
                    q_values = q_snap[:, action_idx].reshape(grid_rows, grid_cols)
                    
                    im = ax.imshow(q_values, cmap='RdYlGn', aspect='auto', 
                                   vmin=np.min(q_snap), vmax=np.max(q_snap))
                    ax.set_title(f'{action_names[action_idx]}', fontsize=14, fontweight='bold')
                    ax.set_xlabel('Column')
                    ax.set_ylabel('Row')
                    
                    # Add value annotations
                    for i in range(grid_rows):
                        for j in range(grid_cols):
                            text = ax.text(j, i, f'{q_values[i, j]:.1f}',
                                         ha="center", va="center", color="black", 
                                         fontsize=10, fontweight='bold')
                
                fig.suptitle(f'Q-Value Heatmap - Episode {ep}/{episodes}', 
                           fontsize=16, fontweight='bold')
                plt.tight_layout()
                
                # Add colorbar
                fig.colorbar(im, ax=axes.ravel().tolist(), label='Q-Value', 
                           shrink=0.6, pad=0.02)
                
                writer.grab_frame()
                
                if (idx + 1) % 10 == 0 or idx == len(q_snapshots) - 1:
                    print(f"  Processed frame {idx + 1}/{len(q_snapshots)}")
        
        plt.close(fig)
        print(f"Heatmap animation saved to: q_learning_heatmap.gif")
    
    # Save Q-table
    with open("pkls/robot_any_episode.pkl", "wb") as f:
        pickle.dump(q, f)
    
    # Print final statistics
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETED!")
    print(f"{'='*60}")
    print(f"Total episodes: {episodes}")
    print(f"Final average reward (last 100): {sum_rewards[-1]:.2f}")
    print(f"Final success rate (last 100): {success_rate[-1]:.0f}%")
    print(f"Best average reward: {np.max(sum_rewards):.2f}")
    print(f"Q-table saved to: robot_no_learn.pkl")
    print(f"Training plot saved to: reze_training.png")
    print(f"{'='*60}")
    
    return q

if __name__ == "__main__":
    # save_gif: whether to create the animated GIF
    # gif_interval: capture Q-table every N episodes (lower = more frames, larger file)
    q_table = run(episodes=100, render=False, save_gif=True, gif_interval=1)