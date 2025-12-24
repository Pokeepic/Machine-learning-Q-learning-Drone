import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import custom_grid_env

def run(episodes, render=False):
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
        
        # Print progress every 1000 episodes
        if (i + 1) % 1000 == 0:
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
    plt.savefig('reze_training.png', dpi=150)
    plt.close()
    
    # Save Q-table
    with open("reze.pkl", "wb") as f:
        pickle.dump(q, f)
    
    # Print final statistics
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETED!")
    print(f"{'='*60}")
    print(f"Total episodes: {episodes}")
    print(f"Final average reward (last 100): {sum_rewards[-1]:.2f}")
    print(f"Final success rate (last 100): {success_rate[-1]:.0f}%")
    print(f"Best average reward: {np.max(sum_rewards):.2f}")
    print(f"Q-table saved to: custom_grid.pkl")
    print(f"Training plot saved to: custom_grid_training.png")
    print(f"{'='*60}")
    
    return q

if __name__ == "__main__":
    q_table = run(15000, render=False)