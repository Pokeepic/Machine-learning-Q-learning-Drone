import gymnasium as gym
import numpy as np
import pickle
import custom_grid_env

def load_and_test_model(model_path="reze.pkl", episodes=10, render=True):
    """
    Load a trained Q-table and test the agent on the custom grid environment
    
    Args:
        model_path: Path to the saved Q-table pickle file
        episodes: Number of episodes to run
        render: Whether to render the environment
    """
    
    # Load the Q-table
    print(f"Loading model from {model_path}...")
    with open(model_path, "rb") as f:
        q = pickle.load(f)
    print(f"Model loaded! Q-table shape: {q.shape}")
    
    # Define the same grid layout used in training
    grid_layout = [
        ['S', ' ', 'R', ' '],
        [' ', 'O', ' ', ' '],
        [' ', ' ', ' ', 'O'],
        ['O', ' ', ' ', 'G']
    ]
    
    # Create environment
    env = gym.make('custom-grid-v0', 
                   grid_layout=grid_layout,
                   render_mode='human' if render else None,
                   cell_size=80)
    
    unwrapped_env = env.unwrapped
    
    rewards_per_episode = []
    steps_per_episode = []
    
    print(f"\nRunning {episodes} episodes...\n")
    
    for i in range(episodes):
        obs = env.reset()[0]
        state = int(obs[0] * unwrapped_env.grid_cols + obs[1])
        
        terminated = False
        truncated = False
        steps = 0
        episode_reward = 0
        
        while (not terminated and not truncated):
            # Use greedy policy (always choose best action)
            action = np.argmax(q[state, :])
            new_obs, reward, terminated, truncated, info = env.step(action)
            new_state = int(new_obs[0] * unwrapped_env.grid_cols + new_obs[1])
            
            state = new_state
            episode_reward += reward
            steps += 1
        
        rewards_per_episode.append(episode_reward)
        steps_per_episode.append(steps)
        
        status = "SUCCESS" if terminated and episode_reward > 0 else "FAILED"
        print(f"Episode {i+1}: {status} - Steps taken: {steps} - Total Reward: {episode_reward:.2f}")
    
    env.close()
    
    # Calculate and print statistics
    rewards_array = np.array(rewards_per_episode)
    steps_array = np.array(steps_per_episode)
    
    successful_episodes = np.sum(rewards_array > 0)
    success_rate = successful_episodes / episodes * 100
    avg_reward = np.mean(rewards_array)
    successful_steps = steps_array[rewards_array > 0]
    avg_steps = np.mean(successful_steps) if len(successful_steps) > 0 else 0
    
    print(f"\n{'='*50}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*50}")
    print(f"Total Episodes: {episodes}")
    print(f"Successful Episodes: {successful_episodes}")
    print(f"Failed Episodes: {episodes - successful_episodes}")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Steps (successful): {avg_steps:.2f}")
    print(f"{'='*50}")
    
    return q, rewards_per_episode, steps_per_episode

if __name__ == "__main__":
    # Test the trained model
    load_and_test_model("reze.pkl", episodes=10, render=True)