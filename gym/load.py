import gymnasium as gym
import numpy as np
import pickle

def load_and_test_model(model_path="frozen_lake_8x8.pkl", episodes=10, render=True):
    """
    Load a trained Q-table and test the agent
    
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
    
    # Create environment
    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False, 
                   render_mode='human' if render else None)
    
    rewards_per_episode = []
    steps_per_episode = []
    
    print(f"\nRunning {episodes} episodes...\n")
    
    for i in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False
        steps = 0
        
        while (not terminated and not truncated):
            # Use greedy policy (always choose best action)
            action = np.argmax(q[state, :])
            new_state, reward, terminated, truncated, info = env.step(action)
            
            state = new_state
            steps += 1
        
        rewards_per_episode.append(reward)
        steps_per_episode.append(steps)
        
        status = "SUCCESS" if reward == 1 else "FAILED"
        print(f"Episode {i+1}: {status} - Steps taken: {steps}")
    
    env.close()
    
    # Calculate and print statistics
    rewards_array = np.array(rewards_per_episode)
    steps_array = np.array(steps_per_episode)
    
    success_rate = np.sum(rewards_array) / episodes * 100
    successful_steps = steps_array[rewards_array == 1]
    avg_steps = np.mean(successful_steps) if len(successful_steps) > 0 else 0
    
    print(f"\n{'='*50}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*50}")
    print(f"Total Episodes: {episodes}")
    print(f"Successful Episodes: {int(np.sum(rewards_array))}")
    print(f"Failed Episodes: {int(episodes - np.sum(rewards_array))}")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Average Steps (successful): {avg_steps:.2f}")
    print(f"{'='*50}")
    
    return q, rewards_per_episode, steps_per_episode

if __name__ == "__main__":
    # Test the trained model
    load_and_test_model("frozen_lake_8x8.pkl", episodes=1, render=True)