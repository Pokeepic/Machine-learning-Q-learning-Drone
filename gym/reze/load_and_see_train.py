import gymnasium as gym
import numpy as np
import pickle
import custom_grid_env
import imageio
import time

def train_with_visualization(episodes=100, render=True, delay=0.1, model_path=None, save_gif=False):
    """
    Train a Q-learning agent with real-time visualization
    
    Args:
        episodes: Number of episodes to train
        render: Whether to render the environment
        delay: Delay between steps for visualization
        model_path: Path to load existing Q-table (None to start fresh)
        save_gif: Whether to save as GIF
    """
    
    grid_layout = [
        ['S', ' ', 'R', ' '],
        [' ', 'O', ' ', ' '],
        [' ', ' ', ' ', 'O'],
        ['O', ' ', ' ', 'G']
    ]

    # Choose render mode based on whether we're saving GIF
    if save_gif:
        render_mode = 'rgb_array'
    elif render:
        render_mode = 'human'
    else:
        render_mode = None
    
    env = gym.make('custom-grid-v0', 
                   grid_layout=grid_layout,
                   render_mode=render_mode,
                   cell_size=80,
                   max_episode_steps=60)
    
    unwrapped_env = env.unwrapped
    num_states = unwrapped_env.grid_rows * unwrapped_env.grid_cols
    num_actions = env.action_space.n
    
    # Load existing Q-table or start fresh
    if model_path:
        print(f"Loading Q-table from {model_path}...")
        with open(model_path, "rb") as f:
            q = pickle.load(f)
        print(f"Q-table loaded! Shape: {q.shape}\n")
    else:
        print("Starting with fresh Q-table...\n")
        q = np.zeros((num_states, num_actions))

    alpha = 0.8
    gamma = 0.93

    epsilon = 1.0
    min_epsilon = 0.05
    decay = 0.01
    
    rng = np.random.default_rng()
    
    frames = []
    
    print(f"Starting training for {episodes} episodes...\n")
    
    for i in range(episodes):
        obs = env.reset()[0]
        state = int(obs[0] * unwrapped_env.grid_cols + obs[1])
        
        terminated = False
        truncated = False
        episode_reward = 0
        steps = 0

        # Capture first frame
        if save_gif:
            frames.append(env.render())

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

            # Capture frame
            if save_gif:
                frames.append(env.render())

            if render and render_mode == 'human':
                time.sleep(delay)

            state = new_state
            episode_reward += reward
            steps += 1

        epsilon = max(min_epsilon, epsilon * np.exp(-decay))
        
        status = "✓ SUCCESS" if episode_reward > 0 else "✗ FAILED"
        print(f"Episode {i+1}/{episodes}: {status} | Steps: {steps:2d} | Reward: {episode_reward:6.2f} | ε: {epsilon:.4f}")

    env.close()

    save_gif_loc = "gifs/training_process.gif"

    # Save GIF
    if save_gif and frames:
        print(f"\nSaving GIF with {len(frames)} frames...")
        imageio.mimsave(f'{save_gif_loc}', frames, fps=5, loop=0)
        print("GIF saved as 'gifs/training_process.gif'")
    
    print(f"\nTraining completed!")

if __name__ == "__main__":
    # Start fresh training
    # train_with_visualization(episodes=50, render=True, delay=0.05, save_gif=False)
    
    # OR load existing model and continue training
    train_with_visualization(episodes=5, render=False, delay=0.05, model_path="pkls/reze_raqib_config.pkl", save_gif=True)
    
    # OR save training as GIF
    # train_with_visualization(episodes=50, render=False, save_gif=True)