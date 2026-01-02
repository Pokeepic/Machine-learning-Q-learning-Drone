import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
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
                   cell_size=80,
                   max_episode_steps=60)
    
    # Access unwrapped environment for custom attributes
    unwrapped_env = env.unwrapped
    num_states = unwrapped_env.grid_rows * unwrapped_env.grid_cols
    num_actions = env.action_space.n
    grid_rows = unwrapped_env.grid_rows
    grid_cols = unwrapped_env.grid_cols
    
    q = np.zeros((num_states, num_actions))

    # Q-Learning hyperparameters
    alpha = 0.8
    gamma = 0.93

    # Epsilon-greedy policy parameters
    epsilon = 1.0
    min_epsilon = 0.05
    decay = 0.01
    
    rng = np.random.default_rng()

    for i in range(episodes):
        obs = env.reset()[0]
        state = int(obs[0] * unwrapped_env.grid_cols + obs[1])
        
        terminated = False
        truncated = False

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

        epsilon = max(min_epsilon, epsilon * np.exp(-decay))
        
        # Save Q-table snapshots ONLY for episodes 1 and 2
        if i == 0 or i == 1:
            save_q_heatmap(q.copy(), i + 1, grid_rows, grid_cols, episodes)
            print(f"Saved Q-table heatmap for Episode {i + 1}")

    env.close()
    return q

def save_q_heatmap(q_snap, episode_num, grid_rows, grid_cols, total_episodes):
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    action_names = ['LEFT', 'DOWN', 'RIGHT', 'UP']
    
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
                ax.text(j, i, f'{q_values[i, j]:.1f}',
                       ha="center", va="center", color="black", 
                       fontsize=20, fontweight='bold')
    
    fig.suptitle(f'Q-Value Heatmap - Episode {episode_num}', 
               fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Add colorbar
    fig.colorbar(im, ax=axes.ravel().tolist(), label='Q-Value', 
               shrink=0.6, pad=0.02)
    
    plt.savefig(f'plots/q_heatmap_episode_{episode_num}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    q_table = run(episodes=100, render=False)