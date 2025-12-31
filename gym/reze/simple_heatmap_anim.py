import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter

def save_heatmap_gif(snapshots, snapshot_episodes, grid_rows, grid_cols, episodes, save_loc):
    print(f"\nCreating heatmap animation with {len(snapshots)} frames...")
        
    fig, ax = plt.subplots(figsize=(8, 6))
    actions = ['←', '↓', '→', '↑']

    # Set up the writer
    writer = PillowWriter(fps=2) # <--------------------------------------- TO CHANGE FPS

    with writer.saving(fig, save_loc, dpi=100):
        for idx, (q_snap, ep) in enumerate(zip(snapshots, snapshot_episodes)):
            ax.clear()
            
            # Get max Q-value for each state
            max_q = np.max(q_snap, axis=1)
            q_grid = max_q.reshape(grid_rows, grid_cols)
            
            # Plot heatmap
            im = ax.imshow(q_grid, cmap='hot', interpolation='nearest')
            ax.set_title(f'Q-table Heatmap - Episode {ep}/{episodes}', fontweight='bold')
            ax.set_xlabel('Column')
            ax.set_ylabel('Row')
            
            # Add text annotations with adaptive color
            for i in range(grid_rows):
                for j in range(grid_cols):
                    state = i * grid_cols + j
                    best_action = np.argmax(q_snap[state])
                    
                    # Use black text for bright cells, white for dark cells
                    text_color = 'black' if q_grid[i, j] > q_grid.max() * 0.5 else 'white'
                    
                    ax.text(j, i, f'{q_grid[i,j]:.1f}\n{actions[best_action]}', 
                            ha='center', va='center', color=text_color, fontweight='bold', fontsize=20)
            
            # Add colorbar (remove old one if exists)
            if idx == 0:
                cbar = fig.colorbar(im, ax=ax, label='Max Q-value')
            else:
                cbar.update_normal(im)
            
            plt.tight_layout()
            writer.grab_frame()
            
            if (idx + 1) % 10 == 0 or idx == len(snapshots) - 1:
                print(f"  Processed frame {idx + 1}/{len(snapshots)}")

    plt.close(fig)
    print(f"Heatmap animation saved to: {save_loc}")