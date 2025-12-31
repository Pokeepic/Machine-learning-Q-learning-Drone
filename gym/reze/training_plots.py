import matplotlib.pyplot as plt
import numpy as np

def plot_training(rewards_per_episode, sum_rewards, success_rate, save_loc):
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
    plt.plot(success_rate)
    plt.xlabel('Episode')
    plt.ylabel('Successes (Last 100 Episodes)')
    plt.title('Goal Reach Rate')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # CHANGE SAVE LOCATION IF NEEDED
    plt.savefig(save_loc, dpi=150)
    plt.close()