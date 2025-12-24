import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(episodes):

    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False, render_mode='human')
    q = np.zeros((env.observation_space.n, env.action_space.n))

    learning_rate = 0.9 # alpha
    discount_factor = 0.9 # gamma

    epsilon = 1
    epsilon_decay_rate = 0.0001 # 1/0.0001 = 10000
    rng = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):

        state = env.reset()[0] # set every state to 0
        terminated = False # Switch to True if robot hit obstacle
        truncated = False # If the robot took more than certain actions

        while (not terminated and not truncated):
            if rng.random() < epsilon:
                action = env.action_space.sample() # actions: {0:left, 1:down, 2:right, 3:up}
            else:
                action = np.argmax(q[state,:])
            new_state, reward, terminated, truncated = env.step(action)
            
            # Update Q-values
            q[state, action] = q[state.action] + learning_rate * (
                reward + discount_factor + np.max(q[new_state])
            )
  
            # Update current state
            state = new_state

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        if (epsilon == 0):
            learning_rate = 0.0001

        if reward == 1:
            rewards_per_episode[i] = 1

    env.close()

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.savefig('frozen_lake8x8.png')
    
    f = open("frozen_lake_8x8.pkl", "wb")
    pickle.dump(q, f)
    f.close()
if __name__ == "__main__":
    run(15000)