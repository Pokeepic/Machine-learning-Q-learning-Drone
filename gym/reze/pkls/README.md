These are pre-trained q-models with these configs:

```python
grid_layout = [
        ['S', ' ', 'R', ' '],
        [' ', 'O', ' ', ' '],
        [' ', ' ', ' ', 'O'],
        ['O', ' ', ' ', 'G']
    ]

# Q-Learning hyperparameters
alpha = 0.8  # learning rate
gamma = 0.93  # discount factor

# Epsilon-greedy policy parameters
epsilon = 1.0
min_epsilon = 0.05
decay = 0.01

# Exponential decay of epsilon
epsilon = max(min_epsilon, epsilon * np.exp(-decay))
```

