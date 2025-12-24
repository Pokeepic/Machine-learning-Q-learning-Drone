### How to Train Your Reze?

1. By default configurations you can always just run train_reze.py using your terminal like this:

```bash
py train_reze.py
```

2. If you want to train with your specific configurations, here is a list of hardcoded configs you can find in train_reze.py:

```python
    # LAYOUT
    grid_layout = [
        ['S', ' ', 'R', ' '],
        [' ', 'O', ' ', ' '],
        [' ', ' ', ' ', 'O'],
        ['O', ' ', ' ', 'G']
    ]
    # S: START (reze), R: REWARD, O: OBSTACLE (makima), G: GOAL (your house)
    # NOTE: Don't forget to change the layout on load_reze.py too if you want the same result.
    
    # Q-VALUE FORMULA
    learning_rate = 0.9  # alpha
    discount_factor = 0.9  # gamma

    # EPSILON GREEDY POLICY
    epsilon = 1
    epsilon_decay_rate = 0.0001

    # NUMBER OF EPISODES (you can find this in at the end of code)
    if __name__ == "__main__":
        q_table = run(NUMBER_OF_EPISODE, render=False)
```

3. To see the training in real time head over to at the end of code in train_reze.py, and set render to True

```python
if __name__ == "__main__":
    q_table = run(15000, render=True)
    # Changed render=True
```

4. Finally, to load reze, and see Reze delivered to your house, run.
```bash
py load_reze.py
```
>Note: Run train_reze.py first to generate reze.pkl (the model)

5. Fun fact: You can also save your gif now, yay!

### Now, you can experience Reze delivered to your house !
![INCOMING REZEEEEEEE](sprites/reze_devil_form.png)