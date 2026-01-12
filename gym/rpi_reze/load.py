import numpy as np
import serial
import time
import maps  # Ensure maps.py is in the same folder

# --- CONFIGURATION ---
SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 9600
Q_FILE = "qtable.npy"

# Directions: 0:Left, 1:Down, 2:Right, 3:Up
# Robot starts facing UP (3)
START_FACING = 3 

def get_best_path(grid, Q):
    n_rows, n_cols = grid.shape
    r, c = tuple(np.argwhere(grid == 4)[0]) # Start pos
    
    path_actions = []
    steps = 0
    max_steps = n_rows * n_cols * 2

    while steps < max_steps:
        if grid[r, c] == 3: # Goal
            break
            
        state_idx = r * n_cols + c
        action = int(np.argmax(Q[state_idx]))
        path_actions.append(action)

        # Update coords for next loop
        if action == 0: c -= 1
        elif action == 1: r += 1
        elif action == 2: c += 1
        elif action == 3: r -= 1
        
        steps += 1
    return path_actions

def commands_from_path(actions):
    """Converts grid moves (Up/Down...) to Robot moves (Forward/Turn)."""
    cmds = []
    current_facing = START_FACING 

    for target_dir in actions:
        # Calculate turn needed: (Target - Current) % 4
        # 0: Same, 1: Left, 2: Back(U-turn), 3: Right
        diff = (target_dir - current_facing) % 4
        
        if diff == 1:
            cmds.append('L') # Turn Left
        elif diff == 3:
            cmds.append('R') # Turn Right
        elif diff == 2:
            cmds.append('R') # U-Turn (Right + Right)
            cmds.append('R')

        cmds.append('F') # Always move forward after turning
        current_facing = target_dir
        
    return cmds

def send_to_arduino(commands):
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
        time.sleep(2) # Wait for Arduino reset
        print(f"Connected to {SERIAL_PORT}")
    except:
        print(f"Failed to connect to {SERIAL_PORT}")
        return

    print(f"Sending {len(commands)} commands...")

    for i, cmd in enumerate(commands):
        print(f"Step {i+1}: Sending '{cmd}'...", end="")
        ser.write(cmd.encode()) # Send char
        
        # Wait for "Done" signal ('K') from Arduino
        while True:
            if ser.in_waiting > 0:
                response = ser.read().decode()
                if response == 'K':
                    print(" Done.")
                    break
        time.sleep(0.1) # Short cool-down

    print("Mission Complete.")
    ser.close()

def run_mission(map_data):
    # 1. Load the Q-Table
    try:
        Q = np.load(Q_FILE)
        print("Q-Table loaded successfully.")
    except FileNotFoundError:
        print(f"Error: {Q_FILE} not found. Train the model first (Option 2)!")
        return

    # 2. Use the map passed from main.py
    if map_data is None:
        print("Error: No map loaded. Please Build Environment (Option 1) first.")
        return
    
    grid = np.array(map_data)

    # 3. Calculate Path
    print("Calculating path...")
    path = get_best_path(grid, Q)
    robot_cmds = commands_from_path(path)
    
    # 4. Execute
    if robot_cmds:
        print(f"Path found: {robot_cmds}")
        send_to_arduino(robot_cmds)
    else:
        print("No path found or Goal unreachable.")