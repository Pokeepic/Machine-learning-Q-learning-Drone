import numpy as np
import os
import sys

# --- Configuration ---
FILE_PATH = "qtable.npy"
# Using slightly thicker arrows for better visibility
ARROWS = ['←', '↓', '→', '↑'] # 0:Left, 1:Down, 2:Right, 3:Up

# ANSI Colors for CLI
RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[91m"     # Negative / Low
GREEN = "\033[92m"   # Positive / High
YELLOW = "\033[93m"  # Neutral / Zero
CYAN = "\033[96m"    # Info

def load_q_table(path):
    if not os.path.exists(path):
        print(f"{RED}Error: File '{path}' not found.{RESET}")
        # Create a dummy one for testing if it doesn't exist
        print(f"{YELLOW}Creating dummy 4x4 qtable for testing...{RESET}")
        dummy_Q = np.random.uniform(-1, 15, (16, 4))
        np.save(path, dummy_Q)
        return dummy_Q

    try:
        Q = np.load(path)
        print(f"{CYAN}Loaded Q-table with shape: {Q.shape}{RESET}")
        return Q
    except Exception as e:
        print(f"{RED}Error loading file: {e}{RESET}")
        sys.exit(1)

def get_grid_dimensions(n_states):
    """Attempts to guess grid dimensions or asks user."""
    sqrt = int(np.sqrt(n_states))
    if sqrt * sqrt == n_states:
        return sqrt, sqrt
    
    print(f"{YELLOW}Grid is not a perfect square (Total states: {n_states}).{RESET}")
    while True:
        try:
            rows_input = input("Enter number of ROWS (or press Enter to guess): ")
            if not rows_input:
                 #Rough guess for non-square
                 rows = int(np.sqrt(n_states))
            else:
                 rows = int(rows_input)

            if rows == 0: continue
            
            cols = n_states // rows
            if rows * cols == n_states:
                return rows, cols
            else:
                print(f"{RED}Rows {rows} * Cols {cols} != {n_states}. Remainder exists.{RESET}")
        except ValueError:
            print("Invalid input.")

def color_val(val):
    """Returns string formatted with color based on value."""
    # The format {:6.2f} ensures a fixed width of 6 chars for the number itself
    if val > 0.01: # Using slight threshold for float comparison
        return f"{GREEN}{val:6.2f}{RESET}"
    elif val < -0.01:
        return f"{RED}{val:6.2f}{RESET}"
    else:
        return f"{YELLOW}{val:6.2f}{RESET}"

def view_best_policy(Q, rows, cols):
    print(f"\n{BOLD}=== VIEW 1: Best Action & Value per Cell ==={RESET}")
    print("Format: [ Action  Value ]\n")
    
    # Define cell width constant for alignment
    # Space + [ + space + content(8) + space + ] + space = 14 chars total width per cell
    CELL_WIDTH = 14 

    for r in range(rows):
        line_top = ""
        line_bot = ""
        for c in range(cols):
            state_idx = r * cols + c
            
            best_action_idx = np.argmax(Q[state_idx])
            max_val = Q[state_idx, best_action_idx]
            
            symbol = ARROWS[best_action_idx]
            val_str = color_val(max_val)
            
            # Top: 3 spaces + 1 symbol + 4 spaces = 8 visible chars
            # Note: We wrap the symbol in BOLD/RESET directly.
            line_top += f" [   {BOLD}{symbol}{RESET}    ] "
            
            # Bottom: 1 space + 6 chars (from {:6.2f}) + 1 space = 8 visible chars
            line_bot += f" [ {val_str} ] "
        
        print(line_top)
        print(line_bot)
        # Update separator line length to match new cell width
        print("-" * (CELL_WIDTH * cols)) 

def view_all_actions(Q, rows, cols):
    action_names = ["LEFT", "DOWN", "RIGHT", "UP"]
    
    print(f"\n{BOLD}=== VIEW 2: All Actions Breakdown ==={RESET}")
    
    for a_idx, name in enumerate(action_names):
        print(f"\n{BOLD}--- Action: {name} ({ARROWS[a_idx]}) ---{RESET}")
        
        for r in range(rows):
            line = ""
            for c in range(cols):
                state_idx = r * cols + c
                val = Q[state_idx, a_idx]
                # Kept the pipe separator here as it works better for single-line grids
                line += f"| {color_val(val)} "
            print(line + "|")
        print("-" * (9 * cols + 1))

def main():
    # Ensure terminal handles ANSI escape codes properly on Windows
    if os.name == 'nt':
        os.system('color')

    Q = load_q_table(FILE_PATH)
    n_states = Q.shape[0]
    
    rows, cols = get_grid_dimensions(n_states)
    print(f"{CYAN}Interpreting as {rows}x{cols} Grid ({n_states} states).{RESET}\n")

    while True:
        print("\nSelect View:")
        print("(1): Best Action Map (Policy + Value)")
        print("(2): Heatmaps for each Action separately")
        print("(q): Quit")
        
        choice = input(": ").lower().strip()
        
        if choice == '1':
            view_best_policy(Q, rows, cols)
        elif choice == '2':
            view_all_actions(Q, rows, cols)
        elif choice == 'q':
            break
        else:
            print("Invalid selection.")

if __name__ == "__main__":
    main()