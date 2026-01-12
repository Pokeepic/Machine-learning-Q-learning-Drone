import os
import platform

def clear_screen():
    if platform.system() == "Windows":
        os.system('cls')
    else:
        os.system('clear')

def displayASCII(file):
    with open(f"ascii/{file}", "r") as f:
            print(f.read())

def check_missing_params(**params):
    missing = [name for name, value in params.items() if not value]

    if missing:
        print(f"\n[!] Error: Missing {', '.join(missing)}. Please set them first.")
        input("Press any key to continue...")
        return False

    return True
