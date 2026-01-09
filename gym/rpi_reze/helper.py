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