import maps, helper

def model():
    pass

def environment():
    title = None
    chosen_map = None
    configs = {}

    helper.clear_screen()
    helper.displayASCII("env.txt")

    print("(1): Map")
    print("(2): Rewards")

    prompt = input("\n: ").lower().strip()

    if prompt == "1":
        title, chosen_map = maps.choose()
        print(f"Building with map: {title}\n")

    elif prompt == "2":
        print(f"Rewards")

        configs = {
            "SAFE_REWARD": +2,
            "OBSTACLE_PENALTY": -1,
            "GOAL_REWARD": +10
        }
        
        for r, v in configs.items():
            print(f"{r}: {v}")


    else:
        print("why are u like this")

    buffer = input("Press any to continue...")

    return {
        "map": {title: chosen_map} if title else None,
        "configs": configs
        }



