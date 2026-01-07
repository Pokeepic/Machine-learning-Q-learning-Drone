# Essemtials
import os
import agent, build, maps, inspect_table, helper, load

CURRENT_MAP = None
CURRENT_REWARDS = None
HYPERPARAMETERS = None

if __name__ == "__main__":

    while(True):
        helper.clear_screen()
        helper.displayASCII("reze.txt")

        print("\nWelcome to Reze Interface\n")

        if CURRENT_MAP:
            print("Current map: ",end="")
            maps.display(CURRENT_MAP, enum=False)

        if CURRENT_REWARDS:
            print("Current rewards: ")
            for r, v in CURRENT_REWARDS.items():
                print(f"{r}: {v}")
            print("\n")

        print("Menu Action")
        print("(1): Build Environment")
        print("(2): Train Model")
        print("(3): Load Model into Arduino")
        print("(4): Learned Q table")
        print("(5): Manual Control")
        print("(q): Exit")

        prompt = input("\n: ").lower().strip()

        if prompt == "1":
            env_update = build.environment()
            if env_update["map"]:
                CURRENT_MAP = env_update["map"]
            if env_update["configs"]:
                CURRENT_REWARDS = env_update["configs"]

        elif prompt == "2":
            helper.clear_screen()
            helper.displayASCII("train.txt")

            if not HYPERPARAMETERS:
                HYPERPARAMETERS = agent.update_params(HYPERPARAMETERS)
            else:
                print("Current Hyperparameters: ")
                for h, v in HYPERPARAMETERS.items():
                    print(f"{h}: {v}")
                print("\n")

            while True:
                print("(1): Start training")
                print("(2): Change hyperparameters")
                print("(q): Quit")

                prompt = input("\n: ").lower().strip()
                
                if prompt == "1":
                    helper.clear_screen()
                    helper.displayASCII("train.txt")
                    agent.train(CURRENT_MAP, CURRENT_REWARDS, HYPERPARAMETERS)
                elif prompt == "2":
                    helper.clear_screen()
                    helper.displayASCII("train.txt")
                    HYPERPARAMETERS = agent.update_params(HYPERPARAMETERS)
                elif prompt == "q":
                    break
                else:
                    print("arono~")
            
        elif prompt == "3":
            helper.clear_screen()
            helper.displayASCII("load.txt")
            
            print("This is Load Section")
            buffer = input("Press any to continue...")

        elif prompt == "4":
            inspect_table.main()
            
        else:
            print("why are u like this?")
            break





