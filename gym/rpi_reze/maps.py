MAPS = {
    "Empty 4x4 (no obstacles)": [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ],
    "With Safe Road": [
        [4, 0, 2, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 3]
    ],
    "New Map": [
        [4, 0, 1, 3],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 0, 2, 0],
    ],
}

SYMBOLS = {
    0: ".",
    1: "#",
    2: "S",
    3: "G",
    4: "A",
}

def visual(map_values):
    for row in map_values:
        print(" ".join(SYMBOLS[cell] for cell in row))
    print("\n")

def display(maps, enum=True):
    if not maps:
        return
    
    if enum:
        for i, (title, values) in enumerate(maps.items(), start=1):
            print(f"({i}) {title}")
            visual(values)
    else:
        for title, values in maps.items():
            print(f"{title}")
            visual(values)

def choose():
    print("Choose map to train Reze: \n")
    display(MAPS)
    
    prompt = int(input("\n: "))

    map_list = list(MAPS.values())
    map_titles = list(MAPS.keys())

    chosen_map = map_list[prompt - 1]
    chosen_title = map_titles[prompt - 1]

    return chosen_title, chosen_map