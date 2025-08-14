import os

from analysis.parser import BSMapParser
from analysis.profiler import BSMapProfiler
from generation import generator

MAPS_DIR = "maps"


def main():
    print("Welcome to Beat Saber Maps AI by Adria!\n")
    print("This program will analyze your Beat Saber maps and provide statistics.\n")

    # Menu
    while True:
        print("\nWhat do you want to do ? :\n")
        print("1.Analyse all maps\n")
        print("2.Visualize a map\n")
        choice = input("Enter your choice (1/2): ").strip()

        if choice == "1":
            print("Analyzing all maps...\n")
            for map_folder in os.listdir(MAPS_DIR):
                map_path = os.path.join(MAPS_DIR, map_folder)
                if os.path.isdir(map_path):
                    parser = BSMapParser(map_path)
                    parser.load_data()
                    m = parser.build_map_from_file()
                    profiler = BSMapProfiler(m)
                    profiler.compute_stats()
                    print(f"Map {m.name} analyzed successfully.\n")

            break
        elif choice == "2":
            print("Visualizing a map is not implemented yet.\n")
            for map_folder in os.listdir(MAPS_DIR):
                map_path = os.path.join(MAPS_DIR, map_folder)
                if os.path.isdir(map_path):
                    parser = BSMapParser(map_path)
                    parser.load_data()
                    m = parser.build_map_from_file()
                    profiler = BSMapProfiler(m)
                    profiler.compute_visualization_data()
                    profiler.visualize()
                    print(f"Map {m.name} visualized successfully.\n")
                    break
            continue
        else:
            print("Invalid choice, please enter 1 or 2.\n")


if __name__ == "__main__":
    main()
