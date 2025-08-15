import os

from analysis.parser import BSMapParser
from analysis.profiler import BSMapProfiler
from generation import generator
from typing import Dict, Any, List

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
            invalid_maps: List[str] = []
            valid_maps: List[str] = []
            maps_with_warnings: List[str] = []

            for map_folder in os.listdir(MAPS_DIR):
                map_path = os.path.join(MAPS_DIR, map_folder)
                map_name = os.path.basename(map_path)
                if os.path.isdir(map_path):
                    parser = BSMapParser(map_path)
                    parser.load_data()
                    m = parser.build_map_from_file()
                    if m is None:
                        print(f"No Expert plus map found in {map_path}, skipping.\n")
                        invalid_maps.append(map_name)
                        continue
                    profiler = BSMapProfiler(m)
                    profiler.compute_stats()
                    print(f"Map {m.name} analyzed successfully.\n")

                    # print warnings for map
                    print("Warnings for this map:")
                    map_has_warnings = False
                    for warning, count in m.warnings.items():
                        if count > 0:
                            map_has_warnings = True
                            print(f"- {warning}: {count} occurrences")
                    print("\n")
                    if map_has_warnings:
                        maps_with_warnings.append(map_name)
                    else:
                        valid_maps.append(map_name)
                    # Generate map

            if valid_maps:
                print("The following maps were valid and analyzed:\n")
                for valid_map in valid_maps:
                    print(f"- {valid_map}")
            if maps_with_warnings:
                print("The following maps had warnings:\n")
                for map_with_warning in maps_with_warnings:
                    print(f"- {map_with_warning}")
            if invalid_maps:
                print("The following maps were not expert plus and skipped:\n")
                for invalid_map in invalid_maps:
                    print(f"- {invalid_map}")

            print("\nAnalysis complete.\n")
            print(
                f"There are {len(invalid_maps)} non expert plus maps, {len(valid_maps)} valid maps and {len(maps_with_warnings)} maps with warnings.\n"
            )
            break
        elif choice == "2":
            print("Visualizing a map is not implemented yet.\n")
            for map_folder in os.listdir(MAPS_DIR):
                map_path = os.path.join(MAPS_DIR, map_folder)
                if os.path.isdir(map_path):
                    parser = BSMapParser(map_path)
                    parser.load_data()
                    m = parser.build_map_from_file()
                    if m is None:
                        print(f"No valid map found in {map_path}, skipping.")
                        continue
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
