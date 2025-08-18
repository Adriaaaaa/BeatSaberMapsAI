import os
import sys

from infra.map_parser import BSMapParser
from analysis.maps.profiler import BSMapProfiler

from infra.audio_loader import AudioLoader
from analysis.audio.profiler import AudioProfiler

from generation import generator
from typing import Dict, Any, List

from infra.logger import LoggerManager

# Configure the logger
log = LoggerManager.get_logger(__name__)
MAPS_DIR = "maps"


def main():
    log.info("Welcome to Beat Saber Maps AI by Adria!\n")
    log.info("This program will analyze your Beat Saber maps and provide statistics.\n")

    # explain first what the program does
    log.info("This program will analyze Beat Saber maps in the 'maps' directory.\n")
    log.info("It will parse the maps, compute statistics, and visualize them.\n")
    log.info(
        "Make sure you have the following directories and files in the 'maps' directory:"
    )
    log.info("- A folder for each map with the following files:")
    log.info("  - info.dat: Contains map information")
    log.info("  - ExpertPlusStandard.dat or ExpertPlus.dat: Contains map data")
    log.info(
        "The program will also analyze the audio files associated with the maps.\n"
    )

    # Menu
    while True:
        print("\nWhat do you want to do ? :\n")
        print("1.Analyse all maps\n")
        print("2.Visualize a map\n")
        print("3.Analyse all songs\n")
        choice = input("Enter your choice (1, 2 or 3): ")

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
        elif choice == "3":
            log.info("Analyzing all songs...\n")
            audio_loader = AudioLoader(sample_rate=44100, mono=True)
            audio_profiler = AudioProfiler(audio_loader.sample_rate)

            # Load all maps and their audio files
            for map_folder in os.listdir(MAPS_DIR):
                map_path = os.path.join(MAPS_DIR, map_folder)
                if os.path.isdir(map_path):
                    audio_data = audio_loader.load_audio(map_path)
                    log.info(f"Loaded audio for map {map_folder} .\n")
                    if audio_data.size == 0:
                        log.warn(
                            f"No audio data found for map {map_folder}, skipping.\n"
                        )
                        continue
                    audio_loader.describe_audio(audio_data)
                    log.info(f"Audio for map {map_folder} analyzed successfully.\n")
            break
        else:
            log.error("Invalid choice, please enter 1, 2 or 3.\n")


if __name__ == "__main__":
    main()
