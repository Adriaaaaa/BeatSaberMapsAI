import os
import sys

from domain.vector_metadata import VectorMetadata

from infra.map_parser import BSMapParser
from analysis.maps.profiler import BSMapProfiler

from infra.audio_loader import AudioLoader
from analysis.audio.profiler import AudioProfiler


from typing import List

from utils.logger import LoggerManager

from analysis.audio.align_utils import align_audio_features

from utils.constants import *

from analysis.audio.cluster_model import ClusterModel

# Configure the logger
log = LoggerManager.get_logger(__name__)


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
        print("4.Print all audio features\n")
        print("5.Convert first song in audio feature then vector\n")
        print("6.Convert all songs to audiofeatures and then to vector\n")
        print("7.Train Clustering Model on all songs\n")
        print("8.Train Clustering Model on all combinations of features and stats\n")
        print("q.Quit\n")
        choice = input("Enter your choice: ")

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
            audio_loader = AudioLoader(sample_rate=AUDIO_DEFAULT_SAMPLE_RATE, mono=True)
            audio_profiler = AudioProfiler(
                audio_loader.sample_rate,
                hop_length=AUDIO_DEFAULT_HOP,
                mono=audio_loader.mono,
            )

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
                    audio_profiler.extract_audio_features(
                        audio_data, map_folder=map_path
                    )
                    log.info(
                        f"Audio for map {map_folder} analyzed successfully and cached.\n"
                    )
            break
        elif choice == "4":
            log.info("Print all audio features...\n")
            audio_loader = AudioLoader(sample_rate=AUDIO_DEFAULT_SAMPLE_RATE, mono=True)
            audio_profiler = AudioProfiler(
                audio_loader.sample_rate,
                hop_length=AUDIO_DEFAULT_HOP,
                mono=audio_loader.mono,
            )

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
                    features = audio_profiler.extract_audio_features(
                        audio_data, map_folder=map_path
                    )
                    if features:
                        log.info(features.summary_str())
                    log.info(
                        f"Audio for map {map_folder} analyzed successfully and cached.\n"
                    )
            break
        elif choice == "5":
            log.info("Convert first song in audio feature then vector\n")
            audio_loader = AudioLoader(sample_rate=AUDIO_DEFAULT_SAMPLE_RATE, mono=True)
            audio_profiler = AudioProfiler(
                audio_loader.sample_rate,
                hop_length=AUDIO_DEFAULT_HOP,
                mono=audio_loader.mono,
            )

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
                    features = audio_profiler.extract_audio_features(
                        audio_data, map_folder=map_path
                    )
                    if features:
                        log.info(
                            f"Audio for map {map_folder} analyzed successfully and cached.\n"
                        )
                        align_audio_features(features)
                        log.info("Audio features aligned.\n")
                        vector = audio_profiler.convert_features_to_vector(
                            track_id=map_folder, features=features, map_folder=map_path
                        )
                        if vector:
                            log.info(
                                f"Audio for map {map_folder} converted to vector successfully and cached.\n"
                            )
        elif choice == "6":
            log.info("Convert all songs to audiofeatures and then to vector\n")
            audio_loader = AudioLoader(sample_rate=AUDIO_DEFAULT_SAMPLE_RATE, mono=True)
            audio_profiler = AudioProfiler(
                audio_loader.sample_rate,
                hop_length=AUDIO_DEFAULT_HOP,
                mono=audio_loader.mono,
            )

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
                    features = audio_profiler.extract_audio_features(
                        audio_data, map_folder=map_path
                    )
                    if features:
                        log.info(
                            f"Audio for map {map_folder} analyzed successfully and cached.\n"
                        )
                        align_audio_features(features)
                        log.info("Audio features aligned.\n")
                        log.info(
                            "Now computing vectors for combinations of features and stats.\n"
                        )
                        combinations = []
                        oned = ONED_FEATURES.copy()
                        twod = TWOD_FEATURES.copy()
                        stat = STATS_ORDER.copy()

                        combinations.append((oned.copy(), twod.copy(), stat))

                        while oned:
                            twodtopop = twod.copy()
                            while twodtopop:
                                other_2dfeature = twodtopop.pop(0)

                                combinations.append(
                                    (oned.copy(), twodtopop.copy(), STATS_ORDER)
                                )
                                combinations.append(
                                    (oned.copy(), [other_2dfeature], STATS_ORDER)
                                )
                            oned.pop(0)

                        n_combinations = len(combinations)
                        log.info(
                            f"Computing {n_combinations} combinations for map {map_folder}"
                        )

                        for oned, twod, stat in combinations:
                            vector = audio_profiler.convert_features_to_vector(
                                track_id=map_folder,
                                features=features,
                                map_folder=map_path,
                                oned_features_to_keep=oned,
                                twod_features_to_keep=twod,
                                stats_to_keep=stat,
                            )
                            if vector:
                                log.info(
                                    f"Audio for map {map_folder} converted to vector successfully and cached.\n"
                                )

        elif choice.lower() == "7":
            log.info("Training clustering model on all songs...\n")
            # Try different number of clusters from 2 to 20 and print silhouette score
            scorebycluster = {}
            features_included = {
                "1d": ONED_FEATURES.copy(),
                "2d": TWOD_FEATURES.copy(),
                "stats": STATS_ORDER.copy(),
            }
            combination_id = VectorMetadata.build_id(features_included)

            log.info(f"Combination ID: {combination_id}")
            log.info(f"Features included: {features_included}")

            score_by_cluster = []
            track_folders = os.listdir(MAPS_DIR)
            track_folders = [os.path.join(MAPS_DIR, folder) for folder in track_folders]

            for i in range(2, 10):
                cluster_model = ClusterModel(nb_clusters=i)
                score = cluster_model.train(track_folders, combination_id)
                score_by_cluster.append({"score": score, "num_clusters": i})

            for item in score_by_cluster:
                log.info(
                    f"Number of clusters: {item['num_clusters']}, Silhouette score: {item['score']}"
                )

        elif choice.lower() == "8":
            # Now that we have a bunch of track vector for each combination, let's try different clustering from them
            # First let's recompute combinations
            combinations = []
            oned = ONED_FEATURES.copy()
            twod = TWOD_FEATURES.copy()
            stat = STATS_ORDER.copy()

            combinations.append((oned.copy(), twod.copy(), stat))
            silhouette_by_combination = []

            while oned:
                twodtopop = twod.copy()
                while twodtopop:
                    other_2dfeature = twodtopop.pop(0)

                    combinations.append((oned.copy(), twodtopop.copy(), STATS_ORDER))
                    combinations.append((oned.copy(), [other_2dfeature], STATS_ORDER))
                oned.pop(0)

            n_combinations = len(combinations)

            # now let's parse maps

            for oned, twod, stat in combinations:
                log.info(f"Processing combination: {oned}, {twod}, {stat}")
                features_included = {
                    "1d": oned,
                    "2d": twod,
                    "stats": stat,
                }
                combination_id = VectorMetadata.build_id(features_included)
                # For each combination, we need to find the list of corresponding track vectors to build a matrix from

                track_folders = os.listdir(MAPS_DIR)
                # need to add MAPS_DIR at the begginiing of each folder
                track_folders = [
                    os.path.join(MAPS_DIR, folder) for folder in track_folders
                ]

                cluster_model = ClusterModel(nb_clusters=NUM_CLUSTERS)
                score = cluster_model.train(track_folders, combination_id)
                if score is None:
                    log.warning(
                        f"Training failed for combination: {oned}, {twod}, {stat}"
                    )
                    continue

                log.info(
                    f"Training completed successfully for combination: {oned}, {twod}, {stat}. Silhouette score: {score}"
                )
                silhouette_by_combination.append(
                    {
                        "score": score,
                        "features_included": features_included,
                    }
                )
            # print all scores
            for item in silhouette_by_combination:
                log.info(
                    f"Combination: {item['features_included']}, Silhouette score: {item['score']}"
                )

        elif choice.lower() == "q":
            print("Quitting the program.\n")
            sys.exit(0)
        else:
            log.error("Invalid choice.\n")


if __name__ == "__main__":
    main()
