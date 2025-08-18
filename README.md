# BeatSaberMapsAI

BeatSaberMapsAI is a tool for analyzing, validating, and visualizing Beat Saber custom maps. It provides statistics, warnings, and insights about map structure and audio, helping mappers improve the quality of their creations.

## Features

- **Map Analysis:** Parses Beat Saber map files (`info.dat`, `ExpertPlusStandard.dat`, `ExpertPlus.dat`) and analyzes notes, obstacles, and bombs.
- **Validation:** Checks for common mapping errors and provides warnings (e.g., invalid note positions, obstacle widths, etc.).
- **Statistics:** Computes and displays statistics for each map.
- **Visualization:** (Planned) Visualizes map structure and patterns.
- **Audio Analysis:** Loads and profiles audio files associated with maps.
- **Logging:** Logs analysis results and warnings to files in the `logs/` directory.

## Project Structure
- BeatSaberMapsAI.py # Main entry point 
- src/ domain/ # Core data structures (Note, Obstacle, Bomb, BSMap) 
- infra/ 

# Infrastructure (parsing, logging, audio loading)
- analysis/ # Map and audio analysis logic 
- generation/ # Map generation utilities 
- sanity_checks.py # Utility and sanity check
- scripts maps/ # Place your Beat Saber maps here
- logs/ # Analysis logs tests/ # Unit tests Utils/


# Usage

* Prepare your maps:
 * Place each Beat Saber map in its own folder inside the maps directory.
 * Each map folder should contain at least:
info.dat
ExpertPlusStandard.dat or ExpertPlus.dat
 * The associated audio file
* Run the main program:
* Follow the menu prompts to analyze or visualize maps.

# Example Output
- Lists valid maps, maps with warnings, and skipped maps.
- Displays detailed warnings for each map.
- Logs results to the logs directory.

# Development
- Source code is organized by domain (domain), infrastructure (infra), and analysis (analysis).
- Unit tests are in the tests directory.
- Logging is handled via LoggerManager.

