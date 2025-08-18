import json
import os

from domain.beatmap import Note, Obstacle, BSMap, Bomb
from typing import Dict, Any, List, Optional

from dataclasses import dataclass

from infra.logger import LoggerManager

log = LoggerManager.get_logger(__name__)


class BSMapParser:

    def __init__(self, map_folder):
        self.map_folder = map_folder
        self.warnings: Dict[str, int] = {
            "load_error": 0,
            "no notes in map": 0,
            "invalid column index": 0,
            "invalid row index": 0,
            "unknown note type": 0,
            "zero width obstacle": 0,
            "negative width obstacle": 0,
            "invalid cut direction": 0,
        }
        log.info(f"Parsing map folder: {self.map_folder}")

    def load_data(self):
        # Ouverture du fichier info.dat et gestion des erreurs
        info_path = os.path.join(self.map_folder, "info.dat")
        self.info_data = load_json_utf8(info_path)

        # Ouverture du fichier expertplus.dat et gestion des erreurs
        expert_path = os.path.join(self.map_folder, "ExpertPlusStandard.dat")
        if not os.path.exists(expert_path):
            expert_path = os.path.join(self.map_folder, "ExpertPlus.dat")
        self.expert_data = load_json_utf8(expert_path)

        if not self.info_data or not self.expert_data:
            log.error(f"Failed to load data from {self.map_folder}.")
            self.warnings["load_error"] += 1
            return

        self.bpm = self.info_data.get("_beatsPerMinute", 120)
        self.version = self.info_data.get("_version", "1.0")
        self.name = self.info_data.get("_songName", "Unknown")

        self.notes = self.expert_data.get("_notes", [])
        self.obstacles = self.expert_data.get("_obstacles", [])
        # Analyse des donn�es de base
        self.duration_sec = self.info_data.get("_songLength", 0)
        self.duration_beat = max([note["_time"] for note in self.notes], default=0)

        if self.duration_sec <= 0:
            # conversion de beats en secondes pour approximer la duree si necessaire
            log.warning(
                f"Duration in seconds is zero or negative for map {self.name}, calculating from BPM."
            )
            self.duration_sec = round(self.duration_beat * 60 / self.bpm, 2)

        self.duration_min = self.duration_sec / 60

    def convert_note(self, n: Dict[str, Any]) -> Note:
        note = Note(
            time=n["_time"],
            saber=n["_type"],
            col=n["_lineIndex"],
            row=n["_lineLayer"],
            dir=n["_cutDirection"],
        )
        note.validate()
        return note

    def convert_obstacle(self, o: Dict[str, Any]) -> Obstacle:
        obstacle = Obstacle(
            time=o["_time"],
            duration=o["_duration"],
            width=o["_width"],
            type=o["_type"],
        )
        obstacle.validate()
        return obstacle

    def convert_bomb(self, b: Dict[str, Any]) -> Bomb:
        bomb = Bomb(
            time=b["_time"],
            col=b["_lineIndex"],
            row=b["_lineLayer"],
        )
        bomb.validate()
        return bomb

    def build_map_from_file(self) -> Optional[BSMap]:

        if not hasattr(self, "notes"):
            log.warning("No notes found in the map")
            self.warnings["no notes in map"] += 1
            return None

        notes = []
        bombs = []
        obstacles = []

        for n in self.notes:
            if n.get("_lineIndex") < 0 or n.get("_lineIndex") >= 4:
                log.warning("Note or bomb has invalid column, skipping")
                self.warnings["invalid column index"] += 1
                continue
            if n.get("_lineLayer") < 0 or n.get("_lineLayer") >= 3:
                log.warning("Note or bomb has invalid row, skipping")
                self.warnings["invalid row index"] += 1
                continue

            if n.get("_type") in {0, 1}:
                if n.get("_cutDirection") < 0 or n.get("_cutDirection") > 8:
                    log.warning("Note has invalid cut direction, skipping")
                    self.warnings["invalid cut direction"] += 1
                    continue
                notes.append(self.convert_note(n))
            if n.get("_type") == 2:
                log.warning("Type is unknown, skipping")
                self.warnings["unknown note type"] += 1
            if n.get("_type") == 3:
                bombs.append(self.convert_bomb(n))

        for o in self.obstacles:
            if o.get("_width") == 0:
                log.warning("Obstacle width is zero, skipping")
                self.warnings["zero width obstacle"] += 1
                continue
            if o.get("_width") < 0:
                log.warning("Obstacle width is negative, skipping")
                self.warnings["negative width obstacle"] += 1
                continue
            obstacles.append(self.convert_obstacle(o))

        m = BSMap(
            version=self.version,
            name=self.name,
            bpm=self.bpm,
            duration={
                "seconds": self.duration_sec,
                "minutes": self.duration_min,
                "beats": self.duration_beat,
            },
            notes=notes,
            obstacles=obstacles,
            bombs=bombs if bombs else None,
            warnings=self.warnings,
        )

        m.validate()

        return m


def load_json_utf8(file_path: str) -> Optional[Dict[str, Any]]:
    """Load a JSON file with UTF-8 encoding."""
    if not os.path.exists(file_path):
        log.error(f"File {file_path} does not exist.")
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        log.error(f"Error decoding JSON from {file_path}: {e}")
        return None
