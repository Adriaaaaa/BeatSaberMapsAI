import json
import os

from domain.beatmap import Note, Obstacle, BSMap, Bomb
from typing import Dict, Any, List


class BSMapParser:

    def __init__(self, map_folder):
        self.map_folder = map_folder
        print(f"Parsing map folder: {self.map_folder}")

    def load_data(self):
        # Ouverture du fichier info.dat et gestion des erreurs
        info_path = os.path.join(self.map_folder, "info.dat")
        if not os.path.exists(info_path):
            print(f"info.dat not found in {self.map_folder}")
        with open(info_path, "r") as info_file:
            try:
                self.info_data = json.load(info_file)
                print("info file opened")
            except json.JSONDecodeError:
                print(f"Error decoding JSON in {info_path}")

        # Ouverture du fichier expertplus.dat et gestion des erreurs
        expert_path = os.path.join(self.map_folder, "ExpertPlusStandard.dat")
        if not os.path.exists(expert_path):
            print(f"ExpertPlusStandard.dat not found in {self.map_folder}")
        with open(expert_path, "r") as expert_file:
            try:
                self.expert_data = json.load(expert_file)
                print("expert file opened")
            except json.JSONDecodeError:
                print(f"Error decoding JSON in {info_path}")
        self.bpm = self.info_data.get("_beatsPerMinute", 120)
        self.notes = self.expert_data.get("_notes", [])
        self.obstacles = self.expert_data.get("_obstacles", [])
        # Analyse des donn�es de base
        self.duration_sec = self.info_data.get("_songLength", 0)
        self.duration_beat = max([note["_time"] for note in self.notes], default=0)

        if self.duration_sec <= 0:
            # conversion de beats en secondes pour approximer la duree si necessaire
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

    def build_map_from_file(self) -> BSMap:

        notes = []
        bombs = []

        for n in self.notes:
            if n.get("_type") in {0, 1}:
                notes.append(self.convert_note(n))
            if n.get("_type") == 2:
                print("Type is unknown, skipping")
            if n.get("_type") == 3:
                bombs.append(self.convert_bomb(n))

        obstacles = [self.convert_obstacle(o) for o in self.obstacles]

        m = BSMap(
            version=self.info_data.get("_version", "1.0"),
            name=self.info_data.get("_songName", "Unknown"),
            bpm=self.bpm,
            duration={
                "seconds": self.duration_sec,
                "minutes": self.duration_min,
                "beats": self.duration_beat,
            },
            notes=notes,
            obstacles=obstacles,
            bombs=bombs if bombs else None,
        )

        m.validate()

        return m
