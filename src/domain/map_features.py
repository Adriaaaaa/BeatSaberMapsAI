from __future__ import annotations

from domain.beatmap import BSMap


class MapFeatures:

    map_folder: str  # Path to the folder containing the map files

    duration_beats: float | None  # duration in beats

    def __init__(self, map: BSMap) -> None:
        self.map_folder = map.name
        if map.duration:
            self.duration_beats = map.duration.get("beats")

    def to_dict(self) -> dict:
        return {"map_folder": self.map_folder, "duration_beats": self.duration_beats}

    @classmethod
    def from_dict(cls, data: dict) -> None:
        if not data:
            raise ValueError("Invalid data for MapFeatures")
        if "map_folder" not in data or "duration_beats" not in data:
            raise ValueError("Missing required fields in MapFeatures data")
        if not data.get("duration_beats"):
            raise ValueError("Invalid duration_beats field in MapFeatures data")
        if not isinstance(data.get("duration_beats"), (int, float)):
            raise ValueError("Invalid duration_beats field in MapFeatures data")
