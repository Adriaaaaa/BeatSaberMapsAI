from dataclasses import dataclass
from typing import List, Optional, Dict

from utils.logger import LoggerManager
from utils.constants import *

log = LoggerManager.get_logger(__name__)


@dataclass(frozen=True)
class Note:
    # a note in a beat saber map
    time: float
    saber: int  # identifier of the saber concerned
    col: int
    row: int
    dir: int  # direction of the cut

    def validate(self) -> None:
        if not (self.time >= 0):
            log.error("time should be positive")
            raise ValueError("time should be positive")
        if not (0 <= self.col < GRID_COLS):
            log.error("column index is off grid")
            raise ValueError("column index is off grid")
        if not (0 <= self.row < GRID_ROWS):
            log.error("row index is off grid")
            raise ValueError("row index is off grid")
        if not self.saber in (0, 1):
            log.error("saber should be 0 or 1")
            raise ValueError("saber should be 0 or 1")
        if not (0 <= self.dir <= 8):
            log.error("direction should be in [0,8]")
            raise ValueError("direction is wrong")


@dataclass(frozen=True)
class Obstacle:
    time: float  # time when it appears
    duration: float  # time it lasts
    width: int  # width in number of columns recovered
    type: int  # 0 : full wall, 1...

    def validate(self) -> None:
        if not (self.time >= 0):
            log.error("time should be positive")
            raise ValueError("time should be positive")
        if not (self.duration >= 0):
            log.error("duration should be positive")
            Warning("duration should be positive, this map may be broken")
        if self.width <= 0:
            log.error("width should be positive")
            raise ValueError("width should be positive")


@dataclass(frozen=True)
class Bomb:
    time: float  # time when it appears
    col: int
    row: int

    def validate(self) -> None:
        if not (self.time >= 0):
            log.error("time should be positive")
            raise ValueError("time should be positive")
        if not (0 <= self.col < GRID_COLS):
            log.error("column index should be in [0,4[")
            raise ValueError("column index should be in [0,4[")
        if not (0 <= self.row < GRID_ROWS):
            log.error("row index should be in [0,3[")
            raise ValueError("row should be in [0,3[")


@dataclass(frozen=True)
class BSMap:
    version: str
    duration: Optional[dict[str, float]]  # duration in seconds and beats
    notes: List[Note]
    warnings: Dict[str, int]  # to store warnings
    name: str = "no_name"  # name of the map
    bpm: int = 120  # beat per minute in this map
    obstacles: Optional[List[Obstacle]] = None
    bombs: Optional[List[Bomb]] = None

    def validate(self) -> None:

        if not (self.bpm >= 0):
            log.error("bpm should be positive")
            raise ValueError("bpm should be positive")
        if self.duration != None:
            if not isinstance(self.duration, dict):
                log.error("duration should be a dict[str->float]")
                raise TypeError("duration should be a dict[str->float]")
            for k in self.duration.keys():
                if k not in {"seconds", "minutes", "beats"}:
                    log.error("keys of duration should be seconds, minutes or beats")
                    raise ValueError(
                        "keys of duration should be seconds, minutes or beats "
                    )
            for k, value in self.duration.items():
                try:
                    fvalue = float(value)
                except (TypeError, ValueError):
                    log.error("duration values should be float")
                    raise ValueError("duration should be a float")
                if fvalue < 0:
                    log.error("duration should be positive")
                    raise ValueError("duration should be postive")
        for n in self.notes:
            n.validate()

        if self.obstacles != None:
            for o in self.obstacles:
                o.validate()
        if self.bombs != None:
            for b in self.bombs:
                b.validate()
