import inspect
import domain

from domain.beatmap import BSMap
from domain.geometry import (
    angle_between_rad,
    dir_to_angle_rad,
    dir_to_angle_deg,
    normalize,
)
import os, sys

print(sys.path)
print(
    "normalize :",
    normalize.__module__,
    normalize.__qualname__,
    inspect.signature(normalize),
)
print("normalize annotation:", normalize.__annotations__)
