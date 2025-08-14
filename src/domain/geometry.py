import math
from typing import Optional, Tuple

DIRECTION_TO_VECTOR = {
    0: (0, 1),  # Up
    1: (0, -1),  # Bas
    2: (-1, 0),  # Left
    3: (1, 0),  # Right
    4: (-1, 1),  # Left Up
    5: (1, 1),  # Right Up
    6: (-1, -1),  # Left Down
    7: (1, -1),  # Right Down
    8: (0, 0),  # Free
}


DIRECTION_TO_ANGLE = {
    0: 90,  # Up
    1: 270,  # Down
    2: 180,  # Left
    3: 0,  # Right
    4: 135,  # UpLeft
    5: 45,  # UpRight
    6: 225,  # DownLeft
    7: 315,  # DownRight
}


def dir_to_vector_2d(code: int) -> Optional[Tuple[float, float]]:
    if code == 8:  # Free direction
        return (0.0, 0.0)

    if code not in DIRECTION_TO_VECTOR:
        raise ValueError(f"Invalid direction code: {code}")

    return DIRECTION_TO_VECTOR[code]


def dir_to_vector_3d(code: int) -> Optional[Tuple[float, float, float]]:
    if code == 8:  # Free direction
        return (0.0, 0.0, 0.0)

    if code not in DIRECTION_TO_VECTOR:
        raise ValueError(f"Invalid direction code: {code}")

    x, y = DIRECTION_TO_VECTOR[code]
    return (x, 0.0, y)  # Assuming z is always 0 for 2D directions


def dir_to_angle_deg(code: int) -> Optional[int]:
    if code == 0:
        return None

    return DIRECTION_TO_ANGLE[int(code)]


def dir_to_angle_rad(code: int) -> Optional[float]:
    angle_deg = dir_to_angle_deg(code)

    if angle_deg == None:
        return None

    return math.radians(angle_deg)


def bsnormalize(vector: Tuple[float, float, float]) -> Tuple[float, float, float]:
    x, y, z = vector
    n = math.sqrt(x * x + y * y + z * z)

    if n == 0:
        raise ValueError("null vector")

    return (x / n, y / n, z / n)


def angle_between_rad(
    vector1: Tuple[float, float, float], vector2: Tuple[float, float, float]
) -> float:
    x1, y1, z1 = bsnormalize(vector1)
    x2, y2, z2 = bsnormalize(vector2)

    dot = max(-1.0, min(1.0, x1 * x2 + y1 * y2 + z1 * z2))

    return math.acos(dot)
