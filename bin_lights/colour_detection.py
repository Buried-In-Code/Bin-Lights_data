__all__ = ["DEFAULT_TOLERANCE", "colour_area_fractions", "detect_cell_colours"]

import numpy as np

from bin_lights.colours import RGB, Colour
from bin_lights.models import DetectionMode

DEFAULT_TOLERANCE = 25


def colour_area_fractions(
    cell: np.ndarray, palette: dict[Colour, RGB], tolerance: int = DEFAULT_TOLERANCE
) -> dict[Colour, float]:
    total_pixels = cell.shape[0] * cell.shape[1]
    if total_pixels == 0:
        return dict.fromkeys(palette, 0.0)

    pixels = cell.reshape(-1, 3).astype(int)

    fractions = {}
    for colour, rgb in palette.items():
        matches = np.all(np.abs(pixels - np.array(rgb)) <= tolerance, axis=1)
        fractions[colour] = matches.sum() / total_pixels
    return fractions


def detect_cell_colours(
    cell: np.ndarray,
    palette: dict[Colour, RGB],
    mode: DetectionMode,
    presence_threshold: float,
    tolerance: int = DEFAULT_TOLERANCE,
) -> frozenset[Colour]:
    fractions = colour_area_fractions(cell=cell, palette=palette, tolerance=tolerance)
    present = {
        colour: fraction for colour, fraction in fractions.items() if fraction >= presence_threshold
    }

    if not present:
        return frozenset()

    if mode is DetectionMode.MULTI:
        return frozenset(present)

    dominant = max(present, key=present.get)  # ty:ignore[no-matching-overload]
    return frozenset({dominant})
