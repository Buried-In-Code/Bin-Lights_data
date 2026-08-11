import numpy as np

from bin_lights.colour_detection import colour_area_fractions, detect_cell_colours
from bin_lights.colours import Colour
from bin_lights.models import DetectionMode

PALETTE = {Colour.RED: (255, 0, 0), Colour.BLUE: (0, 0, 255)}


def _solid(rgb: tuple[int, int, int], shape: tuple[int, int] = (10, 10)) -> np.ndarray:
    return np.full((*shape, 3), rgb, dtype=np.uint8)


def _split(
    left_rgb: tuple[int, int, int], right_rgb: tuple[int, int, int], left_width: int = 6
) -> np.ndarray:
    cell = np.zeros((10, 10, 3), dtype=np.uint8)
    cell[:, :left_width] = left_rgb
    cell[:, left_width:] = right_rgb
    return cell


def test_colour_area_fractions_solid_cell() -> None:
    fractions = colour_area_fractions(cell=_solid((255, 0, 0)), palette=PALETTE)
    assert fractions[Colour.RED] == 1.0
    assert fractions[Colour.BLUE] == 0.0


def test_colour_area_fractions_split_cell() -> None:
    fractions = colour_area_fractions(
        cell=_split((255, 0, 0), (0, 0, 255), left_width=6), palette=PALETTE
    )
    assert fractions[Colour.RED] == 0.6
    assert fractions[Colour.BLUE] == 0.4


def test_colour_area_fractions_empty_cell() -> None:
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    fractions = colour_area_fractions(cell=empty, palette=PALETTE)
    assert fractions == {Colour.RED: 0.0, Colour.BLUE: 0.0}


def test_colour_area_fractions_respects_tolerance() -> None:
    near_red = _solid((250, 5, 5))
    assert colour_area_fractions(cell=near_red, palette=PALETTE, tolerance=10)[Colour.RED] == 1.0
    assert colour_area_fractions(cell=near_red, palette=PALETTE, tolerance=2)[Colour.RED] == 0.0


def test_detect_single_mode_keeps_only_dominant() -> None:
    cell = _split((255, 0, 0), (0, 0, 255), left_width=9)
    colours = detect_cell_colours(
        cell=cell, palette=PALETTE, mode=DetectionMode.SINGLE, presence_threshold=0.3
    )
    assert colours == frozenset({Colour.RED})


def test_detect_multi_mode_keeps_every_colour_above_threshold() -> None:
    cell = _split((255, 0, 0), (0, 0, 255), left_width=6)
    colours = detect_cell_colours(
        cell=cell, palette=PALETTE, mode=DetectionMode.MULTI, presence_threshold=0.3
    )
    assert colours == frozenset({Colour.RED, Colour.BLUE})


def test_detect_multi_mode_drops_colours_below_threshold() -> None:
    cell = _split((255, 0, 0), (0, 0, 255), left_width=9)
    colours = detect_cell_colours(
        cell=cell, palette=PALETTE, mode=DetectionMode.MULTI, presence_threshold=0.3
    )
    assert colours == frozenset({Colour.RED})


def test_detect_returns_empty_when_nothing_present() -> None:
    cell = _solid((0, 255, 0))  # green - not in the palette at all
    colours = detect_cell_colours(
        cell=cell, palette=PALETTE, mode=DetectionMode.SINGLE, presence_threshold=0.1
    )
    assert colours == frozenset()
