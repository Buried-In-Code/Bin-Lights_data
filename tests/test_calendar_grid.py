from datetime import date

import numpy as np
import pytest

from bin_lights.calendar_grid import (
    dominant_square_colour,
    process_calendar_squares,
    resolve_weekly_colours,
)
from bin_lights.models import Cell
from bin_lights.utils import Colour


def test_dominant_square_colour_picks_most_frequent_palette_colour() -> None:
    palette = {Colour.RED: (255, 0, 0), Colour.GREEN: (0, 255, 0)}

    square = np.array([[[0, 0, 255], [0, 0, 255]], [[0, 255, 0], [0, 0, 255]]], dtype=np.uint8)

    rgb = dominant_square_colour(square, palette)

    assert rgb == (0, 0, 255)


def test_dominant_square_colour_can_return_black() -> None:
    palette = {Colour.RED: (255, 0, 0)}

    square = np.zeros((2, 2, 3), dtype=np.uint8)

    rgb = dominant_square_colour(square, palette)

    assert rgb == (0, 0, 0)


def make_calendar_image(rows: int, colour_bgr: tuple[int, int, int]) -> np.ndarray:
    height = rows * 10
    width = 7 * 10
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = colour_bgr
    return img


def test_process_calendar_squares_returns_cells_for_month_only() -> None:
    colours = {Colour.RED: (255, 0, 0)}

    img = make_calendar_image(rows=5, colour_bgr=(0, 0, 255))

    cells = process_calendar_squares(img=img, year=2024, month=1, colours=colours)

    assert len(cells) == 31
    assert all(cell.datestamp.month == 1 for cell in cells)


def test_process_calendar_squares_assigns_correct_colour() -> None:
    colours = {Colour.RED: (255, 0, 0)}

    img = make_calendar_image(rows=5, colour_bgr=(0, 0, 255))

    cells = process_calendar_squares(img=img, year=2024, month=1, colours=colours)

    assert all(cell.colour == Colour.RED for cell in cells)


def test_process_calendar_squares_marks_offset_colours() -> None:
    colours = {Colour.RED: (255, 0, 0)}

    img = make_calendar_image(rows=5, colour_bgr=(0, 0, 255))

    cells = process_calendar_squares(
        img=img, year=2024, month=1, colours=colours, offset_colours={Colour.RED}
    )

    assert all(cell.is_offset is True for cell in cells)


def test_process_calendar_squares_unknown_colour_results_in_none() -> None:
    colours = {Colour.RED: (255, 0, 0)}

    img = make_calendar_image(rows=5, colour_bgr=(255, 0, 0))

    cells = process_calendar_squares(img=img, year=2024, month=1, colours=colours)

    assert all(cell.colour is None for cell in cells)


def test_resolve_weekly_colours_no_cells_is_noop() -> None:
    resolve_weekly_colours(set(), {Colour.RED})


def test_resolve_weekly_colours_propagates_single_action_colour() -> None:
    cells = {
        Cell(date(2024, 1, 1), Colour.RED),
        Cell(date(2024, 1, 2), None),
        Cell(date(2024, 1, 3), Colour.RED),
        Cell(date(2024, 1, 4), None),
    }

    resolve_weekly_colours(cells, {Colour.RED})

    assert all(cell.colour == Colour.RED for cell in cells)


def test_resolve_weekly_colours_does_not_change_when_no_action_colour() -> None:
    cells = {
        Cell(date(2024, 1, 1), Colour.GREEN),
        Cell(date(2024, 1, 2), None),
        Cell(date(2024, 1, 3), Colour.GREEN),
    }

    resolve_weekly_colours(cells, {Colour.RED})

    colours = {cell.colour for cell in cells}
    assert colours == {Colour.GREEN, None}


def test_resolve_weekly_colours_raises_on_ambiguous_action_colours() -> None:
    cells = {
        Cell(date(2024, 1, 1), Colour.RED),
        Cell(date(2024, 1, 2), Colour.BLUE),
        Cell(date(2024, 1, 3), None),
    }

    with pytest.raises(ValueError, match="ambiguous action colours"):
        resolve_weekly_colours(cells, {Colour.RED, Colour.BLUE})
