from datetime import date

import numpy as np

from bin_lights.colours import Colour
from bin_lights.models import Calendar, Cell


def test_calendar_equality_ignores_image() -> None:
    first = Calendar(month=7, year=2025, image=np.zeros((2, 2, 3)))
    second = Calendar(month=7, year=2025, image=np.ones((5, 5, 3)))
    assert first == second
    assert hash(first) == hash(second)


def test_calendar_inequality() -> None:
    first = Calendar(month=7, year=2025, image=np.zeros((2, 2, 3)))
    second = Calendar(month=8, year=2025, image=np.zeros((2, 2, 3)))
    assert first != second


def test_calendar_equality_against_other_type() -> None:
    calendar = Calendar(month=7, year=2025, image=np.zeros((2, 2, 3)))
    assert calendar.__eq__(object()) is NotImplemented


def test_cell_equality_ignores_colours() -> None:
    first = Cell(datestamp=date(2025, 7, 1), colours=frozenset({Colour.RED}))
    second = Cell(datestamp=date(2025, 7, 1), colours=frozenset({Colour.BLUE}), is_offset=True)
    assert first == second
    assert hash(first) == hash(second)


def test_cell_inequality() -> None:
    first = Cell(datestamp=date(2025, 7, 1))
    second = Cell(datestamp=date(2025, 7, 2))
    assert first != second


def test_cell_equality_against_other_type() -> None:
    cell = Cell(datestamp=date(2025, 7, 1))
    assert cell.__eq__(object()) is NotImplemented


def test_cell_defaults() -> None:
    cell = Cell(datestamp=date(2025, 7, 1))
    assert cell.colours == frozenset()
    assert cell.is_offset is False


def test_cells_dedupe_in_a_set_by_datestamp() -> None:
    cells = {
        Cell(datestamp=date(2025, 7, 1), colours=frozenset({Colour.RED})),
        Cell(datestamp=date(2025, 7, 1), colours=frozenset({Colour.BLUE})),
        Cell(datestamp=date(2025, 7, 2)),
    }
    assert len(cells) == 2
