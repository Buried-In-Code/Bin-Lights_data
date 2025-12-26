from datetime import date
from pathlib import Path

import numpy as np
import pytest

from bin_lights.models import Calendar, Cell, Colour, Context


def test_calendar_equality_ignores_image() -> None:
    img1 = np.zeros((10, 10))
    img2 = np.ones((5, 5))

    cal1 = Calendar(year=2024, month=12, calendar_image=img1)
    cal2 = Calendar(year=2024, month=12, calendar_image=img2)

    assert cal1 == cal2


def test_calendar_inequality_by_year_or_month() -> None:
    img = np.zeros((1, 1))

    cal1 = Calendar(year=2024, month=12, calendar_image=img)
    cal2 = Calendar(year=2025, month=12, calendar_image=img)
    cal3 = Calendar(year=2024, month=11, calendar_image=img)

    assert cal1 != cal2
    assert cal1 != cal3


def test_calendar_hash_matches_equality() -> None:
    img = np.zeros((1, 1))

    cal1 = Calendar(year=2024, month=12, calendar_image=img)
    cal2 = Calendar(year=2024, month=12, calendar_image=img)

    assert hash(cal1) == hash(cal2)


def test_calendar_eq_with_non_calendar_returns_not_implemented() -> None:
    img = np.zeros((1, 1))
    cal = Calendar(year=2024, month=12, calendar_image=img)

    assert cal.__eq__("not a calendar") is NotImplemented


def test_cell_equality_ignores_colour_and_offset() -> None:
    d = date(2024, 12, 25)

    cell1 = Cell(datestamp=d, colour=Colour.RED, is_offset=False)
    cell2 = Cell(datestamp=d, colour=None, is_offset=True)

    assert cell1 == cell2


def test_cell_inequality_by_datestamp() -> None:
    cell1 = Cell(datestamp=date(2024, 12, 25), colour=Colour.RED)
    cell2 = Cell(datestamp=date(2024, 12, 26), colour=Colour.RED)

    assert cell1 != cell2


def test_cell_hash_matches_equality() -> None:
    d = date(2024, 12, 25)

    cell1 = Cell(datestamp=d, colour=Colour.RED)
    cell2 = Cell(datestamp=d, colour=Colour.GREEN)

    assert hash(cell1) == hash(cell2)


def test_cell_default_is_offset_is_false() -> None:
    cell = Cell(datestamp=date(2024, 12, 25), colour=None)

    assert cell.is_offset is False


def test_cell_eq_with_non_cell_returns_not_implemented() -> None:
    cell = Cell(datestamp=date(2024, 12, 25), colour=None)

    assert cell.__eq__("not a cell") is NotImplemented


def test_context_defaults() -> None:
    ctx = Context(file=Path("calendar.png"), colours={Colour.RED: (255, 0, 0)})

    assert ctx.rows == 4
    assert ctx.columns == 3
    assert ctx.date_fixes == {}
    assert ctx.colour_fixes == {}


def test_context_requires_keyword_arguments() -> None:
    with pytest.raises(TypeError):
        Context(Path("calendar.png"), {})  # kw_only=True


def test_context_default_dicts_are_not_shared() -> None:
    ctx1 = Context(file=Path("a.png"), colours={Colour.RED: (255, 0, 0)})
    ctx2 = Context(file=Path("b.png"), colours={Colour.BLUE: (0, 0, 255)})

    ctx1.date_fixes[date(2024, 1, 1)] = date(2024, 1, 2)

    assert ctx2.date_fixes == {}
