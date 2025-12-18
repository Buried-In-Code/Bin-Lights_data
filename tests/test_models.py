from datetime import date

import numpy as np
from calendar_extract import Calendar, Cell


def test_calendar_equality_and_hash() -> None:
    img = np.zeros((10, 10, 3))
    a = Calendar(month=1, year=2024, calendar_image=img)
    b = Calendar(month=1, year=2024, calendar_image=img.copy())
    c = Calendar(month=2, year=2024, calendar_image=img)

    assert a == b
    assert a != c
    assert len({a, b, c}) == 2


def test_cell_equality_and_hash() -> None:
    d = date(2024, 1, 1)
    a = Cell(datestamp=d, colour="red")
    b = Cell(datestamp=d, colour="blue")
    c = Cell(datestamp=date(2024, 1, 2), colour="red")

    assert a == b
    assert a != c
    assert len({a, b, c}) == 2
