from datetime import date

import pytest

from bin_lights import get_project_root
from bin_lights.colours import Colour
from bin_lights.main import south_wairarapa_config, wollondilly_shire_config
from bin_lights.pipeline import extract_source_cells
from bin_lights.sources import extract_calendars

SOURCES = get_project_root() / "sources"


@pytest.fixture(scope="module")
def swdc_cells() -> dict[date, frozenset]:
    config = south_wairarapa_config("South-Wairarapa_Apr-2025_Mar-2026.pdf")
    cells = extract_source_cells(config)
    return {cell.datestamp: cell for cell in cells}


@pytest.fixture(scope="module")
def razorback_cells() -> dict[date, frozenset]:
    config = wollondilly_shire_config("Wollondilly-Shire_Jul-2025_Jun-2026.pdf")
    cells = extract_source_cells(config)
    return {cell.datestamp: cell for cell in cells}


def test_swdc_pdf_finds_all_twelve_months() -> None:
    config = south_wairarapa_config("South-Wairarapa_Apr-2025_Mar-2026.pdf")
    calendars = extract_calendars(config)
    assert {(c.year, c.month) for c in calendars} == {
        (2025, 4), (2025, 5), (2025, 6), (2025, 7), (2025, 8), (2025, 9),
        (2025, 10), (2025, 11), (2025, 12), (2026, 1), (2026, 2), (2026, 3),
    }  # fmt: skip


def test_swdc_week_resolves_to_single_colour(swdc_cells: dict) -> None:
    # May 2025: 12-18 is a solid blue week in the source calendar.
    week = [swdc_cells[date(2025, 5, day)] for day in range(12, 19)]
    assert all(cell.colours == frozenset({Colour.BLUE}) for cell in week)


def test_swdc_holiday_marker_shifts_the_date(swdc_cells: dict) -> None:
    # 2 June 2025 is marked with the black "public holiday" circle.
    assert swdc_cells[date(2025, 6, 2)].is_offset is True


def test_razorback_finds_all_twelve_months() -> None:
    config = wollondilly_shire_config("Wollondilly-Shire_Jul-2025_Jun-2026.pdf")
    calendars = extract_calendars(config)
    years_months = {(c.year, c.month) for c in calendars}
    assert (2026, 5) in years_months  # the mislabelled block, corrected via date_fixes
    assert (2025, 5) not in years_months
    assert len(years_months) == 12


def test_razorback_festive_week_has_both_colours(razorback_cells: dict) -> None:
    for day in (29, 30, 31):
        colours = razorback_cells[date(2025, 12, day)].colours
        assert colours == frozenset({Colour.GREEN, Colour.YELLOW}), (day, colours)


def test_razorback_ordinary_week_has_one_colour(razorback_cells: dict) -> None:
    assert razorback_cells[date(2025, 7, 1)].colours == frozenset({Colour.GREEN})


def test_razorback_weekend_has_no_colour(razorback_cells: dict) -> None:
    assert razorback_cells[date(2025, 7, 5)].colours == frozenset()  # Saturday
