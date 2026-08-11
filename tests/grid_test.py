import calendar
from datetime import date, timedelta

import numpy as np
import pytest

from bin_lights.colours import Colour
from bin_lights.grid import build_cells, find_grid_regions, palette_mask
from bin_lights.models import DetectionMode

PALETTE = {Colour.RED: (255, 0, 0)}


def test_palette_mask_matches_within_tolerance() -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[0, 0] = (255, 0, 0)
    image[1, 1] = (250, 5, 5)  # within default tolerance
    mask = palette_mask(image=image, palette=PALETTE)
    assert mask[0, 0]
    assert mask[1, 1]
    assert not mask[2, 2]


def _blank_image(height: int, width: int) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def _fill(
    image: np.ndarray, *, y0: int, y1: int, x0: int, x1: int, rgb: tuple[int, int, int]
) -> None:
    image[y0:y1, x0:x1] = rgb


def test_find_grid_regions_locates_a_single_block() -> None:
    image = _blank_image(50, 50)
    _fill(image, y0=10, y1=40, x0=10, x1=40, rgb=(255, 0, 0))
    mask = palette_mask(image=image, palette=PALETTE)

    regions = find_grid_regions(mask=mask, rows=1, columns=1)
    assert regions == [(10, 40, 10, 40)]


def test_find_grid_regions_locates_a_grid_of_blocks() -> None:
    image = _blank_image(120, 120)
    for row in range(2):
        for col in range(2):
            y0, x0 = row * 60 + 10, col * 60 + 10
            _fill(image, y0=y0, y1=y0 + 40, x0=x0, x1=x0 + 40, rgb=(255, 0, 0))
    mask = palette_mask(image=image, palette=PALETTE)

    regions = find_grid_regions(mask=mask, rows=2, columns=2)
    assert len(regions) == 4
    # Reading order: top-left, top-right, bottom-left, bottom-right.
    assert regions[0][2] < regions[1][2]  # top-left is left of top-right
    assert regions[0][0] < regions[2][0]  # top-left is above bottom-left


def test_find_grid_regions_ignores_small_stray_content() -> None:
    image = _blank_image(120, 60)
    _fill(
        image, y0=5, y1=8, x0=5, x1=8, rgb=(255, 0, 0)
    )  # a tiny unrelated speck, e.g. a legend swatch
    _fill(image, y0=20, y1=60, x0=5, x1=55, rgb=(255, 0, 0))
    _fill(image, y0=80, y1=120, x0=5, x1=55, rgb=(255, 0, 0))
    mask = palette_mask(image=image, palette=PALETTE)

    regions = find_grid_regions(mask=mask, rows=2, columns=1)
    assert len(regions) == 2


def test_find_grid_regions_wrong_row_count_raises() -> None:
    image = _blank_image(50, 50)
    _fill(image, y0=10, y1=40, x0=10, x1=40, rgb=(255, 0, 0))
    mask = palette_mask(image=image, palette=PALETTE)

    with pytest.raises(ValueError, match="row bands"):
        find_grid_regions(mask=mask, rows=2, columns=1)


def test_find_grid_regions_wrong_column_count_raises() -> None:
    image = _blank_image(50, 50)
    _fill(image, y0=10, y1=40, x0=10, x1=40, rgb=(255, 0, 0))
    mask = palette_mask(image=image, palette=PALETTE)

    with pytest.raises(ValueError, match="column bands"):
        find_grid_regions(mask=mask, rows=1, columns=2)


def _make_day_grid(year: int, month: int, cell_size: int = 10) -> np.ndarray:
    first_of_month = date(year, month, 1)
    last_of_month = date(year, month, calendar.monthrange(year, month)[1])
    grid_start = first_of_month - timedelta(days=first_of_month.weekday())
    total_rows = -(-((last_of_month - grid_start).days + 1) // 7)

    return np.full((total_rows * cell_size, 7 * cell_size, 3), (255, 0, 0), dtype=np.uint8)


def test_build_cells_covers_every_day_of_the_month() -> None:
    image = _make_day_grid(2025, 7)
    cells = build_cells(
        image=image,
        year=2025,
        month=7,
        palette=PALETTE,
        mode=DetectionMode.SINGLE,
        presence_threshold=0.1,
    )
    assert {cell.datestamp for cell in cells} == {date(2025, 7, day) for day in range(1, 32)}


def test_build_cells_aligns_first_day_to_correct_weekday() -> None:
    # July 1st 2025 is a Tuesday.
    image = _make_day_grid(2025, 7, cell_size=10)
    cells = build_cells(
        image=image,
        year=2025,
        month=7,
        palette=PALETTE,
        mode=DetectionMode.SINGLE,
        presence_threshold=0.1,
    )
    by_date = {cell.datestamp: cell for cell in cells}
    assert by_date[date(2025, 7, 1)].colours == frozenset({Colour.RED})


def test_build_cells_marks_offset_from_offset_colours() -> None:
    image = _make_day_grid(2025, 7)
    cells = build_cells(
        image=image,
        year=2025,
        month=7,
        palette=PALETTE,
        mode=DetectionMode.SINGLE,
        presence_threshold=0.1,
        offset_colours={Colour.RED},
    )
    assert all(cell.is_offset for cell in cells)


def test_build_cells_no_colour_present_gives_empty_colours() -> None:
    image = np.zeros((40, 70, 3), dtype=np.uint8)  # blank - no palette colour anywhere
    cells = build_cells(
        image=image,
        year=2025,
        month=7,
        palette=PALETTE,
        mode=DetectionMode.SINGLE,
        presence_threshold=0.1,
    )
    assert all(cell.colours == frozenset() for cell in cells)
    assert all(not cell.is_offset for cell in cells)
