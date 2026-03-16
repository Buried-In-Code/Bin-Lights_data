from datetime import date

import numpy as np

from bin_lights.image_ops import (
    adjust_mappings,
    find_largest_inner_contour,
    keep_only_colours,
    parse_month,
    remove_colours,
)


def test_find_largest_inner_contour_skips_largest() -> None:
    c1 = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
    c2 = np.array([[0, 0], [5, 0], [5, 10], [0, 10]])
    c3 = np.array([[0, 0], [2, 0], [2, 5], [0, 5]])

    contours = [c3, c1, c2]

    result = find_largest_inner_contour(contours, count=2)

    assert result == [c2, c3]


def test_remove_colours_replaces_matching_colour_with_white() -> None:
    img = np.array([[(10, 10, 10), (255, 0, 0)], [(10, 10, 10), (0, 0, 0)]], dtype=np.uint8)

    result = remove_colours(img, colours_to_remove=[(10, 10, 10)])

    assert (result[0, 0] == (255, 255, 255)).all()
    assert (result[1, 0] == (255, 255, 255)).all()
    assert (result[0, 1] == (255, 0, 0)).all()


def test_keep_only_colours_keeps_selected_colours() -> None:
    img = np.array([[(10, 10, 10), (255, 0, 0)], [(0, 255, 0), (0, 0, 255)]], dtype=np.uint8)

    result = keep_only_colours(img, colours_to_keep=[(255, 0, 0), (0, 0, 255)])

    assert (result[0, 1] == (255, 0, 0)).all()
    assert (result[1, 1] == (0, 0, 255)).all()
    assert (result[0, 0] == (0, 0, 0)).all()
    assert (result[1, 0] == (0, 0, 0)).all()


def test_parse_month_valid() -> None:
    result = parse_month("January2024")

    assert result == date(2024, 1, 1)


def test_parse_month_strips_whitespace() -> None:
    result = parse_month("  February2023  ")

    assert result == date(2023, 2, 1)


def test_parse_month_invalid_returns_none() -> None:
    assert parse_month("Jan 2024") is None
    assert parse_month("NotAMonth") is None


def test_adjust_mappings_basic_grid() -> None:
    raw_blocks = [
        {"x0": 0, "y0": 0, "month": 1, "year": 2024},
        {"x0": 50, "y0": 0, "month": 1, "year": 2024},
        {"x0": 0, "y0": 40, "month": 1, "year": 2024},
        {"x0": 50, "y0": 40, "month": 1, "year": 2024},
    ]

    result = adjust_mappings(
        raw_blocks=raw_blocks, image_edges=(100, 100), x_padding=5, y_padding=10, rows=2, columns=2
    )

    assert len(result) == 4

    first = result[0]
    assert first["x0"] == 0
    assert first["y0"] == 0
    assert first["x1"] == 45
    assert first["y1"] == 30
    assert first["month"] == 1
    assert first["year"] == 2024


def test_adjust_mappings_uses_image_edges_on_last_row_and_column() -> None:
    raw_blocks = [
        {"x0": 0, "y0": 0, "month": 1, "year": 2024},
        {"x0": 50, "y0": 0, "month": 1, "year": 2024},
        {"x0": 0, "y0": 40, "month": 1, "year": 2024},
        {"x0": 50, "y0": 40, "month": 1, "year": 2024},
    ]

    result = adjust_mappings(
        raw_blocks=raw_blocks, image_edges=(100, 100), x_padding=5, y_padding=10, rows=2, columns=2
    )

    bottom_right = result[-1]

    assert bottom_right["x1"] == 100
    assert bottom_right["y1"] == 100
