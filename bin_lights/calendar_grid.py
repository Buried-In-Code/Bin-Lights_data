__all__ = ["analyze_square", "process_calendar_squares"]

import calendar
from collections import Counter
from datetime import date

import numpy as np

from bin_lights.models import Cell


def analyze_square(square: np.ndarray, colours: dict[str, tuple[int, int, int]]) -> str:
    pixels = square.reshape(-1, 3)
    color_counts = Counter(map(tuple, pixels))
    target_colors = [tuple(color[::-1]) for color in list(colours.values())] + [(0, 0, 0)]
    return max(target_colors, key=lambda c: color_counts.get(c, 0))


def process_calendar_squares(
    img: np.ndarray, year: int, month: int, colours: dict[str, tuple[int, int, int]]
) -> set[Cell]:
    start_day, days_in_month = calendar.monthrange(year, month)
    last_cell_idx = start_day + days_in_month - 1
    last_row, _ = divmod(last_cell_idx, 7)
    row_count = last_row + 1

    height, width, _ = img.shape
    square_width = width // 7
    square_height = height // row_count

    colours_map = {tuple(v[::-1]): k for k, v in colours.items()}

    cell_map = {}
    for day in range(1, days_in_month + 1):
        cell_idx = start_day + day - 1
        row, col = divmod(cell_idx, 7)

        x, y = col * square_width, row * square_height
        square = img[y : y + square_height, x : x + square_width]

        predominant_colour = colours_map.get(analyze_square(square=square, colours=colours))
        cell_map[(row, col)] = Cell(datestamp=date(year, month, day), colour=predominant_colour)

    for (row, _col), cell in cell_map.items():
        row_colours = {cell_map[(r, c)].colour for (r, c) in cell_map if r == row}
        cell.is_recycling = "red" in row_colours
        cell.is_glass = "blue" in row_colours
        cell.is_offset = cell.colour in {"yellow", "black"}

    return set(cell_map.values())
