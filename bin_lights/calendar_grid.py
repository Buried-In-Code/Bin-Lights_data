__all__ = ["process_calendar_squares", "resolve_weekly_colours"]

import calendar
from collections import Counter
from datetime import date, timedelta

import numpy as np

from bin_lights.models import Cell
from bin_lights.utils import Colour


def dominant_square_colour(
    square: np.ndarray, palette: dict[Colour, tuple[int, int, int]]
) -> tuple[int, int, int]:
    pixels = square.reshape(-1, 3)
    pixel_counts = Counter(map(tuple, pixels))

    candidate_colours = {tuple(rgb[::-1]) for rgb in palette.values()} | {(0, 0, 0)}

    return max(candidate_colours, key=lambda colour: pixel_counts.get(colour, 0))


def process_calendar_squares(
    img: np.ndarray,
    year: int,
    month: int,
    colours: dict[Colour, tuple[int, int, int]],
    offset_colours: set[Colour] | None = None,
) -> set[Cell]:
    offset_colours = offset_colours or set()

    _, days_in_month = calendar.monthrange(year, month)

    first_of_month = date(year, month, 1)
    last_of_month = date(year, month, days_in_month)

    grid_start = first_of_month - timedelta(days=first_of_month.weekday())
    total_days = (last_of_month - grid_start).days + 1
    total_rows = (total_days + 6) // 7

    image_height, image_width, _ = img.shape
    cell_width = image_width // 7
    cell_height = image_height // total_rows

    rgb_to_colour = {tuple(rgb[::-1]): colour for colour, rgb in colours.items()}

    cells: set[Cell] = set()

    for row in range(total_rows):
        for col in range(7):
            cell_date = grid_start + timedelta(days=row * 7 + col)
            if cell_date.month != month:
                continue

            x = col * cell_width
            y = row * cell_height
            square = img[y : y + cell_height, x : x + cell_width]

            rgb = dominant_square_colour(square=square, palette=colours)
            colour = rgb_to_colour.get(rgb)

            cells.add(Cell(datestamp=cell_date, colour=colour, is_offset=colour in offset_colours))

    return cells


def resolve_weekly_colours(cells: set[Cell], action_colours: set[Colour]) -> None:
    cells_by_date = {cell.datestamp: cell for cell in cells}
    all_dates = sorted(cells_by_date)

    if not all_dates:
        return

    week_start = all_dates[0] - timedelta(days=all_dates[0].weekday())
    week_end = all_dates[-1]

    current_week = week_start
    while current_week <= week_end:
        week_dates = {current_week + timedelta(days=i) for i in range(7)}
        week_cells = [cells_by_date[d] for d in week_dates if d in cells_by_date]

        if not week_cells:
            current_week += timedelta(days=7)
            continue

        action_candidates = {cell.colour for cell in week_cells if cell.colour in action_colours}

        if len(action_candidates) > 1:
            raise ValueError(
                f"Week starting {current_week} has ambiguous action colours: {action_candidates}"
            )

        if action_candidates:
            resolved_colour = next(iter(action_candidates))
            for cell in week_cells:
                cell.colour = resolved_colour

        current_week += timedelta(days=7)
