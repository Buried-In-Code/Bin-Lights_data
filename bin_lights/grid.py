__all__ = ["build_cells", "find_grid_regions", "palette_mask"]

import calendar as cal
from datetime import date

import numpy as np

from bin_lights.colour_detection import DEFAULT_TOLERANCE, detect_cell_colours
from bin_lights.colours import RGB, Colour
from bin_lights.models import Cell, DetectionMode

Region = tuple[int, int, int, int]  # y0, y1, x0, x1


def palette_mask(
    image: np.ndarray, palette: dict[Colour, RGB], tolerance: int = DEFAULT_TOLERANCE
) -> np.ndarray:
    pixels = image.astype(int)
    mask = np.zeros(image.shape[:2], dtype=bool)
    for rgb in palette.values():
        mask |= np.all(np.abs(pixels - np.array(rgb)) <= tolerance, axis=-1)
    return mask


def _find_bands(has_content: np.ndarray, min_gap: int) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], has_content, [False]))
    edges = np.diff(padded.astype(int))
    starts = np.where(edges == 1)[0].tolist()
    ends = np.where(edges == -1)[0].tolist()

    merged: list[list[int]] = []
    for start, end in zip(starts, ends, strict=True):
        if merged and start - merged[-1][1] <= min_gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _discard_small_bands(
    bands: list[tuple[int, int]], relative_threshold: float = 0.3
) -> list[tuple[int, int]]:
    if not bands:
        return bands
    largest = max(end - start for start, end in bands)
    return [(start, end) for start, end in bands if (end - start) >= largest * relative_threshold]


def _largest_bands(bands: list[tuple[int, int]], count: int) -> list[tuple[int, int]]:
    if len(bands) <= count:
        return bands
    tallest = sorted(bands, key=lambda band: band[1] - band[0], reverse=True)[:count]
    return sorted(tallest)


def find_grid_regions(mask: np.ndarray, rows: int, columns: int, min_gap: int = 15) -> list[Region]:
    row_bands = _largest_bands(
        _discard_small_bands(_find_bands(mask.any(axis=1), min_gap=min_gap)), rows
    )
    if len(row_bands) != rows:
        raise ValueError(f"Expected {rows} row bands, found {len(row_bands)}: {row_bands}")

    regions: list[Region] = []
    for row_y0, row_y1 in row_bands:
        row_mask = mask[row_y0:row_y1]
        col_bands = _discard_small_bands(_find_bands(row_mask.any(axis=0), min_gap=min_gap))
        if len(col_bands) != columns:
            found = f"{row_y0}:{row_y1}, found {len(col_bands)}"
            raise ValueError(f"Expected {columns} column bands in row {found}: {col_bands}")

        for col_x0, col_x1 in col_bands:
            column_mask = row_mask[:, col_x0:col_x1]
            y_bands = _discard_small_bands(_find_bands(column_mask.any(axis=1), min_gap=min_gap))
            regions.append((row_y0 + y_bands[0][0], row_y0 + y_bands[-1][1], col_x0, col_x1))

    return regions


def build_cells(
    *,
    image: np.ndarray,
    year: int,
    month: int,
    palette: dict[Colour, RGB],
    mode: DetectionMode,
    presence_threshold: float,
    offset_colours: set[Colour] | None = None,
    coloured_weekdays: int = 7,
    wraps_month_overflow: bool = False,
) -> set[Cell]:
    offset_colours = offset_colours or set()

    _, days_in_month = cal.monthrange(year, month)
    first_weekday = date(year, month, 1).weekday()  # 0 = Monday .. 6 = Sunday

    standard_rows = -(-(first_weekday + days_in_month) // 7)  # ceil division
    compact_rows = -(-days_in_month // 7)
    overflow = first_weekday + days_in_month - compact_rows * 7
    use_wrap = wraps_month_overflow and standard_rows > compact_rows and overflow < first_weekday
    leading_rows = 0 if use_wrap else int(first_weekday >= coloured_weekdays)
    total_rows = compact_rows if use_wrap else standard_rows - leading_rows
    total_cells = total_rows * 7

    image_height, image_width, _ = image.shape
    cell_width = image_width // 7
    cell_height = image_height // total_rows

    cells: set[Cell] = set()
    for day in range(1, days_in_month + 1):
        position = first_weekday + (day - 1)
        if use_wrap and position >= total_cells:
            row, col = 0, position - total_cells
        else:
            row, col = divmod(position, 7)
            row -= leading_rows
        if row < 0:
            cells.add(Cell(datestamp=date(year, month, day)))
            continue

        x, y = col * cell_width, row * cell_height
        square = image[y : y + cell_height, x : x + cell_width]

        colours = detect_cell_colours(
            cell=square, palette=palette, mode=mode, presence_threshold=presence_threshold
        )
        cells.add(
            Cell(
                datestamp=date(year, month, day),
                colours=colours,
                is_offset=bool(colours & offset_colours),
            )
        )

    return cells
