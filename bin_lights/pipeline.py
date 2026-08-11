__all__ = [
    "extract_location_data",
    "extract_source_cells",
    "resolve_weekly_colours",
    "write_location_json_outputs",
]

import calendar as cal
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Final

from bin_lights.colours import Colour
from bin_lights.grid import build_cells
from bin_lights.models import Cell, SourceConfig
from bin_lights.sources import extract_calendars

WEEKDAY_INDEX: Final[dict[str, int]] = {name: index for index, name in enumerate(cal.day_name)}

HISTORY_LIMIT: Final[int] = 100


def resolve_weekly_colours(cells: set[Cell], offset_colours: set[Colour] | None = None) -> None:
    offset_colours = offset_colours or set()
    cells_by_date = {cell.datestamp: cell for cell in cells}
    if not cells_by_date:
        return

    all_dates = sorted(cells_by_date)
    week_start = all_dates[0] - timedelta(days=all_dates[0].weekday())

    current_week_start = week_start
    while current_week_start <= all_dates[-1]:
        week_cells = [
            cells_by_date[current_week_start + timedelta(days=offset)]
            for offset in range(7)
            if (current_week_start + timedelta(days=offset)) in cells_by_date
        ]

        candidates = {
            cell.colours
            for cell in week_cells
            if cell.colours and not (cell.colours & offset_colours)
        }
        if len(candidates) > 1:
            message = f"Week starting {current_week_start} has ambiguous colours: {candidates}"
            raise ValueError(message)

        if candidates:
            resolved_colours = next(iter(candidates))
            for cell in week_cells:
                if cell.colours:
                    cell.colours = resolved_colours

        current_week_start += timedelta(days=7)


def extract_source_cells(config: SourceConfig) -> set[Cell]:
    calendars = extract_calendars(config=config)

    cells: set[Cell] = set()
    for calendar_page in calendars:
        cells.update(
            build_cells(
                image=calendar_page.image,
                year=calendar_page.year,
                month=calendar_page.month,
                palette=config.palette,
                mode=config.mode,
                presence_threshold=config.presence_threshold,
                offset_colours=config.offset_colours,
                coloured_weekdays=config.coloured_weekdays,
                wraps_month_overflow=config.wraps_month_overflow,
            )
        )

    resolve_weekly_colours(cells=cells, offset_colours=config.offset_colours)

    for cell in cells:
        cell.colours = frozenset(config.colour_fixes.get(colour, colour) for colour in cell.colours)

    return cells


def write_location_json_outputs(
    cells: set[Cell],
    location_weekdays: dict[str, str],
    output_dir: Path,
    always_colours: set[Colour] | None = None,
    history_limit: int = HISTORY_LIMIT,
) -> None:
    always_colours = always_colours or set()
    output_dir.mkdir(parents=True, exist_ok=True)

    for location_name, weekday_name in location_weekdays.items():
        weekday_index = WEEKDAY_INDEX[weekday_name]
        matching_cells = sorted(
            (cell for cell in cells if cell.datestamp.weekday() == weekday_index and cell.colours),
            key=lambda cell: cell.datestamp,
        )

        dated_colours: dict[date, frozenset[Colour]] = {}
        for cell in matching_cells:
            actual_date = cell.datestamp + timedelta(days=1) if cell.is_offset else cell.datestamp
            dated_colours[actual_date] = cell.colours | always_colours

        recent_items = list(dated_colours.items())[-history_limit:]
        output_payload = {
            date_key.isoformat(): sorted(c.display for c in colours)
            for date_key, colours in recent_items
        }

        output_path = output_dir / f"{location_name}.json"
        with output_path.open("w") as stream:
            json.dump(output_payload, stream, indent=2)
            stream.write("\n")


def extract_location_data(
    configs: list[SourceConfig],
    locations: dict[str, str],
    output_dir: Path,
    always_colours: set[Colour] | None = None,
) -> None:
    all_cells: set[Cell] = set()
    for config in configs:
        all_cells.update(extract_source_cells(config=config))

    write_location_json_outputs(
        cells=all_cells,
        location_weekdays=locations,
        output_dir=output_dir,
        always_colours=always_colours,
    )
