import calendar
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Final

from rich import print  # noqa: A004

from bin_lights import get_project_root
from bin_lights.calendar_grid import process_calendar_squares, resolve_weekly_colours
from bin_lights.models import Cell, Context
from bin_lights.pdf_extract import (
    crop_calendar as crop_pdf_calendar,
    extract_calendars as extract_pdf_calendars,
)
from bin_lights.png_extract import (
    crop_calendar as crop_png_calendar,
    extract_calendars as extract_png_calendars,
)
from bin_lights.utils import Colour

DAY_NAME_TO_INDEX: Final[dict[str, int]] = {name: idx for idx, name in enumerate(calendar.day_name)}


def write_location_json_outputs(
    cells: set[Cell], location_weekdays: dict[str, str], always_include_colour: Colour | None = None
) -> None:
    for location_name, weekday_name in location_weekdays.items():
        print(f"Writing '{location_name}' data to file")

        weekday_index = DAY_NAME_TO_INDEX[weekday_name]

        matching_cells = sorted(
            (cell for cell in cells if cell.datestamp.weekday() == weekday_index),
            key=lambda cell: cell.datestamp,
        )

        dated_colours: dict[date, Colour] = {}
        for cell in matching_cells:
            if not cell.colour:
                continue
            actual_date = cell.datestamp + timedelta(days=1) if cell.is_offset else cell.datestamp
            dated_colours[actual_date] = cell.colour

        recent_items = list(dated_colours.items())[-100:]

        output_payload = {
            date_key.isoformat(): sorted(
                {always_include_colour, colour} if always_include_colour else {colour}
            )
            for date_key, colour in recent_items
        }

        output_path = Path("output") / f"{location_name}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w") as stream:
            json.dump(
                output_payload,
                stream,
                indent=2,
                default=lambda value: value.display if isinstance(value, Colour) else value,
            )
            stream.write("\n")


def extract_calendar_cells(
    contexts: list[Context],
    locations: dict[str, str],
    offset_colours: set[Colour] | None = None,
    always_colour: Colour | None = None,
) -> None:
    offset_colours = offset_colours or set()
    all_cells: set[Cell] = set()

    for ctx in contexts:
        print(f"Extracting calendar information from '{ctx.file.name}'")

        file_suffix = ctx.file.suffix.lower()
        if file_suffix == ".png":
            calendars = extract_png_calendars(
                file=ctx.file, rows=ctx.rows, columns=ctx.columns, date_fixes=ctx.date_fixes
            )
            crop_calendar = crop_png_calendar
        elif file_suffix == ".pdf":
            calendars = extract_pdf_calendars(file=ctx.file, rows=ctx.rows, columns=ctx.columns)
            crop_calendar = crop_pdf_calendar
        else:
            continue

        actionable_colours = set(ctx.colours) - offset_colours
        extracted_cells: set[Cell] = set()

        for calendar_page in calendars:
            cropped_image = crop_calendar(
                img=calendar_page.calendar_image, colours=list(ctx.colours.values())
            )

            extracted_cells.update(
                process_calendar_squares(
                    img=cropped_image,
                    year=calendar_page.year,
                    month=calendar_page.month,
                    colours=ctx.colours,
                    offset_colours=offset_colours,
                )
            )

        resolve_weekly_colours(cells=extracted_cells, action_colours=actionable_colours)

        for cell in extracted_cells:
            cell.colour = ctx.colour_fixes.get(cell.colour, cell.colour)  # ty: ignore[no-matching-overload]

        all_cells.update(extracted_cells)

    write_location_json_outputs(
        cells=all_cells, location_weekdays=locations, always_include_colour=always_colour
    )


def hex_to_rgb(hex_colour: str) -> tuple[int, int, int]:
    if not hex_colour.startswith("#"):
        raise ValueError("Colour must start with '#': %s", hex_colour)

    hex_value = hex_colour[1:].upper()
    if len(hex_value) != 6:  # noqa: PLR2004
        raise ValueError("Colour must be in the format '#RRGGBB': %s", hex_colour)

    try:
        return (int(hex_value[0:2], 16), int(hex_value[2:4], 16), int(hex_value[4:6], 16))
    except ValueError as err:
        raise ValueError("Colour contains invalid hex digits: %s", hex_colour) from err


def main() -> None:
    extract_calendar_cells(
        contexts=[
            Context(
                file=get_project_root()
                / "sources"
                / "2025-26-SWDC-recycling-and-rubbish-collection-calendar.pdf",
                colours={
                    Colour.RED: (239, 65, 35),
                    Colour.BLUE: (169, 221, 228),
                    Colour.YELLOW: (255, 223, 0),
                    Colour.BLACK: (35, 31, 32),
                },
                colour_fixes={Colour.RED: Colour.YELLOW},
            ),
            Context(
                file=get_project_root()
                / "sources"
                / "2026-27-SWDC-recycling-and-rubbish-collection-calendar.pdf",
                colours={
                    Colour.RED: (239, 65, 35),
                    Colour.BLUE: (169, 221, 228),
                    Colour.YELLOW: (255, 223, 0),
                    Colour.BLACK: (35, 31, 32),
                },
                colour_fixes={Colour.RED: Colour.YELLOW},
            ),
        ],
        locations={"Greytown": "Tuesday", "Martinborough": "Wednesday", "Featherston": "Thursday"},
        offset_colours={Colour.YELLOW, Colour.BLACK},
        always_colour=Colour.RED,
    )
    return

    extract_calendar_cells(
        contexts=[
            Context(
                file=get_project_root() / "sources" / "Razorback_Jul-2025_Jun-2026.png",
                colours={
                    Colour.GREEN: hex_to_rgb(hex_colour="#8FC73F"),
                    Colour.YELLOW: hex_to_rgb(hex_colour="#FFCA05"),
                },
                date_fixes={date(2025, 5, 1): date(2026, 5, 1)},
            )
        ],
        locations={"Razorback": "Friday"},
        always_colour=Colour.RED,
    )


if __name__ == "__main__":
    main()
