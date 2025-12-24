import calendar
import json
from datetime import date, timedelta
from pathlib import Path

from bin_lights.calendar_grid import process_calendar_squares
from bin_lights.models import Cell, Context
from bin_lights.pdf_extract import (
    crop_calendar as crop_pdf_calendar,
    extract_calendars as extract_pdf_calendars,
)
from bin_lights.png_extract import (
    crop_calendar as crop_png_calendar,
    extract_calendars as extract_png_calendars,
)


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def write_location_outputs(all_cells: set[Cell], locations: dict[str, str]) -> None:
    for location, default_day in locations.items():
        location_data: dict[date, str] = {}

        filtered_cells = [
            cell
            for cell in all_cells
            if cell.datestamp.weekday() == list(calendar.day_name).index(default_day)
        ]

        for cell in sorted(filtered_cells, key=lambda c: c.datestamp):
            actual_date = cell.datestamp + timedelta(days=1) if cell.is_offset else cell.datestamp
            location_data[actual_date] = (
                "recycling" if cell.is_recycling else "glass" if cell.is_glass else ""
            )

        location_data = dict(sorted(location_data.items())[-100:])
        output_payload = {k.isoformat(): ["rubbish", v] for k, v in location_data.items()}

        output_file = Path(f"output/{location}.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w") as stream:
            json.dump(output_payload, stream, indent=2)
            stream.write("\n")


def extract_dates(context: list[Context], locations: dict[str, str]) -> None:
    all_cells: set[Cell] = set()

    for ctx in context:
        if ctx.file.suffix.lower() == ".png":
            calendars = extract_png_calendars(file=ctx.file)
            cropper = crop_png_calendar
        elif ctx.file.suffix.lower() == ".pdf":
            calendars = extract_pdf_calendars(file=ctx.file)
            cropper = crop_pdf_calendar
        else:
            continue

        for cal in calendars:
            all_cells.update(
                process_calendar_squares(
                    img=cropper(img=cal.calendar_image, colours=list(ctx.colours.values())),
                    year=cal.year,
                    month=cal.month,
                    colours=ctx.colours,
                )
            )

    write_location_outputs(all_cells=all_cells, locations=locations)


def main() -> None:
    extract_dates(
        context=[
            Context(
                file=get_project_root() / "sources" / "South-Wairarapa_Jan-2025_Mar-2025.png",
                colours={
                    "red": (255, 0, 0),
                    "blue": (148, 220, 248),
                    "yellow": (255, 255, 0),
                    "black": (0, 0, 0),
                },
            ),
            Context(
                file=get_project_root() / "sources" / "South-Wairarapa_Apr-2025_Mar-2026.pdf",
                colours={
                    "red": (239, 65, 35),
                    "blue": (169, 221, 228),
                    "yellow": (255, 223, 0),
                    "black": (35, 31, 32),
                },
            ),
        ],
        locations={"Greytown": "Tuesday", "Martinborough": "Wednesday", "Featherston": "Thursday"},
    )


if __name__ == "__main__":
    main()
