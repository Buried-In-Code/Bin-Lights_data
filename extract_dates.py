import calendar
import json
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pymupdf
import tessdata


@dataclass
class Calendar:
    month: int
    year: int
    calendar_image: np.ndarray

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return (self.year, self.month) == (other.year, other.month)

    def __hash__(self) -> int:
        return hash((type(self), self.year, self.month))


@dataclass
class Cell:
    datestamp: date
    colour: str | None
    is_recycling: bool = False
    is_glass: bool = False
    is_offset: bool = False

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.datestamp == other.datestamp

    def __hash__(self) -> int:
        return hash((type(self), self.datestamp))


ROWS = 4
COLUMNS = 3
COLOURS = {"red": (255, 0, 0), "blue": (148, 220, 248), "yellow": (255, 255, 0), "black": (0, 0, 0)}


def analyze_square(square: np.ndarray) -> str:
    pixels = square.reshape(-1, 3)
    color_counts = Counter(map(tuple, pixels))
    target_colors = [tuple(color[::-1]) for color in list(COLOURS.values())] + [(0, 0, 0)]
    return max(target_colors, key=lambda c: color_counts.get(c, 0))


def process_calendar_squares(img: np.ndarray, year: int, month: int) -> set[Cell]:
    img = crop_calendar(img=img)
    height, width, _ = img.shape
    square_size = width // 7
    start_day, days_in_month = calendar.monthrange(year, month)
    colours_map = {tuple(v[::-1]): k for k, v in COLOURS.items()}

    cell_map = {}
    from rich import print  # noqa: A004

    for day in range(1, days_in_month + 1):
        cell_idx = start_day + day - 1
        row, col = divmod(cell_idx, 7)

        x, y = col * square_size, row * square_size + 35
        square = img[y : y + square_size, x : x + square_size]

        predominant_colour = colours_map.get(analyze_square(square=square))
        print(f"{year:04}-{month:02}-{day:02}: {predominant_colour}")
        cell_map[(row, col)] = Cell(datestamp=date(year, month, day), colour=predominant_colour)

    for (row, _col), cell in cell_map.items():
        row_colours = {cell_map[(r, c)].colour for (r, c) in cell_map if r == row}
        cell.is_recycling = "red" in row_colours
        cell.is_glass = "blue" in row_colours
        cell.is_offset = cell.colour in {"yellow", "black"}

    return set(cell_map.values())


def find_and_filter_contours(
    contours: list[np.ndarray], width_range: tuple[int, int], height_range: tuple[int, int]
) -> list[np.ndarray]:
    return [
        c
        for c in contours
        if width_range[0] <= cv2.boundingRect(c)[2] <= width_range[1]
        and height_range[0] <= cv2.boundingRect(c)[3] <= height_range[1]
    ]


def remove_colours(img: np.ndarray, colours_to_mask: list[tuple[int, int, int]]) -> np.ndarray:
    masked_image = img.copy()
    for colour in colours_to_mask:
        lower = np.array([max(c - 10, 0) for c in colour], dtype="uint8")
        upper = np.array([min(c + 10, 255) for c in colour], dtype="uint8")
        mask = cv2.inRange(masked_image, lower, upper)
        masked_image[mask > 0] = [255, 255, 255]
    return masked_image


def crop_calendar(img: np.ndarray) -> np.ndarray:
    bgr_colours_to_remove = [colour[::-1] for colour in list(COLOURS.values())]
    masked_image = remove_colours(img=img, colours_to_mask=bgr_colours_to_remove)
    gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contour = find_and_filter_contours(
        contours=contours, width_range=(1680, 1715), height_range=(1200, 1470)
    )[0]
    x, y, w, h = cv2.boundingRect(filtered_contour)
    return img[y : y + h, x : x + w]


def parse_month(text: str) -> datetime | None:
    try:
        return datetime.strptime(text.strip(), "%B%Y")  # noqa: DTZ007
    except ValueError:
        return None


def adjust_mappings(
    mapped: list[dict[str, float | int]],
    edges: tuple[int, int],
    rows: int = ROWS,
    columns: int = COLUMNS,
) -> list[dict[str, int]]:
    output = []
    for row in range(rows):
        for col in range(columns):
            idx = row * (rows - 1) + col
            block = mapped[idx]

            x0, y0 = block["x0"], block["y0"]
            x1 = mapped[idx + 1]["x0"] - 20 if col < columns - 1 else edges[1]
            y1 = mapped[idx + COLUMNS]["y0"] - 20 if row < rows - 1 else edges[0]

            output.append(
                {
                    "x0": int(x0),
                    "y0": int(y0),
                    "x1": int(x1),
                    "y1": int(y1),
                    "month": block["month"],
                    "year": block["year"],
                }
            )
    return output


def extract_png_calendars(file: Path) -> list[Calendar]:
    page = pymupdf.open(file)[0]
    pix_bytes = page.get_pixmap().tobytes("png")
    img = cv2.imdecode(np.frombuffer(pix_bytes, np.uint8), cv2.IMREAD_COLOR)
    text_page = page.get_textpage_ocr(tessdata=tessdata.data_path())

    words = text_page.extractWORDS(delimiters="\n")
    mapped = []
    for idx, block in enumerate(words):
        if idx + 1 >= len(words):
            continue
        if month_date := parse_month(block[4] + words[idx + 1][4]):
            mapped.append(
                {
                    "x0": block[0] - 5,
                    "y0": block[1] - 5,
                    "x1": block[2] + 5,
                    "y1": block[3] + 5,
                    "month": month_date.month,
                    "year": month_date.year,
                }
            )
    mapped.sort(key=lambda x: (x["year"], x["month"]))
    mapped = adjust_mappings(mapped, edges=img.shape[:2], rows=1)

    return [
        Calendar(
            month=entry["month"],
            year=entry["year"],
            calendar_image=img[entry["y0"] : entry["y1"], entry["x0"] : entry["x1"]],
        )
        for entry in mapped
    ]


def extract_pdf_calendars(file: Path) -> list[Calendar]:
    page = pymupdf.open(file)[0]
    pix_bytes = page.get_pixmap().tobytes("png")
    img = cv2.imdecode(np.frombuffer(pix_bytes, np.uint8), cv2.IMREAD_COLOR)
    text_page = page.get_textpage_ocr(tessdata=tessdata.data_path())

    mapped = [
        {
            "x0": block[0] - 5,
            "y0": block[1] - 5,
            "x1": block[2] + 5,
            "y1": block[3] + 5,
            "month": month_date.month,
            "year": month_date.year,
        }
        for block in text_page.extractBLOCKS()
        if (month_date := parse_month(block[4].replace(" ", "")))
    ]
    mapped.sort(key=lambda x: (x["year"], x["month"]))
    mapped = adjust_mappings(mapped, edges=img.shape[:2])

    return [
        Calendar(
            month=entry["month"],
            year=entry["year"],
            calendar_image=img[entry["y0"] : entry["y1"], entry["x0"] : entry["x1"]],
        )
        for entry in mapped
    ]


def main() -> None:
    locations = {"Greytown": "Tuesday", "Martinborough": "Wednesday", "Featherston": "Thursday"}

    all_cells = set()
    for png_file in Path("sources").glob("*.png"):
        all_cells.update(
            cell
            for cal in extract_png_calendars(file=png_file)
            for cell in process_calendar_squares(
                img=cal.calendar_image, year=cal.year, month=cal.month
            )
        )
    for pdf_file in Path("sources").glob("*.pdf"):
        continue
        all_cells.update(
            cell
            for cal in extract_pdf_calendars(file=pdf_file)
            for cell in process_calendar_squares(
                img=cal.calendar_image, year=cal.year, month=cal.month
            )
        )

    for location, default_day in locations.items():
        location_data = {}

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
        location_data = {k.isoformat(): ["rubbish", v] for k, v in location_data.items()}

        output_file = Path(f"output/{location}.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w") as stream:
            json.dump(location_data, stream, indent=4)


if __name__ == "__main__":
    main()
