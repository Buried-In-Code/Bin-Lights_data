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
PNG_COLOURS = {
    "red": (255, 0, 0),
    "blue": (148, 220, 248),
    "yellow": (255, 255, 0),
    "black": (0, 0, 0),
}
PDF_COLOURS = {
    "red": (239, 65, 35),
    "blue": (169, 221, 228),
    "yellow": (255, 223, 0),
    "black": (35, 31, 32),
}


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


def find_largest_inner_contour(
    contours: list[np.ndarray], num_contours: int = 1
) -> list[np.ndarray]:
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return sorted_contours[1 : 1 + num_contours]


def remove_colours(img: np.ndarray, colours_to_mask: list[tuple[int, int, int]]) -> np.ndarray:
    masked_image = img.copy()
    for colour in colours_to_mask:
        lower = np.array([max(c - 10, 0) for c in colour], dtype="uint8")
        upper = np.array([min(c + 10, 255) for c in colour], dtype="uint8")
        mask = cv2.inRange(masked_image, lower, upper)
        masked_image[mask > 0] = [255, 255, 255]
    return masked_image


def keep_only_colours(img: np.ndarray, colours_to_keep: list[tuple[int, int, int]]) -> np.ndarray:
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for colour in colours_to_keep:
        lower = np.array([max(c - 10, 0) for c in colour], dtype="uint8")
        upper = np.array([min(c + 10, 255) for c in colour], dtype="uint8")
        colour_mask = cv2.inRange(img, lower, upper)
        mask = cv2.bitwise_or(mask, colour_mask)
    return cv2.bitwise_and(img, img, mask=mask)


def crop_png_calendar(img: np.ndarray) -> np.ndarray:
    bgr_colours_to_remove = [colour[::-1] for colour in list(PNG_COLOURS.values())]
    masked_image = remove_colours(img=img, colours_to_mask=bgr_colours_to_remove)
    gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    inner_contour = find_largest_inner_contour(contours, num_contours=1)[0]
    x, y, w, h = cv2.boundingRect(inner_contour)
    return img[y : y + h, x : x + w]


def crop_pdf_calendar(img: np.ndarray) -> np.ndarray:
    bgr_colours_to_keep = [colour[::-1] for colour in list(PDF_COLOURS.values())]
    masked_image = keep_only_colours(img=img, colours_to_keep=bgr_colours_to_keep)
    gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    rows = np.max(gray, axis=1) > 0
    min_y = np.where(rows)[0][0]
    max_y = np.where(rows)[0][-1]
    cols = np.max(gray, axis=0) > 0
    min_x = np.where(cols)[0][0]
    max_x = np.where(cols)[0][-1]
    return img[min_y:max_y, min_x:max_x]


def parse_month(text: str) -> datetime | None:
    try:
        return datetime.strptime(text.strip(), "%B%Y")  # noqa: DTZ007
    except ValueError:
        return None


def adjust_mappings(
    mapped: list[dict[str, float | int]],
    edges: tuple[int, int],
    x_offset: int,
    y_offset: int,
    rows: int = ROWS,
    columns: int = COLUMNS,
) -> list[dict[str, int]]:
    output = []
    for row in range(rows):
        for col in range(columns):
            idx = row * (rows - 1) + col
            block = mapped[idx]

            x0, y0 = block["x0"], block["y0"]
            x1 = mapped[idx + 1]["x0"] - x_offset if col < columns - 1 else edges[1]
            y1 = mapped[idx + COLUMNS]["y0"] - y_offset if row < rows - 1 else edges[0]

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
    mapped = adjust_mappings(mapped, edges=img.shape[:2], x_offset=20, y_offset=20, rows=1)

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
    mapped = adjust_mappings(mapped, edges=img.shape[:2], x_offset=20, y_offset=0)

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
                img=crop_png_calendar(img=cal.calendar_image),
                year=cal.year,
                month=cal.month,
                colours=PNG_COLOURS,
            )
        )
    for pdf_file in Path("sources").glob("*.pdf"):
        all_cells.update(
            cell
            for cal in extract_pdf_calendars(file=pdf_file)
            for cell in process_calendar_squares(
                img=crop_pdf_calendar(img=cal.calendar_image),
                year=cal.year,
                month=cal.month,
                colours=PDF_COLOURS,
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
            json.dump(location_data, stream, indent=2)
            stream.write("\n")


if __name__ == "__main__":
    main()
