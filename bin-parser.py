import calendar
import json
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pytesseract


@dataclass
class Calendar:
    month: int
    year: int
    calendar_image: np.ndarray


@dataclass
class Cell:
    datestamp: date
    colour: str | None
    is_recycling: bool = False
    is_glass: bool = False
    is_offset: bool = False

    def __eq__(self, other) -> bool:  # noqa: ANN001
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.datestamp == other.datestamp

    def __hash__(self) -> int:
        return hash((type(self), self.datestamp))


colours = {"red": (255, 0, 0), "blue": (148, 220, 248), "yellow": (255, 255, 0), "black": (0, 0, 0)}


def remove_colors(img: np.ndarray, colors_to_mask: list[tuple[int, int, int]]) -> np.ndarray:
    masked_image = img.copy()
    for color in colors_to_mask:
        lower = np.array([max(c - 10, 0) for c in color], dtype="uint8")
        upper = np.array([min(c + 10, 255) for c in color], dtype="uint8")
        mask = cv2.inRange(masked_image, lower, upper)
        masked_image[mask > 0] = [255, 255, 255]
    return masked_image


def find_and_filter_contours(
    contours: list[np.ndarray], width_range: tuple[int, int], height_range: tuple[int, int]
) -> list[np.ndarray]:
    return [
        c
        for c in contours
        if width_range[0] <= cv2.boundingRect(c)[2] <= width_range[1]
        and height_range[0] <= cv2.boundingRect(c)[3] <= height_range[1]
    ]


def analyze_square(square: np.ndarray, target_colors: list[tuple[int, int, int]]) -> str:
    pixels = square.reshape(-1, 3)
    color_counts = Counter(map(tuple, pixels))
    target_colors = [tuple(color[::-1]) for color in target_colors] + [(0, 0, 0)]
    return max(target_colors, key=lambda c: color_counts.get(c, 0))


def process_calendar_squares(
    img: np.ndarray, year: int, month: int, target_colors: list[tuple[int, int, int]]
) -> set[Cell]:
    height, width, _ = img.shape
    square_size = width // 7

    start_day, days_in_month = calendar.monthrange(year, month)

    colours_map = {tuple(v[::-1]): k for k, v in colours.items()}

    cell_map = {}
    for day in range(1, days_in_month + 1):
        cell_idx = start_day + day - 1
        row, col = divmod(cell_idx, 7)
        x, y = col * square_size, row * square_size
        square = img[y : y + square_size, x : x + square_size]
        predominant_color = analyze_square(square=square, target_colors=target_colors)
        color_name = colours_map.get(predominant_color)
        cell_map[(row, col)] = Cell(datestamp=date(year, month, day), colour=color_name)

    for (row, _col), cell in cell_map.items():
        row_colors = {cell_map[(r, c)].colour for (r, c) in cell_map if r == row}
        cell.is_recycling = "red" in row_colors
        cell.is_glass = "blue" in row_colors
        cell.is_offset = cell.colour in {"yellow", "black"}

    return set(cell_map.values())


def extract_calendars(
    img: np.ndarray, colours_to_remove: list[tuple[int, int, int]]
) -> list[Calendar]:
    bgr_colors_to_remove = [color[::-1] for color in colours_to_remove]
    masked_image = remove_colors(img=img, colors_to_mask=bgr_colors_to_remove)
    gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = find_and_filter_contours(
        contours=contours, width_range=(1680, 1715), height_range=(1200, 1470)
    )

    calendars = []
    for contour in filtered_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cropped_calendar = img[y : y + h, x : x + w]
        header_y = max(y - 320, 0)
        month_text = pytesseract.image_to_string(
            img[header_y : header_y + 130, x : x + 1300]
        ).strip()
        try:
            month_date = datetime.strptime(month_text, "%B %Y")  # noqa: DTZ007
            calendars.append(
                Calendar(
                    month=month_date.month, year=month_date.year, calendar_image=cropped_calendar
                )
            )
        except ValueError:
            pass

    return calendars


def main() -> None:
    locations = {"Greytown": "Tuesday", "Martinborough": "Wednesday", "Featherston": "Thursday"}

    all_cells = set()
    for img in Path("sources").glob("*.png"):
        cv_img = cv2.imread(str(img))
        colours_to_remove = [colours["red"], colours["blue"], colours["yellow"], colours["black"]]
        calendars = extract_calendars(img=cv_img, colours_to_remove=colours_to_remove)

        for cal in calendars:
            all_cells.update(
                process_calendar_squares(
                    img=cal.calendar_image,
                    year=cal.year,
                    month=cal.month,
                    target_colors=colours_to_remove,
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
            location_data[actual_date.isoformat()] = (
                "recycling" if cell.is_recycling else "glass" if cell.is_glass else ""
            )

        output_file = Path(f"output/{location}.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w") as stream:
            json.dump(location_data, stream, indent=4)


if __name__ == "__main__":
    main()
