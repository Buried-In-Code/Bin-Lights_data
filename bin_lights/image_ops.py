__all__ = [
    "adjust_mappings",
    "find_largest_inner_contour",
    "keep_only_colours",
    "parse_month",
    "remove_colours",
]

from datetime import datetime
from typing import Final

import cv2
import numpy as np

ROWS: Final[int] = 4
COLUMNS: Final[int] = 3


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


def keep_only_colours(img: np.ndarray, colours: list[tuple[int, int, int]]) -> np.ndarray:
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for colour in colours:
        lower = np.array([max(c - 10, 0) for c in colour], dtype="uint8")
        upper = np.array([min(c + 10, 255) for c in colour], dtype="uint8")
        colour_mask = cv2.inRange(img, lower, upper)
        mask = cv2.bitwise_or(mask, colour_mask)
    return cv2.bitwise_and(img, img, mask=mask)


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
