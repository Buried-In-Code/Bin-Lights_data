__all__ = [
    "adjust_mappings",
    "find_largest_inner_contour",
    "keep_only_colours",
    "parse_month",
    "remove_colours",
]

from datetime import date, datetime
from typing import Final

import cv2
import numpy as np

COLOUR_TOLERANCE: Final[int] = 10


def find_largest_inner_contour(contours: list[np.ndarray], count: int = 1) -> list[np.ndarray]:
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return sorted_contours[1 : 1 + count]


def colour_bounds(colour: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    lower = np.array([max(c - COLOUR_TOLERANCE, 0) for c in colour], dtype=np.uint8)
    upper = np.array([min(c + COLOUR_TOLERANCE, 255) for c in colour], dtype=np.uint8)
    return lower, upper


def remove_colours(img: np.ndarray, colours_to_remove: list[tuple[int, int, int]]) -> np.ndarray:
    output = img.copy()

    for colour in colours_to_remove:
        lower, upper = colour_bounds(colour=colour)
        mask = cv2.inRange(output, lower, upper)
        output[mask > 0] = (255, 255, 255)

    return output


def keep_only_colours(img: np.ndarray, colours_to_keep: list[tuple[int, int, int]]) -> np.ndarray:
    combined_mask = np.zeros(img.shape[:2], dtype=np.uint8)

    for colour in colours_to_keep:
        lower, upper = colour_bounds(colour=colour)
        colour_mask = cv2.inRange(img, lower, upper)
        combined_mask = cv2.bitwise_or(combined_mask, colour_mask)

    return cv2.bitwise_and(img, img, mask=combined_mask)


def parse_month(text: str) -> date | None:
    try:
        return datetime.strptime(text.strip(), "%B%Y").date()  # noqa: DTZ007
    except ValueError:
        return None


def adjust_mappings(
    raw_blocks: list[dict[str, float | int]],
    image_edges: tuple[int, int],
    x_padding: int,
    y_padding: int,
    rows: int,
    columns: int,
) -> list[dict[str, int]]:
    adjusted_blocks: list[dict[str, int]] = []

    for row in range(rows):
        for col in range(columns):
            index = row * (rows - 1) + col
            block = raw_blocks[index]

            x0 = block["x0"]
            y0 = block["y0"]

            x1 = raw_blocks[index + 1]["x0"] - x_padding if col < columns - 1 else image_edges[1]
            y1 = raw_blocks[index + columns]["y0"] - y_padding if row < rows - 1 else image_edges[0]

            adjusted_blocks.append(
                {
                    "x0": int(x0),
                    "y0": int(y0),
                    "x1": int(x1),
                    "y1": int(y1),
                    "month": block["month"],
                    "year": block["year"],
                }
            )

    return adjusted_blocks
