__all__ = ["crop_calendar", "extract_calendars"]

from datetime import date
from pathlib import Path

import cv2
import numpy as np
import pymupdf
import tessdata

from bin_lights.image_ops import (
    adjust_mappings,
    find_largest_inner_contour,
    parse_month,
    remove_colours,
)
from bin_lights.models import Calendar


def crop_calendar(img: np.ndarray, colours: list[tuple[int, int, int]]) -> np.ndarray:
    bgr_colours = [rgb[::-1] for rgb in colours]

    masked = remove_colours(img=img, colours_to_remove=bgr_colours)
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    inner_contour = find_largest_inner_contour(contours=contours, count=1)[0]
    x, y, w, h = cv2.boundingRect(inner_contour)

    return img[y : y + h, x : x + w]


def extract_calendars(
    file: Path, rows: int, columns: int, date_fixes: dict[date, date]
) -> list[Calendar]:
    document = pymupdf.open(file)
    page = document[0]

    pixmap = page.get_pixmap()
    png_bytes = pixmap.tobytes("png")
    img = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_COLOR)

    text_page = page.get_textpage_ocr(tessdata=tessdata.data_path())
    words = text_page.extractWORDS(delimiters="\n")

    raw_blocks: list[dict[str, int]] = []

    for index in range(len(words) - 1):
        first_word = words[index][4]
        second_word = words[index + 1][4]

        month_date = parse_month(text=first_word + second_word)
        if not month_date:
            continue
        month_date = date_fixes.get(month_date, month_date)

        block = words[index]
        raw_blocks.append(
            {
                "x0": block[0] - 5,
                "y0": block[1] - 5,
                "x1": block[2] + 5,
                "y1": block[3] + 5,
                "month": month_date.month,
                "year": month_date.year,
            }
        )

    raw_blocks.sort(key=lambda b: (b["year"], b["month"]))

    adjusted_blocks = adjust_mappings(
        raw_blocks=raw_blocks,
        image_edges=img.shape[:2],
        x_padding=20,
        y_padding=20,
        rows=rows,
        columns=columns,
    )

    return [
        Calendar(
            month=block["month"],
            year=block["year"],
            calendar_image=img[block["y0"] : block["y1"], block["x0"] : block["x1"]],
        )
        for block in adjusted_blocks
    ]
