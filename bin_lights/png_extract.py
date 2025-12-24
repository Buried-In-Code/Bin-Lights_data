__all__ = ["crop_calendar", "extract_calendars"]

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
    bgr_colours_to_remove = [colour[::-1] for colour in colours]
    masked_image = remove_colours(img=img, colours_to_mask=bgr_colours_to_remove)
    gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    inner_contour = find_largest_inner_contour(contours, num_contours=1)[0]
    x, y, w, h = cv2.boundingRect(inner_contour)
    return img[y : y + h, x : x + w]


def extract_calendars(file: Path) -> list[Calendar]:
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
