__all__ = ["crop_calendar", "extract_calendars"]

from pathlib import Path

import cv2
import numpy as np
import pymupdf
import tessdata

from bin_lights.image_ops import adjust_mappings, keep_only_colours, parse_month
from bin_lights.models import Calendar


def crop_calendar(img: np.ndarray, colours: list[tuple[int, int, int]]) -> np.ndarray:
    bgr_colours = [rgb[::-1] for rgb in colours]

    masked = keep_only_colours(img=img, colours_to_keep=bgr_colours)
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

    non_empty_rows = np.max(gray, axis=1) > 0
    non_empty_cols = np.max(gray, axis=0) > 0

    min_y = np.where(non_empty_rows)[0][0]
    max_y = np.where(non_empty_rows)[0][-1]
    min_x = np.where(non_empty_cols)[0][0]
    max_x = np.where(non_empty_cols)[0][-1]

    return img[min_y:max_y, min_x:max_x]


def extract_calendars(file: Path, rows: int, columns: int) -> list[Calendar]:
    document = pymupdf.open(file)
    page = document[0]

    pixmap = page.get_pixmap()
    png_bytes = pixmap.tobytes("png")
    img = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_COLOR)

    text_page = page.get_textpage_ocr(tessdata=tessdata.data_path())

    raw_blocks = []
    for block in text_page.extractBLOCKS():
        raw_text = block[4].replace(" ", "")
        month_date = parse_month(text=raw_text)
        if not month_date:
            continue

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
        y_padding=0,
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
