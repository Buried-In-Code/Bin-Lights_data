__all__ = ["crop_calendar", "extract_calendars"]

from pathlib import Path

import cv2
import numpy as np
import pymupdf
import tessdata

from bin_lights.image_ops import adjust_mappings, keep_only_colours, parse_month
from bin_lights.models import Calendar


def crop_calendar(img: np.ndarray, colours: list[tuple[int, int, int]]) -> np.ndarray:
    bgr_colours_to_keep = [colour[::-1] for colour in colours]
    masked_image = keep_only_colours(img=img, colours=bgr_colours_to_keep)
    gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    rows = np.max(gray, axis=1) > 0
    min_y = np.where(rows)[0][0]
    max_y = np.where(rows)[0][-1]
    cols = np.max(gray, axis=0) > 0
    min_x = np.where(cols)[0][0]
    max_x = np.where(cols)[0][-1]
    return img[min_y:max_y, min_x:max_x]


def extract_calendars(file: Path) -> list[Calendar]:
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
