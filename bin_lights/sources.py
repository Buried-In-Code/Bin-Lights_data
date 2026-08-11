__all__ = ["RENDER_ZOOM", "extract_calendars"]

import pymupdf
import tessdata

from bin_lights.grid import find_grid_regions, palette_mask
from bin_lights.models import Calendar, SourceConfig
from bin_lights.rendering import render_page
from bin_lights.text_blocks import locate_month_labels, reading_order

RENDER_ZOOM = 3.0
DAYS_PER_WEEK = 7
OCR_DPI = 150


def _ocr_words(page: pymupdf.Page) -> list:
    text_page = page.get_textpage_ocr(full=True, dpi=OCR_DPI, tessdata=tessdata.data_path())
    return text_page.extractWORDS()


def _words_for(page: pymupdf.Page, suffix: str) -> list:
    if suffix != ".pdf":
        raise ValueError(f"Unsupported source file type: {suffix!r}")
    words = page.get_text("words")
    return words or _ocr_words(page)


def extract_calendars(config: SourceConfig) -> list[Calendar]:
    document = pymupdf.open(config.file)
    page = document[0]

    image = render_page(page=page, zoom=RENDER_ZOOM)
    words = _words_for(page=page, suffix=config.file.suffix.lower())

    labels = locate_month_labels(words=words, date_fixes=config.date_fixes)
    ordered_labels = reading_order(
        labels,
        rows=config.rows,
        columns=config.columns,
        x=lambda label: label.x0,
        y=lambda label: label.y0,
    )

    top = min(label.y0 for label in labels) * RENDER_ZOOM - 100
    left = min(label.x0 for label in labels) * RENDER_ZOOM - 100
    search_area = image[max(int(top), 0) :, max(int(left), 0) :]

    crop_palette = {
        colour: rgb
        for colour, rgb in config.palette.items()
        if not config.crop_colours or colour in config.crop_colours
    }
    mask = palette_mask(image=search_area, palette=crop_palette)
    regions = find_grid_regions(mask=mask, rows=config.rows, columns=config.columns)

    y_offset, x_offset = max(int(top), 0), max(int(left), 0)
    image_width = image.shape[1]

    calendars = []
    for label, (y0, y1, x0, x1) in zip(ordered_labels, regions, strict=True):
        abs_x0, abs_x1 = x_offset + x0, x_offset + x1

        if config.coloured_weekdays < DAYS_PER_WEEK:
            cell_width = (abs_x1 - abs_x0) / config.coloured_weekdays
            extra_columns = DAYS_PER_WEEK - config.coloured_weekdays
            abs_x1 = min(int(abs_x1 + cell_width * extra_columns), image_width)

        calendars.append(
            Calendar(
                month=label.month,
                year=label.year,
                image=image[y_offset + y0 : y_offset + y1, abs_x0:abs_x1],
            )
        )

    return calendars
