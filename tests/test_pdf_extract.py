from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from bin_lights.models import Calendar
from bin_lights.pdf_extract import crop_calendar, extract_calendars


def test_crop_calendar_crops_to_coloured_region() -> None:
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[3:7, 2:8] = (0, 0, 255)

    cropped = crop_calendar(img=img, colours=[(255, 0, 0)])

    assert cropped.size > 0
    assert (cropped == (0, 0, 255)).all()


@pytest.fixture
def fake_image() -> np.ndarray:
    return np.zeros((100, 100, 3), dtype=np.uint8)


@patch("bin_lights.pdf_extract.adjust_mappings")
@patch("bin_lights.pdf_extract.parse_month")
@patch("bin_lights.pdf_extract.cv2.imdecode")
@patch("bin_lights.pdf_extract.pymupdf.open")
def test_extract_calendars_happy_path(
    mock_open: MagicMock,
    mock_imdecode: MagicMock,
    mock_parse_month: MagicMock,
    mock_adjust_mappings: MagicMock,
    fake_image: np.ndarray,
) -> None:
    mock_page = MagicMock()
    mock_document = MagicMock()
    mock_document.__getitem__.return_value = mock_page
    mock_open.return_value = mock_document

    mock_pixmap = MagicMock()
    mock_pixmap.tobytes.return_value = b"pngbytes"
    mock_page.get_pixmap.return_value = mock_pixmap

    mock_imdecode.return_value = fake_image

    mock_page.get_textpage_ocr.return_value.extractBLOCKS.return_value = [
        (10, 10, 40, 40, "January2024", None),
        (50, 10, 90, 40, "February2024", None),
    ]

    mock_parse_month.side_effect = [MagicMock(month=1, year=2024), MagicMock(month=2, year=2024)]

    mock_adjust_mappings.return_value = [
        {"x0": 10, "y0": 10, "x1": 40, "y1": 40, "month": 1, "year": 2024},
        {"x0": 50, "y0": 10, "x1": 90, "y1": 40, "month": 2, "year": 2024},
    ]

    result = extract_calendars(Path("dummy.pdf"), rows=1, columns=2)

    assert len(result) == 2
    assert all(isinstance(c, Calendar) for c in result)

    jan, feb = result
    assert (jan.year, jan.month) == (2024, 1)
    assert (feb.year, feb.month) == (2024, 2)

    assert jan.calendar_image.shape == (30, 30, 3)
    assert feb.calendar_image.shape == (30, 40, 3)


@patch("bin_lights.pdf_extract.adjust_mappings")
@patch("bin_lights.pdf_extract.parse_month")
@patch("bin_lights.pdf_extract.cv2.imdecode")
@patch("bin_lights.pdf_extract.pymupdf.open")
def test_extract_calendars_ignores_invalid_month_text(
    mock_open: MagicMock,
    mock_imdecode: MagicMock,
    mock_parse_month: MagicMock,
    mock_adjust_mappings: MagicMock,
    fake_image: np.ndarray,
) -> None:
    mock_page = MagicMock()
    mock_document = MagicMock()
    mock_document.__getitem__.return_value = mock_page
    mock_open.return_value = mock_document

    mock_page.get_pixmap.return_value.tobytes.return_value = b"pngbytes"
    mock_imdecode.return_value = fake_image

    mock_page.get_textpage_ocr.return_value.extractBLOCKS.return_value = [
        (10, 10, 40, 40, "NotAMonth", None)
    ]

    mock_parse_month.return_value = None
    mock_adjust_mappings.return_value = []

    result = extract_calendars(Path("dummy.pdf"), rows=1, columns=1)

    assert result == []


@patch("bin_lights.pdf_extract.adjust_mappings")
@patch("bin_lights.pdf_extract.parse_month")
@patch("bin_lights.pdf_extract.cv2.imdecode")
@patch("bin_lights.pdf_extract.pymupdf.open")
def test_extract_calendars_sorts_blocks_by_year_month(
    mock_open: MagicMock,
    mock_imdecode: MagicMock,
    mock_parse_month: MagicMock,
    mock_adjust_mappings: MagicMock,
    fake_image: np.ndarray,
) -> None:
    mock_page = MagicMock()
    mock_document = MagicMock()
    mock_document.__getitem__.return_value = mock_page
    mock_open.return_value = mock_document

    mock_page.get_pixmap.return_value.tobytes.return_value = b"pngbytes"
    mock_imdecode.return_value = fake_image

    mock_page.get_textpage_ocr.return_value.extractBLOCKS.return_value = [
        (10, 10, 40, 40, "February2024", None),
        (10, 10, 40, 40, "January2024", None),
    ]

    mock_parse_month.side_effect = [MagicMock(month=2, year=2024), MagicMock(month=1, year=2024)]

    mock_adjust_mappings.return_value = []

    extract_calendars(Path("dummy.pdf"), rows=1, columns=2)

    passed_blocks = mock_adjust_mappings.call_args.kwargs["raw_blocks"]

    assert [(b["year"], b["month"]) for b in passed_blocks] == [(2024, 1), (2024, 2)]
