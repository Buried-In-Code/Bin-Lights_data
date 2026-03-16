from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from bin_lights.models import Calendar
from bin_lights.png_extract import crop_calendar, extract_calendars


@patch("bin_lights.png_extract.find_largest_inner_contour")
@patch("bin_lights.png_extract.cv2.findContours")
@patch("bin_lights.png_extract.cv2.threshold")
@patch("bin_lights.png_extract.cv2.cvtColor")
@patch("bin_lights.png_extract.remove_colours")
def test_crop_calendar_uses_inner_contour_bounding_box(
    mock_remove_colours: MagicMock,
    mock_cvt: MagicMock,
    mock_thresh: MagicMock,
    mock_find: MagicMock,
    mock_find_inner: MagicMock,
) -> None:
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    mock_remove_colours.return_value = img
    mock_cvt.return_value = img[:, :, 0]
    mock_thresh.return_value = (None, img[:, :, 0])

    fake_contour = np.array([[[10, 20]], [[60, 20]], [[60, 70]], [[10, 70]]])
    mock_find.return_value = ([fake_contour], None)
    mock_find_inner.return_value = [fake_contour]

    cropped = crop_calendar(img, colours=[(255, 0, 0)])

    assert cropped.shape[:2] == (51, 51)


@pytest.fixture
def fake_image() -> np.ndarray:
    return np.zeros((200, 200, 3), dtype=np.uint8)


@patch("bin_lights.png_extract.adjust_mappings")
@patch("bin_lights.png_extract.parse_month")
@patch("bin_lights.png_extract.cv2.imdecode")
@patch("bin_lights.png_extract.pymupdf.open")
def test_extract_calendars_parses_adjacent_words_and_applies_date_fixes(
    mock_open: MagicMock,
    mock_imdecode: MagicMock,
    mock_parse_month: MagicMock,
    mock_adjust: MagicMock,
    fake_image: np.ndarray,
) -> None:
    mock_page = MagicMock()
    mock_doc = MagicMock()
    mock_doc.__getitem__.return_value = mock_page
    mock_open.return_value = mock_doc

    mock_page.get_pixmap.return_value.tobytes.return_value = b"png"
    mock_imdecode.return_value = fake_image

    mock_page.get_textpage_ocr.return_value.extractWORDS.return_value = [
        (10, 10, 40, 40, "January", None),
        (45, 10, 80, 40, "2024", None),
        (10, 60, 40, 90, "February", None),
        (45, 60, 80, 90, "2024", None),
    ]

    def parse_side_effect(*, text: str) -> date | None:
        if text == "February2024":
            return date(2024, 2, 1)
        if text == "January2024":
            return date(2024, 1, 1)
        return None

    mock_parse_month.side_effect = parse_side_effect

    mock_adjust.return_value = [
        {"x0": 10, "y0": 10, "x1": 80, "y1": 40, "month": 1, "year": 2024},
        {"x0": 10, "y0": 60, "x1": 80, "y1": 90, "month": 2, "year": 2024},
    ]

    result = extract_calendars(
        Path("dummy.png"), rows=2, columns=1, date_fixes={date(2024, 1, 1): date(2024, 1, 15)}
    )

    assert len(result) == 2
    assert isinstance(result[0], Calendar)

    assert (result[0].year, result[0].month) == (2024, 1)
    assert result[0].calendar_image.shape == (30, 70, 3)


@patch("bin_lights.png_extract.adjust_mappings")
@patch("bin_lights.png_extract.parse_month")
@patch("bin_lights.png_extract.cv2.imdecode")
@patch("bin_lights.png_extract.pymupdf.open")
def test_extract_calendars_ignores_non_month_word_pairs(
    mock_open: MagicMock,
    mock_imdecode: MagicMock,
    mock_parse_month: MagicMock,
    mock_adjust: MagicMock,
    fake_image: np.ndarray,
) -> None:
    mock_page = MagicMock()
    mock_doc = MagicMock()
    mock_doc.__getitem__.return_value = mock_page
    mock_open.return_value = mock_doc

    mock_page.get_pixmap.return_value.tobytes.return_value = b"png"
    mock_imdecode.return_value = fake_image

    mock_page.get_textpage_ocr.return_value.extractWORDS.return_value = [
        (10, 10, 40, 40, "Hello", None),
        (45, 10, 80, 40, "World", None),
    ]

    mock_parse_month.return_value = None
    mock_adjust.return_value = []

    result = extract_calendars(Path("dummy.png"), rows=1, columns=1, date_fixes={})

    assert result == []


@patch("bin_lights.png_extract.adjust_mappings")
@patch("bin_lights.png_extract.parse_month")
@patch("bin_lights.png_extract.cv2.imdecode")
@patch("bin_lights.png_extract.pymupdf.open")
def test_extract_calendars_sorts_blocks_before_adjustment(
    mock_open: MagicMock,
    mock_imdecode: MagicMock,
    mock_parse_month: MagicMock,
    mock_adjust: MagicMock,
    fake_image: np.ndarray,
) -> None:
    mock_page = MagicMock()
    mock_doc = MagicMock()
    mock_doc.__getitem__.return_value = mock_page
    mock_open.return_value = mock_doc

    mock_page.get_pixmap.return_value.tobytes.return_value = b"png"
    mock_imdecode.return_value = fake_image

    mock_page.get_textpage_ocr.return_value.extractWORDS.return_value = [
        (10, 10, 40, 40, "February", None),
        (45, 10, 80, 40, "2024", None),
        (10, 60, 40, 90, "January", None),
        (45, 60, 80, 90, "2024", None),
    ]

    def parse_side_effect(*, text: str) -> date | None:
        if text == "February2024":
            return date(2024, 2, 1)
        if text == "January2024":
            return date(2024, 1, 1)
        return None

    mock_parse_month.side_effect = parse_side_effect

    mock_adjust.return_value = []

    extract_calendars(Path("dummy.png"), rows=2, columns=1, date_fixes={})

    passed = mock_adjust.call_args.kwargs["raw_blocks"]
    assert [(b["year"], b["month"]) for b in passed] == [(2024, 1), (2024, 2)]
