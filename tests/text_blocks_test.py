from datetime import date

import pytest

from bin_lights.text_blocks import MonthLabel, locate_month_labels, parse_month, reading_order


def test_parse_month_valid() -> None:
    assert parse_month("JULY2025") == date(2025, 7, 1)


def test_parse_month_invalid() -> None:
    assert parse_month("NOTAMONTH") is None


def test_parse_month_strips_whitespace() -> None:
    assert parse_month("  JULY2025\n") == date(2025, 7, 1)


def _word(*, x0: float, text: str, block: int, line: int, word: int, y0: float = 0.0) -> tuple:
    return (x0, y0, x0 + 10, y0 + 10, text, block, line, word)


def test_locate_month_labels_single_word() -> None:
    words = [
        _word(x0=70, text="JULY", block=0, line=0, word=0),
        _word(x0=135, text="2025", block=0, line=0, word=1),
    ]
    labels = locate_month_labels(words)
    assert len(labels) == 1
    assert labels[0].month == 7
    assert labels[0].year == 2025


def test_locate_month_labels_wrapped_across_three_words() -> None:
    words = [
        _word(x0=0, text="SEPTE", block=26, line=0, word=0),
        _word(x0=20, text="MBER", block=26, line=0, word=1),
        _word(x0=45, text="2025", block=26, line=0, word=2),
    ]
    labels = locate_month_labels(words)
    assert len(labels) == 1
    assert labels[0].month == 9
    assert labels[0].year == 2025


def test_locate_month_labels_ignores_non_month_lines() -> None:
    words = [
        _word(x0=0, text="M", block=1, line=0, word=0),
        _word(x0=10, text="T", block=1, line=0, word=1),
        _word(x0=20, text="W", block=1, line=0, word=2),
    ]
    assert locate_month_labels(words) == []


def test_locate_month_labels_applies_date_fixes() -> None:
    words = [
        _word(x0=0, text="MAY", block=4, line=1, word=0),
        _word(x0=30, text="2025", block=4, line=1, word=1),
    ]
    labels = locate_month_labels(words, date_fixes={date(2025, 5, 1): date(2026, 5, 1)})
    assert labels[0].month == 5
    assert labels[0].year == 2026


def _label(x0: float, y0: float, month: int = 1, year: int = 2025) -> MonthLabel:
    return MonthLabel(x0=x0, y0=y0, x1=x0 + 10, y1=y0 + 10, month=month, year=year)


def test_reading_order_sorts_top_to_bottom_left_to_right() -> None:
    # Deliberately scrambled input order, 2 items per row.
    labels = [
        _label(x0=100, y0=100, month=4),
        _label(x0=100, y0=0, month=2),
        _label(x0=0, y0=100, month=3),
        _label(x0=0, y0=0, month=1),
    ]
    ordered = reading_order(
        labels, rows=2, columns=2, x=lambda label: label.x0, y=lambda label: label.y0
    )
    assert [label.month for label in ordered] == [1, 2, 3, 4]


def test_reading_order_wrong_count_raises() -> None:
    labels = [_label(x0=0, y0=0)]
    with pytest.raises(ValueError, match="Expected 4 items"):
        reading_order(labels, rows=2, columns=2, x=lambda label: label.x0, y=lambda label: label.y0)
