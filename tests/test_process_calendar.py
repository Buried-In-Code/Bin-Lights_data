import numpy as np

from extract_dates import PNG_COLOURS, process_calendar_squares


def test_process_calendar_single_day(monkeypatch) -> None:  # noqa: ANN001
    img = np.zeros((100, 700, 3), dtype=np.uint8)

    def fake_analyze_square(*args, **kwargs) -> tuple[int, int, int]:  # noqa: ANN002, ANN003
        return (0, 0, 255)  # red (BGR)

    monkeypatch.setattr("extract_dates.analyze_square", fake_analyze_square)

    cells = process_calendar_squares(img=img, year=2024, month=1, colours=PNG_COLOURS)

    assert len(cells) == 31
    assert any(cell.is_recycling for cell in cells)
