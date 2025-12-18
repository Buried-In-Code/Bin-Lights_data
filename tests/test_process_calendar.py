import numpy as np
from calendar_extract import PNG_COLOURS, process_calendar_squares


def test_process_calendar_single_day(monkeypatch) -> None:
    img = np.zeros((100, 700, 3), dtype=np.uint8)

    def fake_analyze_square(*args, **kwargs):
        return (0, 0, 255)  # red (BGR)

    monkeypatch.setattr("calendar_extract.analyze_square", fake_analyze_square)

    cells = process_calendar_squares(img=img, year=2024, month=1, colours=PNG_COLOURS)

    assert len(cells) == 31
    assert any(cell.is_recycling for cell in cells)
