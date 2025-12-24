import numpy as np

from bin_lights.calendar_grid import process_calendar_squares


def test_process_calendar_single_day(monkeypatch) -> None:  # noqa: ANN001
    img = np.zeros((100, 700, 3), dtype=np.uint8)

    def fake_analyze_square(*args, **kwargs) -> tuple[int, int, int]:  # noqa: ANN002, ANN003
        return (0, 0, 255)  # red (BGR)

    monkeypatch.setattr("bin_lights.calendar_grid.analyze_square", fake_analyze_square)

    cells = process_calendar_squares(
        img=img,
        year=2024,
        month=1,
        colours={
            "red": (255, 0, 0),
            "blue": (148, 220, 248),
            "yellow": (255, 255, 0),
            "black": (0, 0, 0),
        },
    )

    assert len(cells) == 31
    assert any(cell.is_recycling for cell in cells)
