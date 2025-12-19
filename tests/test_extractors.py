from pathlib import Path

import numpy as np

import extract_dates as mod


class FakeTextPage:
    def extractWORDS(self, delimiters: str | None = None) -> list[tuple]:  # noqa: N802, ARG002
        return [
            (0, 0, 10, 10, "January"),
            (11, 0, 20, 10, "2024"),
            (30, 0, 40, 10, "February"),
            (41, 0, 50, 10, "2024"),
            (60, 0, 70, 10, "March"),
            (71, 0, 80, 10, "2024"),
        ]


class FakePage:
    def get_pixmap(self) -> "PM":  # noqa: F821
        class PM:
            def tobytes(self, fmt: str) -> bytes:  # noqa: ARG002
                return b"\x89PNG\r\n\x1a\n"

        return PM()

    def get_textpage_ocr(self, tessdata=None) -> FakeTextPage:  # noqa: ARG002, ANN001
        return FakeTextPage()


class FakeDoc:
    def __getitem__(self, idx: int) -> FakePage:
        return FakePage()


def test_extract_png_calendars(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(mod.pymupdf, "open", lambda _: FakeDoc())
    monkeypatch.setattr(mod.cv2, "imdecode", lambda *_: np.zeros((200, 300, 3)))

    calendars = mod.extract_png_calendars(Path("fake.png"))

    assert len(calendars) == 3
    assert [(c.month, c.year) for c in calendars] == [(1, 2024), (2, 2024), (3, 2024)]
