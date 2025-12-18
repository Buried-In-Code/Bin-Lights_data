from pathlib import Path

import calendar_extract as mod
import numpy as np


class FakeTextPage:
    def extractWORDS(self, delimiters=None):
        return [(0, 0, 10, 10, "January"), (11, 0, 20, 10, "2024")]


class FakePage:
    def get_pixmap(self):
        class PM:
            def tobytes(self, fmt) -> bytes:
                return b"\x89PNG\r\n\x1a\n"

        return PM()

    def get_textpage_ocr(self, tessdata=None):
        return FakeTextPage()


class FakeDoc:
    def __getitem__(self, idx):
        return FakePage()


def test_extract_png_calendars(monkeypatch) -> None:
    monkeypatch.setattr(mod.pymupdf, "open", lambda _: FakeDoc())
    monkeypatch.setattr(mod.cv2, "imdecode", lambda *_: np.zeros((200, 300, 3)))

    calendars = mod.extract_png_calendars(Path("fake.png"))

    assert len(calendars) == 1
    cal = calendars[0]
    assert cal.month == 1
    assert cal.year == 2024
