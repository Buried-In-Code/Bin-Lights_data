import cv2
import numpy as np
from calendar_extract import keep_only_colours, remove_colours


def test_remove_colours_masks_pixels(monkeypatch) -> None:
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    img[:] = [10, 10, 10]

    def fake_in_range(*args, **kwargs):
        return np.array([[255, 0], [0, 0]], dtype=np.uint8)

    monkeypatch.setattr(cv2, "inRange", fake_in_range)

    result = remove_colours(img, [(10, 10, 10)])

    assert (result[0, 0] == [255, 255, 255]).all()


def test_keep_only_colours(monkeypatch) -> None:
    img = np.ones((2, 2, 3), dtype=np.uint8) * 50

    def fake_in_range(*args, **kwargs):
        return np.array([[255, 0], [0, 0]], dtype=np.uint8)

    monkeypatch.setattr(cv2, "inRange", fake_in_range)

    result = keep_only_colours(img, [(50, 50, 50)])

    assert result.shape == img.shape
