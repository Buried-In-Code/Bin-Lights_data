__all__ = ["render_page"]

import io

import numpy as np
import pymupdf
from PIL import Image


def render_page(page: pymupdf.Page, zoom: float) -> np.ndarray:
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
    image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
    return np.array(image)
