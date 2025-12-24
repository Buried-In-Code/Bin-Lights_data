__all__ = ["Calendar", "Cell", "Context"]

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np


@dataclass
class Calendar:
    month: int
    year: int
    calendar_image: np.ndarray

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return (self.year, self.month) == (other.year, other.month)

    def __hash__(self) -> int:
        return hash((type(self), self.year, self.month))


@dataclass
class Cell:
    datestamp: date
    colour: str | None
    is_recycling: bool = False
    is_glass: bool = False
    is_offset: bool = False

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.datestamp == other.datestamp

    def __hash__(self) -> int:
        return hash((type(self), self.datestamp))


@dataclass(kw_only=True)
class Context:
    file: Path
    colours: dict[str, tuple[int, int, int]]
