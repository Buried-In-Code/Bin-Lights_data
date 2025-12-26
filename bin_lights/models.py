__all__ = ["Calendar", "Cell", "Context"]

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np

from bin_lights.utils import Colour


@dataclass(frozen=True)
class Calendar:
    month: int
    year: int
    calendar_image: np.ndarray

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Calendar):
            return NotImplemented
        return (self.year, self.month) == (other.year, other.month)

    def __hash__(self) -> int:
        return hash((self.year, self.month))


@dataclass
class Cell:
    datestamp: date
    colour: Colour | None
    is_offset: bool = False

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Cell):
            return NotImplemented
        return self.datestamp == other.datestamp

    def __hash__(self) -> int:
        return hash(self.datestamp)


@dataclass(kw_only=True)
class Context:
    file: Path
    colours: dict[Colour, tuple[int, int, int]]
    rows: int = 4
    columns: int = 3
    date_fixes: dict[date, date] = field(default_factory=dict)
    colour_fixes: dict[Colour, Colour] = field(default_factory=dict)
