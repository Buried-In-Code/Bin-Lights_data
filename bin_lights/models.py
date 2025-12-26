__all__ = ["Calendar", "Cell", "Colour", "Context"]

from dataclasses import dataclass, field
from datetime import date
from enum import Enum, auto
from pathlib import Path

import numpy as np


class Colour(Enum):
    RED = auto()
    YELLOW = auto()
    GREEN = auto()
    CYAN = auto()
    BLUE = auto()
    MAGENTA = auto()
    BLACK = auto()
    WHITE = auto()

    @property
    def display(self) -> str:
        return self.name.lower()

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Colour):
            return NotImplemented
        return self.value < other.value


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
