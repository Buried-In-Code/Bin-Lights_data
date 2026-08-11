__all__ = ["Calendar", "Cell", "DetectionMode", "SourceConfig"]

from dataclasses import dataclass, field
from datetime import date
from enum import Enum, auto
from pathlib import Path

import numpy as np

from bin_lights.colours import RGB, Colour


class DetectionMode(Enum):
    SINGLE = auto()
    MULTI = auto()


@dataclass(frozen=True)
class Calendar:
    month: int
    year: int
    image: np.ndarray

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Calendar):
            return NotImplemented
        return (self.year, self.month) == (other.year, other.month)

    def __hash__(self) -> int:
        return hash((self.year, self.month))


@dataclass
class Cell:
    datestamp: date
    colours: frozenset[Colour] = frozenset()
    is_offset: bool = False

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Cell):
            return NotImplemented
        return self.datestamp == other.datestamp

    def __hash__(self) -> int:
        return hash(self.datestamp)


@dataclass(kw_only=True)
class SourceConfig:
    file: Path
    palette: dict[Colour, RGB]
    mode: DetectionMode = DetectionMode.SINGLE
    rows: int = 4
    columns: int = 3
    presence_threshold: float = 0.1
    date_fixes: dict[date, date] = field(default_factory=dict)
    colour_fixes: dict[Colour, Colour] = field(default_factory=dict)
    offset_colours: set[Colour] = field(default_factory=set)
    crop_colours: set[Colour] = field(default_factory=set)
    coloured_weekdays: int = 7
    wraps_month_overflow: bool = False
