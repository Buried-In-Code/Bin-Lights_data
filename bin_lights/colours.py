__all__ = ["RGB", "Colour"]

from enum import Enum, auto

RGB = tuple[int, int, int]


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
