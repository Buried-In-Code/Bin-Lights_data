__all__ = ["Colour"]

from enum import Enum, auto


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
