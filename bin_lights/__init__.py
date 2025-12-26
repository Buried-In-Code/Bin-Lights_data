__all__ = ["__version__", "get_project_root"]
__version__ = "0.2.0"

from functools import cache
from pathlib import Path


@cache
def get_project_root() -> Path:
    return Path(__file__).parent.parent
