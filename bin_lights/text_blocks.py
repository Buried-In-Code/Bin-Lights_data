__all__ = ["MonthLabel", "locate_month_labels", "parse_month", "reading_order"]

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime

Word = tuple[float, float, float, float, str, int, int, int]


@dataclass(frozen=True)
class MonthLabel:
    x0: float
    y0: float
    x1: float
    y1: float
    month: int
    year: int


def parse_month(text: str) -> date | None:
    try:
        return datetime.strptime(text.strip(), "%B%Y").date()  # noqa: DTZ007 - date-only, tz is meaningless
    except ValueError:
        return None


def locate_month_labels(
    words: list[Word], date_fixes: dict[date, date] | None = None
) -> list[MonthLabel]:
    date_fixes = date_fixes or {}

    lines: dict[tuple[int, int], list[Word]] = {}
    for word in words:
        lines.setdefault((word[5], word[6]), []).append(word)

    labels: list[MonthLabel] = []
    for parts in lines.values():
        parts.sort(key=lambda part: part[0])
        joined = "".join(part[4] for part in parts).replace(" ", "")

        month_date = parse_month(joined)
        if month_date is None:
            continue
        month_date = date_fixes.get(month_date, month_date)

        labels.append(
            MonthLabel(
                x0=min(part[0] for part in parts),
                y0=min(part[1] for part in parts),
                x1=max(part[2] for part in parts),
                y1=max(part[3] for part in parts),
                month=month_date.month,
                year=month_date.year,
            )
        )

    return labels


def reading_order[HasPosition](
    items: list[HasPosition],
    *,
    rows: int,
    columns: int,
    x: Callable[[HasPosition], float],
    y: Callable[[HasPosition], float],
) -> list[HasPosition]:
    if len(items) != rows * columns:
        raise ValueError(f"Expected {rows * columns} items, found {len(items)}")

    by_row = sorted(items, key=y)
    ordered: list[HasPosition] = []
    for start in range(0, len(by_row), columns):
        ordered.extend(sorted(by_row[start : start + columns], key=x))
    return ordered
