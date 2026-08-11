import json
from datetime import date, timedelta
from typing import TYPE_CHECKING

import pytest

from bin_lights.colours import Colour
from bin_lights.models import Cell
from bin_lights.pipeline import resolve_weekly_colours, write_location_json_outputs

if TYPE_CHECKING:
    from pathlib import Path


def _week_of_cells(monday: date, colour_by_offset: dict[int, Colour | None]) -> set[Cell]:
    cells = set()
    for offset, colour in colour_by_offset.items():
        colours = frozenset({colour}) if colour is not None else frozenset()
        cells.add(Cell(datestamp=monday + timedelta(days=offset), colours=colours))
    return cells


def test_resolve_weekly_colours_collapses_week_to_one_colour() -> None:
    monday = date(2025, 6, 2)
    cells = _week_of_cells(monday, {0: Colour.RED, 1: Colour.RED, 2: Colour.RED})
    resolve_weekly_colours(cells=cells, offset_colours=set())
    assert all(cell.colours == frozenset({Colour.RED}) for cell in cells)


def test_resolve_weekly_colours_overrides_excluded_cells_too() -> None:
    monday = date(2025, 6, 2)
    cells = _week_of_cells(monday, {0: Colour.RED, 1: Colour.RED, 2: Colour.YELLOW})
    resolve_weekly_colours(cells=cells, offset_colours={Colour.YELLOW})
    assert all(cell.colours == frozenset({Colour.RED}) for cell in cells)


def test_resolve_weekly_colours_ambiguous_week_raises() -> None:
    monday = date(2025, 6, 2)
    cells = _week_of_cells(monday, {0: Colour.RED, 1: Colour.BLUE})
    with pytest.raises(ValueError, match="ambiguous colours"):
        resolve_weekly_colours(cells=cells, offset_colours=set())


def test_resolve_weekly_colours_leaves_uncoloured_week_alone() -> None:
    monday = date(2025, 6, 2)
    cells = _week_of_cells(monday, {0: None, 1: None})
    resolve_weekly_colours(cells=cells, offset_colours=set())
    assert all(cell.colours == frozenset() for cell in cells)


def test_resolve_weekly_colours_empty_input() -> None:
    resolve_weekly_colours(cells=set(), offset_colours={Colour.RED})  # should not raise


def test_write_location_json_outputs_filters_by_weekday(tmp_path: Path) -> None:
    cells = {
        Cell(datestamp=date(2025, 6, 3), colours=frozenset({Colour.RED})),  # Tuesday
        Cell(datestamp=date(2025, 6, 4), colours=frozenset({Colour.BLUE})),  # Wednesday
    }
    write_location_json_outputs(
        cells=cells, location_weekdays={"Somewhere": "Tuesday"}, output_dir=tmp_path
    )

    payload = json.loads((tmp_path / "Somewhere.json").read_text())
    assert payload == {"2025-06-03": ["red"]}


def test_write_location_json_outputs_shifts_offset_cells(tmp_path: Path) -> None:
    cells = {Cell(datestamp=date(2025, 6, 3), colours=frozenset({Colour.RED}), is_offset=True)}
    write_location_json_outputs(
        cells=cells, location_weekdays={"Somewhere": "Tuesday"}, output_dir=tmp_path
    )

    payload = json.loads((tmp_path / "Somewhere.json").read_text())
    assert list(payload.keys()) == ["2025-06-04"]


def test_write_location_json_outputs_applies_always_colours(tmp_path: Path) -> None:
    cells = {Cell(datestamp=date(2025, 6, 3), colours=frozenset({Colour.BLUE}))}
    write_location_json_outputs(
        cells=cells,
        location_weekdays={"Somewhere": "Tuesday"},
        output_dir=tmp_path,
        always_colours={Colour.RED},
    )

    payload = json.loads((tmp_path / "Somewhere.json").read_text())
    assert payload["2025-06-03"] == ["blue", "red"]


def test_write_location_json_outputs_skips_cells_with_no_colours(tmp_path: Path) -> None:
    cells = {Cell(datestamp=date(2025, 6, 3), colours=frozenset())}
    write_location_json_outputs(
        cells=cells, location_weekdays={"Somewhere": "Tuesday"}, output_dir=tmp_path
    )

    payload = json.loads((tmp_path / "Somewhere.json").read_text())
    assert payload == {}


def test_write_location_json_outputs_respects_history_limit(tmp_path: Path) -> None:
    monday = date(2020, 1, 6)
    cells = {
        Cell(datestamp=monday + timedelta(weeks=week), colours=frozenset({Colour.RED}))
        for week in range(10)
    }
    write_location_json_outputs(
        cells=cells, location_weekdays={"Somewhere": "Monday"}, output_dir=tmp_path, history_limit=3
    )

    payload = json.loads((tmp_path / "Somewhere.json").read_text())
    assert len(payload) == 3
    assert list(payload.keys()) == [
        (monday + timedelta(weeks=week)).isoformat() for week in (7, 8, 9)
    ]


def test_write_location_json_outputs_creates_output_dir(tmp_path: Path) -> None:
    nested = tmp_path / "nested" / "output"
    write_location_json_outputs(
        cells=set(), location_weekdays={"Somewhere": "Monday"}, output_dir=nested
    )
    assert (nested / "Somewhere.json").exists()
