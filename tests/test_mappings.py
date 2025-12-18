from calendar_extract import adjust_mappings


def test_adjust_mappings_single_row() -> None:
    mapped = [
        {"x0": 0, "y0": 0, "month": 1, "year": 2024},
        {"x0": 100, "y0": 0, "month": 2, "year": 2024},
        {"x0": 200, "y0": 0, "month": 3, "year": 2024},
    ]

    result = adjust_mappings(
        mapped=mapped, edges=(300, 300), x_offset=10, y_offset=10, rows=1, columns=3
    )

    assert len(result) == 3
    assert result[0]["x0"] == 0
    assert result[0]["x1"] == 90
    assert result[-1]["x1"] == 300
