from calendar_extract import parse_month


def test_parse_month_valid() -> None:
    dt = parse_month("January2024")
    assert dt.year == 2024
    assert dt.month == 1


def test_parse_month_invalid() -> None:
    assert parse_month("FooBar") is None
    assert parse_month("") is None
