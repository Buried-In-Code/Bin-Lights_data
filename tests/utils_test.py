from bin_lights.models import Colour


def test_colour_display_is_lowercase_name() -> None:
    assert Colour.RED.display == "red"
    assert Colour.MAGENTA.display == "magenta"


def test_colour_ordering_uses_enum_value() -> None:
    assert Colour.RED < Colour.YELLOW
    assert Colour.BLACK < Colour.WHITE


def test_colour_lt_with_non_colour_returns_not_implemented() -> None:
    assert Colour.RED.__lt__("red") is NotImplemented
