from bin_lights.colours import Colour


def test_display() -> None:
    assert Colour.RED.display == "red"
    assert Colour.YELLOW.display == "yellow"


def test_ordering() -> None:
    assert Colour.RED < Colour.YELLOW
    assert sorted([Colour.YELLOW, Colour.RED]) == [Colour.RED, Colour.YELLOW]


def test_ordering_against_other_type() -> None:
    assert Colour.RED.__lt__(object()) is NotImplemented
