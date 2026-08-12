from datetime import date

from bin_lights import get_project_root
from bin_lights.colours import Colour
from bin_lights.models import DetectionMode, SourceConfig
from bin_lights.pipeline import extract_location_data

SOUTH_WAIRARAPA_PALETTE = {
    Colour.RED: (239, 65, 35),
    Colour.BLUE: (169, 221, 228),
    Colour.YELLOW: (255, 223, 0),
    Colour.BLACK: (35, 31, 32),
}
WOLLONDILLY_SHIRE_PALETTE = {Colour.GREEN: (143, 199, 63), Colour.YELLOW: (255, 202, 5)}


def south_wairarapa_config(filename: str) -> SourceConfig:
    return SourceConfig(
        file=get_project_root() / "sources" / filename,
        palette=SOUTH_WAIRARAPA_PALETTE,
        mode=DetectionMode.SINGLE,
        offset_colours={Colour.YELLOW, Colour.BLACK},
        colour_fixes={Colour.RED: Colour.YELLOW},
        crop_colours={Colour.RED, Colour.BLUE, Colour.YELLOW},
    )


def wollondilly_shire_config(filename: str) -> SourceConfig:
    return SourceConfig(
        file=get_project_root() / "sources" / filename,
        palette=WOLLONDILLY_SHIRE_PALETTE,
        mode=DetectionMode.MULTI,
        presence_threshold=0.1,
        date_fixes={date(2025, 5, 1): date(2026, 5, 1)},
        coloured_weekdays=5,
        wraps_month_overflow=True,
    )


def main() -> None:
    output_dir = get_project_root() / "output"

    extract_location_data(
        configs=[
            south_wairarapa_config("South-Wairarapa_Apr-2025_Mar-2026.pdf"),
            south_wairarapa_config("South-Wairarapa_Apr-2026_Mar-2027.pdf"),
        ],
        locations={"Greytown": "Tuesday", "Martinborough": "Wednesday", "Featherston": "Thursday"},
        output_dir=output_dir,
        always_colours={Colour.RED},
    )

    extract_location_data(
        configs=[
            wollondilly_shire_config("Wollondilly-Shire_Jul-2025_Jun-2026.pdf"),
            wollondilly_shire_config("Wollondilly-Shire_Jul-2026_Jun-2027.pdf"),
        ],
        locations={"Razorback": "Friday"},
        output_dir=output_dir,
        always_colours={Colour.RED},
    )


if __name__ == "__main__":
    main()
