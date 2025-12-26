import json
import platform
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Final

from httpx import Client
from rich import print  # noqa: A004

from bin_lights import __version__
from bin_lights.utils import Colour

BASE_URL: Final[str] = "https://yokqi4ofx1.execute-api.ap-southeast-2.amazonaws.com/Live"
PARCEL_NUMBER: Final[str] = "10190"
OUTPUT_DIR: Final[Path] = Path("output")


@dataclass(slots=True)
class ApiField:
    caption: str
    name: str
    value: str


def flatten_fields(groups: list[list[ApiField]]) -> Iterable[ApiField]:
    for group in groups:
        yield from group


def parse_pickup_date(text: str) -> date | None:
    try:
        head = ",".join(text.split(",")[:2]).strip()
        return date.strptime(head, "%A, %d %B %Y")
    except ValueError:
        return None


def extract_pickup_date(fields: Iterable[ApiField], field_name: str) -> date | None:
    for field in fields:
        if field.name == field_name:
            return parse_pickup_date(field.value)
    return None


def load_location_data(location: str) -> dict[date, set[Colour]]:
    path = OUTPUT_DIR / f"{location}.json"
    if not path.exists():
        return {}

    with path.open() as f:
        raw = json.load(f)
    return {
        date.fromisoformat(k): {Colour[x.upper()] for x in v if x.upper() in Colour.__members__}
        for k, v in raw.items()
    }


def save_location_data(location: str, data: dict[date, set[Colour]]) -> None:
    path = OUTPUT_DIR / f"{location}.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    data = dict(sorted(data.items())[-100:])
    serialised = {day.isoformat(): sorted(bins) for day, bins in data.items()}

    with path.open("w") as f:
        json.dump(
            serialised, f, indent=2, default=lambda x: x.display if isinstance(x, Colour) else x
        )
        f.write("\n")


def fetch_fields(client: Client) -> list[list[ApiField]]:
    response = client.get("wcc_details_lookup", params={"fields": PARCEL_NUMBER})
    response.raise_for_status()

    return [[ApiField(**item) for item in group] for group in response.json()]


def main() -> None:
    client = Client(
        base_url=BASE_URL,
        headers={
            "Accept": "application/json",
            "User-Agent": f"Bin-Lights/{__version__}/{platform.system()}:{platform.release()}",
        },
        timeout=30,
    )

    raw_fields = fetch_fields(client=client)
    fields = list(flatten_fields(groups=raw_fields))

    yellow_date = extract_pickup_date(fields=fields, field_name="WasteNextPickupType1")
    green_date = extract_pickup_date(fields=fields, field_name="WasteNextPickupType2")

    print(f"Yellow: {yellow_date}")
    print(f"Green: {green_date}")

    location = "Razorback"
    schedule = load_location_data(location=location)

    if yellow_date:
        schedule.setdefault(yellow_date, set()).update({Colour.RED, Colour.YELLOW})

    if green_date:
        schedule.setdefault(green_date, set()).update({Colour.RED, Colour.GREEN})

    save_location_data(location=location, data=schedule)


if __name__ == "__main__":
    main()
