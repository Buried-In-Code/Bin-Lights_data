from pathlib import Path

import bin_lights.__main__ as mod


def test_main_writes_output(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    monkeypatch.setattr(Path, "glob", lambda self, pattern: [])  # noqa: ARG005
    monkeypatch.setattr("bin_lights.png_extract.extract_calendars", lambda *a, **k: [])
    monkeypatch.setattr("bin_lights.pdf_extract.extract_calendars", lambda *a, **k: [])

    monkeypatch.chdir(tmp_path)

    mod.main()

    output_dir = tmp_path / "output"
    assert output_dir.exists()
