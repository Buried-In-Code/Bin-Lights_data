from pathlib import Path

import extract_dates as mod


def test_main_writes_output(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    monkeypatch.setattr(mod.Path, "glob", lambda self, pattern: [])  # noqa: ARG005
    monkeypatch.setattr(mod, "extract_png_calendars", lambda *a, **k: [])
    monkeypatch.setattr(mod, "extract_pdf_calendars", lambda *a, **k: [])

    monkeypatch.chdir(tmp_path)

    mod.main()

    output_dir = tmp_path / "output"
    assert output_dir.exists()
