import pytest

import main


def test_run_stops_before_era5_without_openeo_credentials(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(main, "username", None)
    monkeypatch.setattr(main, "password", None)
    monkeypatch.setattr(
        main,
        "load_metafilter_parameters",
        lambda *_: pytest.fail("ERA5 stage ran without openEO credentials"),
    )

    assert main.main() == 1
    assert "OPENEO_USERNAME and OPENEO_PASSWORD" in capsys.readouterr().err
