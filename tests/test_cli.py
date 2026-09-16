from unittest.mock import Mock

import pandas as pd
import pytest

from metafilter import cli


@pytest.mark.parametrize(
    "entrypoint",
    [cli.process_era5_main, cli.download_open_meteo_main],
)
def test_cli_help_exits_without_running_work(entrypoint, capsys):
    with pytest.raises(SystemExit) as exc_info:
        entrypoint(["--help"])

    assert exc_info.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_process_cli_forwards_paths_and_writes_metrics(monkeypatch, tmp_path, capsys):
    metrics = pd.DataFrame({"date": ["2024-08-01"], "selected": [True]})
    process = Mock(
        return_value={"selected_dates": ["2024-08-01"], "daily_metrics": metrics}
    )
    monkeypatch.setattr(cli, "load_metafilter_parameters", Mock(return_value={"rules": {}}))
    monkeypatch.setattr(cli, "process_era5_data", process)
    output = tmp_path / "daily.csv"

    cli.process_era5_main(
        [
            "input.nc",
            "--filter",
            "profile.json",
            "--cloud-file",
            "cloud.nc",
            "--metrics-output",
            str(output),
        ]
    )

    from metafilter.config import AREA

    process.assert_called_once_with(
        "input.nc", {"rules": {}}, area=AREA, cloud_file_path="cloud.nc"
    )
    assert output.exists()
    assert "2024-08-01" in capsys.readouterr().out


def test_open_meteo_cli_forwards_month_bbox_and_variables(monkeypatch, capsys):
    download = Mock(return_value="data/era5/openmeteo_land_2023_06.nc")
    monkeypatch.setattr(cli, "download_open_meteo_land", download)

    cli.download_open_meteo_main(
        [
            "--year",
            "2023",
            "--month",
            "6",
            "--bbox",
            "17.0",
            "58.0",
            "19.0",
            "60.0",
            "--variables",
            "temperature_2m, precipitation",
        ]
    )

    download.assert_called_once_with(
        year=2023,
        month=6,
        area={"west": 17.0, "south": 58.0, "east": 19.0, "north": 60.0},
        variables=["temperature_2m", "precipitation"],
    )
    assert "openmeteo_land_2023_06.nc" in capsys.readouterr().out


@pytest.mark.parametrize(
    "bbox",
    [
        ["19", "58", "17", "60"],
        ["17", "60", "19", "58"],
        ["-181", "58", "19", "60"],
        ["17", "58", "19", "91"],
    ],
)
def test_open_meteo_cli_rejects_invalid_bbox(monkeypatch, bbox, capsys):
    download = Mock()
    monkeypatch.setattr(cli, "download_open_meteo_land", download)

    with pytest.raises(SystemExit) as exc_info:
        cli.download_open_meteo_main(["--bbox", *bbox])

    assert exc_info.value.code == 2
    assert "--bbox requires" in capsys.readouterr().err
    download.assert_not_called()


def test_process_cli_reports_missing_profile_without_traceback(capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli.process_era5_main(["input.nc", "--filter", "missing.json"])

    assert exc_info.value.code == 1
    assert "missing.json" in capsys.readouterr().err


def test_process_cli_reports_invalid_profile_shape(tmp_path, capsys):
    profile = tmp_path / "invalid.json"
    profile.write_text("[]")

    with pytest.raises(SystemExit) as exc_info:
        cli.process_era5_main(["input.nc", "--filter", str(profile)])

    assert exc_info.value.code == 1
    assert "JSON object" in capsys.readouterr().err


def test_open_meteo_cli_reports_download_failure(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "download_open_meteo_land",
        Mock(side_effect=RuntimeError("service unavailable")),
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.download_open_meteo_main([])

    assert exc_info.value.code == 1
    assert "service unavailable" in capsys.readouterr().err


def test_process_cli_forwards_bbox_as_area(monkeypatch, capsys):
    process = Mock(return_value={"selected_dates": ["2024-08-01"], "daily_metrics": pd.DataFrame()})
    monkeypatch.setattr(cli, "load_metafilter_parameters", Mock(return_value={"rules": {}}))
    monkeypatch.setattr(cli, "process_era5_data", process)

    cli.process_era5_main(["input.nc", "--filter", "p.json", "--bbox", "2", "48", "3", "49"])

    assert process.call_args.kwargs["area"] == {"west": 2.0, "south": 48.0, "east": 3.0, "north": 49.0}


def test_process_cli_defaults_to_configured_area(monkeypatch, capsys):
    from metafilter.config import AREA

    process = Mock(return_value={"selected_dates": ["2024-08-01"], "daily_metrics": pd.DataFrame()})
    monkeypatch.setattr(cli, "load_metafilter_parameters", Mock(return_value={"rules": {}}))
    monkeypatch.setattr(cli, "process_era5_data", process)

    cli.process_era5_main(["input.nc", "--filter", "p.json"])

    assert process.call_args.kwargs["area"] == AREA


def test_process_cli_rejects_inverted_bbox(monkeypatch):
    monkeypatch.setattr(cli, "load_metafilter_parameters", Mock(return_value={"rules": {}}))
    with pytest.raises(SystemExit) as exc_info:
        cli.process_era5_main(["input.nc", "--filter", "p.json", "--bbox", "3", "48", "2", "49"])
    assert exc_info.value.code == 2
