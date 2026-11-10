import importlib
import math
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.job_store import JobStore


def test_clock_wind_comes_from_the_right_side(monkeypatch):
    monkeypatch.setenv("LUMINARY_API_KEY", "test-only")
    main = importlib.import_module("app.main")

    speed, direction = main._wind_from_clock(24.59, "0", "90")
    assert speed == pytest.approx(24.59)
    assert direction == pytest.approx((-24.59, 0, 0))

    speed, direction = main._wind_from_clock(24.59, "3", "90")
    assert speed == pytest.approx(math.hypot(24.59, 3))
    assert direction == pytest.approx((-24.59, 3, 0))

    speed, direction = main._wind_from_clock(24.59, "6", "30")
    assert direction == pytest.approx((-24.59 - 6 * math.cos(math.pi / 6), 3, 0))
    assert speed == pytest.approx(math.hypot(direction[0], direction[1]))

    speed, direction = main._wind_from_clock(24.59, "6", "0")
    assert direction == pytest.approx((-30.59, 0, 0))
    assert speed == pytest.approx(30.59)

    speed, direction = main._wind_from_clock(24.59, "6", "180")
    assert direction == pytest.approx((-18.59, 0, 0))
    assert speed == pytest.approx(18.59)


def test_form_uses_only_raw_wind_inputs(monkeypatch, tmp_path):
    monkeypatch.setenv("LUMINARY_API_KEY", "test-only")
    main = importlib.import_module("app.main")
    monkeypatch.setattr(main, "job_store", JobStore())
    monkeypatch.setattr(main.settings, "uploads_dir", tmp_path)
    monkeypatch.setattr(main, "_last_submission_time", 0.0)
    queued = []
    received = []
    monkeypatch.setattr(main, "executor", SimpleNamespace(submit=queued.append))

    class FakePipeline:
        def __init__(self, settings):
            pass

        def run_case(self, config, callback, check_cancelled):
            received.append(config)
            return {}

    monkeypatch.setattr(main, "LuminaryCFDPipeline", FakePipeline)
    client = TestClient(main.app)
    html = client.get("/").text
    assert 'name="raw_wind_speed"' in html
    assert 'name="raw_wind_angle"' in html
    assert 'name="ground_speed"' in html
    assert 'name="raw_wind_speed" type="number" min="0" step="any" value="0"' in html
    assert 'name="raw_wind_angle" type="number" min="0" max="180" step="any" value="90"' in html
    assert 'name="farfield_speed"' not in html
    assert 'name="wind_direction"' not in html
    assert 'name="crosswind_flow_direction"' not in html

    def submit(data):
        monkeypatch.setattr(main, "_last_submission_time", 0.0)
        return client.post(
            "/run", data={"cad_label": "clock-wind", **data},
            files={"cad_file": ("car.step", b"geometry", "application/octet-stream")},
        )

    result = submit({
        "ground_speed": "24.59", "raw_wind_speed": "3", "raw_wind_angle": "90",
        "farfield_speed": "999", "wind_direction": "0,-1,0",
    })
    assert result.status_code == 200
    queued.pop(0)()
    assert received[-1].farfield_direction == pytest.approx((-24.59, 3, 0))
    assert received[-1].farfield_speed == pytest.approx(math.hypot(24.59, 3))
    assert received[-1].ground_speed == 24.59

    for data in (
        {"raw_wind_speed": "-1", "raw_wind_angle": "90"},
        {"raw_wind_speed": "nan", "raw_wind_angle": "90"},
        {"raw_wind_speed": "3", "raw_wind_angle": "-1"},
        {"raw_wind_speed": "3", "raw_wind_angle": "181"},
        {"raw_wind_speed": "3", "raw_wind_angle": "nan"},
        {"raw_wind_speed": "24.59", "raw_wind_angle": "180"},
        {"raw_wind_speed": "0", "raw_wind_angle": "90", "ground_speed": "0"},
    ):
        assert submit(data).status_code == 400
