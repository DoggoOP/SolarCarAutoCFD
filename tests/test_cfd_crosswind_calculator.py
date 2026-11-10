import importlib
import math
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.job_store import JobStore


def test_crosswind_calculator_combines_driving_and_lateral_speeds(monkeypatch):
    monkeypatch.setenv("LUMINARY_API_KEY", "test-only")
    main = importlib.import_module("app.main")
    speed, direction = main._crosswind_apparent_wind(24.59, "3", "+y")
    assert speed == pytest.approx(math.hypot(24.59, 3))
    assert direction == (-24.59, 3.0, 0.0)
    speed, direction = main._crosswind_apparent_wind(24.59, "0", "-y")
    assert speed == pytest.approx(24.59)
    assert direction == (-24.59, 0.0, 0.0)
    _, direction = main._crosswind_apparent_wind(30, "4", "-y")
    assert direction == (-30, -4.0, 0.0)


def test_crosswind_form_submits_exact_values_and_presets_remain_manual(monkeypatch, tmp_path):
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
    assert 'name="crosswind_magnitude"' in html
    assert 'name="ground_speed"' in html
    assert 'name="wind_source" type="hidden" value="crosswind"' in html
    assert 'name="crosswind_magnitude" type="number" min="0" step="any" value="0"' in html

    def submit(data):
        monkeypatch.setattr(main, "_last_submission_time", 0.0)
        return client.post(
            "/run", data={"cad_label": "crosswind", **data},
            files={"cad_file": ("car.step", b"geometry", "application/octet-stream")},
        )

    result = submit({
        "wind_source": "crosswind", "ground_speed": "24.59",
        "crosswind_magnitude": "3", "crosswind_flow_direction": "+y",
        "farfield_speed": "999", "wind_direction": "0,1,0",
    })
    assert result.status_code == 200
    queued.pop(0)()
    assert received[-1].farfield_direction == (-24.59, 3.0, 0.0)
    assert received[-1].farfield_speed == pytest.approx(math.hypot(24.59, 3))
    assert received[-1].ground_speed == 24.59

    result = submit({
        "wind_source": "manual", "ground_speed": "24.59",
        "farfield_speed": "27.38", "wind_direction": "-27.36,1.15,0",
    })
    assert result.status_code == 200
    queued.pop(0)()
    assert received[-1].farfield_direction == (-27.36, 1.15, 0.0)
    assert received[-1].farfield_speed == 27.38

    for data in (
        {"wind_source": "crosswind", "crosswind_magnitude": "-1"},
        {"wind_source": "crosswind", "crosswind_magnitude": "nan"},
        {"wind_source": "crosswind", "crosswind_flow_direction": "sideways"},
        {"wind_source": "crosswind", "ground_speed": "0"},
    ):
        assert submit(data).status_code == 400
