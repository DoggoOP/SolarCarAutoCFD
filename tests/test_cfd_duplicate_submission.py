import importlib
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.job_store import JobStore


def test_identical_active_cfd_submission_reuses_job(monkeypatch, tmp_path):
    monkeypatch.setenv("LUMINARY_API_KEY", "test-only")
    main = importlib.import_module("app.main")
    monkeypatch.setattr(main, "job_store", JobStore())
    monkeypatch.setattr(main.settings, "uploads_dir", tmp_path)
    monkeypatch.setattr(main, "_last_submission_time", 0.0)
    queued = []
    monkeypatch.setattr(main, "executor", SimpleNamespace(submit=queued.append))

    client = TestClient(main.app)
    def submit():
        return client.post(
            "/run",
            data={"cad_label": "90Deg/6_Bugatti"},
            files={"cad_file": ("car.step", b"same CAD bytes", "application/octet-stream")},
        )

    first = submit()
    second = submit()
    assert first.status_code == second.status_code == 200
    assert second.json() == {"job_id": first.json()["job_id"], "duplicate": True}
    assert len(queued) == 1
    assert len(list(tmp_path.iterdir())) == 1

    # The same geometry with a different wind setting is a distinct scenario.
    monkeypatch.setattr(main, "_last_submission_time", 0.0)
    different = client.post(
        "/run",
        data={"cad_label": "90Deg/6_Bugatti", "wind_direction": "0,1,0"},
        files={"cad_file": ("car.step", b"same CAD bytes", "application/octet-stream")},
    )
    assert different.status_code == 200
    assert different.json()["job_id"] != first.json()["job_id"]
    assert len(queued) == 2

    # Explicit adaptation must create a distinct run from the fixed default.
    monkeypatch.setattr(main, "_last_submission_time", 0.0)
    adaptive = client.post(
        "/run",
        data={"cad_label": "90Deg/6_Bugatti", "mesh_mode": "LUMI_ADAPTATION"},
        files={"cad_file": ("car.step", b"same CAD bytes", "application/octet-stream")},
    )
    assert adaptive.status_code == 200
    assert adaptive.json()["job_id"] != first.json()["job_id"]
    assert len(queued) == 3

    # Iteration caps change the CFD setup, so they must not deduplicate a run.
    monkeypatch.setattr(main, "_last_submission_time", 0.0)
    extended = client.post(
        "/run",
        data={"cad_label": "90Deg/6_Bugatti", "max_iterations": "20000"},
        files={"cad_file": ("car.step", b"same CAD bytes", "application/octet-stream")},
    )
    assert extended.status_code == 200
    assert extended.json()["job_id"] != first.json()["job_id"]
    assert len(queued) == 4
    received = []

    class FakePipeline:
        def __init__(self, settings):
            pass

        def run_case(self, config, callback, check_cancelled):
            received.append((config.mesh_mode, config.max_iterations))
            return {}

    monkeypatch.setattr(main, "LuminaryCFDPipeline", FakePipeline)
    queued[-1]()
    assert received == [("FIXED_MANUAL", 20000)]

    invalid = client.post(
        "/run",
        data={"cad_label": "90Deg/6_Bugatti", "mesh_mode": "unsupported"},
        files={"cad_file": ("car.step", b"same CAD bytes", "application/octet-stream")},
    )
    assert invalid.status_code == 400
    for invalid_limit in ("0", "-5", "1.5", "banana"):
        response = client.post(
            "/run",
            data={"cad_label": "90Deg/6_Bugatti", "max_iterations": invalid_limit},
            files={"cad_file": ("car.step", b"same CAD bytes", "application/octet-stream")},
        )
        assert response.status_code == 400

    # A failed case may be retried with the same CAD and settings.
    main.job_store.set_status(first.json()["job_id"], "failed")
    monkeypatch.setattr(main, "_last_submission_time", 0.0)
    third = submit()
    assert third.status_code == 200
    assert third.json()["job_id"] != first.json()["job_id"]
    assert len(queued) == 5
