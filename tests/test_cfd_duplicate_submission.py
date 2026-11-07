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

    # A failed case may be retried with the same CAD and settings.
    main.job_store.set_status(first.json()["job_id"], "failed")
    monkeypatch.setattr(main, "_last_submission_time", 0.0)
    third = submit()
    assert third.status_code == 200
    assert third.json()["job_id"] != first.json()["job_id"]
    assert len(queued) == 3
