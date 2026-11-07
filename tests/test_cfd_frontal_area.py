import json
from pathlib import Path
from types import SimpleNamespace

import meshio
import numpy as np
import pytest

from app.frontal_area import projected_frontal_area


def _write_overlapping_triangles(path: Path, scale: float = 1.0) -> None:
    # Two identical projected triangles at different X positions: true silhouette
    # is 0.5 m², while a bounding box is 1 m² and a face sum is 1 m².
    points = np.array([
        [0, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 0, 0], [1, 1, 0], [1, 0, 1],
    ], dtype=float) * scale
    triangles = np.array([[0, 1, 2], [3, 4, 5]])
    meshio.write(path, meshio.Mesh(points, [("triangle", triangles)]), binary=True)


def test_stl_frontal_area_is_union_of_projected_faces(tmp_path):
    cad_path = tmp_path / "car.stl"
    _write_overlapping_triangles(cad_path)
    area = projected_frontal_area(cad_path, (0, 0, 0), (1, 1, 1))
    assert area == pytest.approx(0.5, abs=0.003)


def test_step_projection_uses_tessellated_cad_and_normalizes_mm(tmp_path, monkeypatch):
    cad_path = tmp_path / "car.step"
    cad_path.write_text("STEP placeholder for mocked subprocess")

    def tessellate(command, **kwargs):
        assert Path(command[1]).name == "step_tessellate.py"
        assert command[2] == str(cad_path)
        _write_overlapping_triangles(Path(command[3]), scale=1000)
        Path(command[4]).write_text(json.dumps({
            "min": [0, 0, 0], "max": [1000, 1000, 1000],
        }))
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr("app.frontal_area.subprocess.run", tessellate)
    area = projected_frontal_area(cad_path, (0, 0, 0), (1, 1, 1))
    assert area == pytest.approx(0.5, abs=0.003)


def test_bad_cad_projection_reports_error_instead_of_bbox(tmp_path, monkeypatch):
    cad_path = tmp_path / "car.step"
    cad_path.write_text("bad STEP")
    monkeypatch.setattr("app.frontal_area.subprocess.run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=1, stderr="invalid STEP", stdout=""))
    with pytest.raises(RuntimeError, match="invalid STEP"):
        projected_frontal_area(cad_path, (0, 0, 0), (1, 1, 1))
