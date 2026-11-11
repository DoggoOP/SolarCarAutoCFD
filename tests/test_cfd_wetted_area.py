"""CAD area measurements used for wetted area and CdW."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.wetted_area import _triangles_area, cad_wetted_area


def test_step_surface_area_is_normalized_from_mm_to_m(tmp_path: Path) -> None:
    gmsh = pytest.importorskip("gmsh")
    step_path = tmp_path / "car.step"
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("car_box")
        gmsh.model.occ.addBox(0, 0, 0, 2000, 1000, 1000)
        gmsh.model.occ.synchronize()
        gmsh.write(str(step_path))
    finally:
        gmsh.finalize()

    # A 2 m × 1 m × 1 m box has 10 m² of boundary surface.
    assert cad_wetted_area(step_path, (0, 0, 0), (2, 1, 1)) == pytest.approx(10, rel=1e-5)


def test_triangle_area_sums_surface_faces_without_projection() -> None:
    points = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0],
        [0, 0, 1],
    ], dtype=float)
    triangles = np.array([[0, 1, 2], [0, 1, 3]])
    assert _triangles_area(points, triangles) == pytest.approx(1.0)


def test_stl_surface_area_uses_triangles_and_unit_scale(monkeypatch, tmp_path: Path) -> None:
    mesh = SimpleNamespace(
        points=np.array([
            [0, 0, 0], [1000, 0, 0], [0, 1000, 0], [0, 0, 1000],
        ], dtype=float),
        cells=[SimpleNamespace(type="triangle", data=np.array([[0, 1, 2], [0, 1, 3]]))],
    )
    monkeypatch.setitem(sys.modules, "meshio", SimpleNamespace(read=lambda _: mesh))
    assert cad_wetted_area(tmp_path / "car.stl", (0, 0, 0), (1, 1, 1)) == pytest.approx(1.0)


def test_touching_step_solids_do_not_double_count_shared_face(tmp_path: Path) -> None:
    gmsh = pytest.importorskip("gmsh")
    step_path = tmp_path / "two_bodies.step"
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("two_bodies")
        gmsh.model.occ.addBox(0, 0, 0, 1000, 1000, 1000)
        gmsh.model.occ.addBox(1000, 0, 0, 1000, 1000, 1000)
        gmsh.model.occ.synchronize()
        gmsh.write(str(step_path))
    finally:
        gmsh.finalize()

    # The shared 1 m² interface disappears in the exterior union.
    assert cad_wetted_area(step_path, (0, 0, 0), (2, 1, 1)) == pytest.approx(10, rel=1e-5)


def test_sealed_step_cavity_does_not_count_as_wetted(tmp_path: Path) -> None:
    gmsh = pytest.importorskip("gmsh")
    step_path = tmp_path / "sealed_cavity.step"
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("sealed_cavity")
        exterior = gmsh.model.occ.addBox(0, 0, 0, 2000, 1000, 1000)
        cavity = gmsh.model.occ.addBox(500, 250, 250, 500, 500, 500)
        gmsh.model.occ.cut([(3, exterior)], [(3, cavity)])
        gmsh.model.occ.synchronize()
        gmsh.write(str(step_path))
    finally:
        gmsh.finalize()

    # The internal cavity adds 1.5 m² of CAD faces, but has no fluid contact.
    assert cad_wetted_area(step_path, (0, 0, 0), (2, 1, 1)) == pytest.approx(10, rel=1e-5)


def test_step_area_rejects_geometry_with_inconsistent_units(tmp_path: Path) -> None:
    gmsh = pytest.importorskip("gmsh")
    step_path = tmp_path / "car.step"
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("car_box")
        gmsh.model.occ.addBox(0, 0, 0, 2000, 1000, 1000)
        gmsh.model.occ.synchronize()
        gmsh.write(str(step_path))
    finally:
        gmsh.finalize()

    with pytest.raises(ValueError, match="bounds disagree"):
        cad_wetted_area(step_path, (0, 0, 0), (2, 2, 1))
