from pathlib import Path
from types import SimpleNamespace

import pytest

from luminarycloud.meshing.sizing_strategy import Minimal, TargetCount

from app.luminary_pipeline import (
    CaseConfig,
    MESH_MODE_ADAPTIVE,
    MESH_MODE_FIXED,
    _cfd_mesh_params,
    _fixed_mesh_wall_surfaces,
    _fixed_mesh_wheel_surfaces,
)


def test_mesh_modes_build_distinct_serializable_base_meshes():
    config = CaseConfig(
        cad_path=Path("car.step"), cad_label="car", project_name="test",
        farfield_direction=(-1, 0, 0), farfield_speed=24.59,
        mesh_min_size=0.002, mesh_max_size=0.05,
    )
    bounds_min = (0.0, -0.8, 0.0)
    bounds_max = (5.8, 0.8, 1.2)
    adaptive = _cfd_mesh_params(config, "geometry", bounds_min, bounds_max)
    assert isinstance(adaptive.sizing_strategy, Minimal)
    assert not adaptive.refinement_regions
    assert not adaptive.boundary_layer_params
    assert adaptive._to_proto().mesh_complexity_params.type == adaptive._to_proto().mesh_complexity_params.MIN

    config.mesh_mode = MESH_MODE_FIXED
    fixed = _cfd_mesh_params(
        config, "geometry", bounds_min, bounds_max,
        boundary_layer_surfaces=["car-face", "wheel-face", "floor-face"],
        model_surfaces=["car-face", "wheel-face"],
        wheel_model_surfaces=["wheel-face"],
    )
    assert isinstance(fixed.sizing_strategy, TargetCount)
    assert fixed.sizing_strategy.target_count == 10_000_000
    assert fixed.max_size == 1.0
    assert fixed.add_refinement is False
    assert len(fixed.refinement_regions) == 2
    assert len(fixed.boundary_layer_params) == 1
    assert len(fixed.model_meshing_params) == 2
    assert fixed.model_meshing_params[0].max_size == pytest.approx(0.025)
    assert fixed.model_meshing_params[0].curvature == pytest.approx(4.0)
    assert fixed.model_meshing_params[1].max_size == pytest.approx(0.006)
    assert fixed.model_meshing_params[1].curvature == pytest.approx(3.0)
    proto = fixed._to_proto()
    assert proto.mesh_complexity_params.target_cells == 10_000_000
    assert len(proto.refine_params) == 2
    assert len(proto.bl_params) == 1
    assert len(proto.model_params) == 2
    assert list(proto.model_params[0].surfaces) == ["car-face"]
    assert list(proto.model_params[1].surfaces) == ["wheel-face"]
    assert list(proto.bl_params[0].surfaces) == ["car-face", "wheel-face", "floor-face"]
    assert proto.refine_params[0].WhichOneof("shape") == "cube"
    assert proto.refine_params[1].WhichOneof("shape") == "cube"
    assert proto.refine_params[0].cube.min.x.value < 0
    assert proto.refine_params[0].cube.max.x.value > bounds_max[0]
    assert proto.refine_params[1].cube.min.x.value < 0
    assert proto.refine_params[1].cube.min.x.value < 0
    assert fixed.refinement_regions[1].shape.max.x >= bounds_min[0]


@pytest.mark.parametrize(
    ("direction", "expected_axis"),
    [((-1, 0, 0), "negative_x"), ((-24.59, 3, 0), "positive_y"),
     ((0, 1, 0), "positive_y"), ((1, 0, 0), "positive_x")],
)
def test_wake_region_follows_horizontal_airflow(direction, expected_axis):
    config = CaseConfig(
        cad_path=Path("car.step"), cad_label="car", project_name="test",
        farfield_direction=direction, farfield_speed=24.59,
        mesh_min_size=0.002, mesh_max_size=0.05, mesh_mode=MESH_MODE_FIXED,
    )
    params = _cfd_mesh_params(
        config, "geometry", (0, -0.8, 0), (5.8, 0.8, 1.2),
        ["body", "floor"], model_surfaces=["body"],
    )
    wake = params.refinement_regions[1].shape
    if expected_axis == "negative_x":
        assert wake.min.x < -5.8
    elif expected_axis == "positive_x":
        assert wake.max.x > 11.6
    else:
        assert wake.max.y > 0.8
    assert params._to_proto().refine_params[1].cube.max.y.value == pytest.approx(wake.max.y)


def test_fixed_mesh_selects_geometry_walls_and_floor_not_farfield():
    def surface(name, low, high):
        return SimpleNamespace(
            id=name,
            bbox_min=SimpleNamespace(**dict(zip("xyz", low))),
            bbox_max=SimpleNamespace(**dict(zip("xyz", high))),
        )

    faces = [
        surface("body", (0, -0.8, 0), (5.8, 0.8, 1.2)),
        surface("wheel", (1, -0.7, 0), (1.4, -0.3, 0.4)),
        surface("floor", (-100, -58, -0.001), (48, 58, -0.001)),
        surface("farfield-side", (-100, -58, -0.001), (48, -58, 12)),
    ]
    selected, car_count, floor_count = _fixed_mesh_wall_surfaces(
        faces, (0, -0.8, 0), (5.8, 0.8, 1.2), -0.001
    )
    assert selected == ["body", "wheel", "floor"]
    assert (car_count, floor_count) == (2, 1)

    with pytest.raises(RuntimeError, match="floor surfaces"):
        _fixed_mesh_wall_surfaces(faces[:2], (0, -0.8, 0), (5.8, 0.8, 1.2), -0.001)


def test_wheel_surface_sizing_only_selects_compact_contact_faces():
    def surface(name, low, high):
        return SimpleNamespace(
            id=name,
            bbox_min=SimpleNamespace(**dict(zip("xyz", low))),
            bbox_max=SimpleNamespace(**dict(zip("xyz", high))),
        )

    faces = [
        surface("body", (0, -0.8, 0.01), (5.8, 0.8, 1.2)),
        surface("front-wheel", (1, -0.7, 0), (1.55, -0.2, 0.55)),
        surface("rear-contact", (4, -0.3, 0), (4.6, 0.3, 0.56)),
        surface("small-underside", (2, -0.2, 0), (2.5, 0.2, 0.04)),
        surface("floor", (-100, -58, -0.001), (48, 58, -0.001)),
    ]
    assert _fixed_mesh_wheel_surfaces(
        faces, [face.id for face in faces[:-1]], (0, -0.8, 0),
    ) == ["front-wheel", "rear-contact"]
