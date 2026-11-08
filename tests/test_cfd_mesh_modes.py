from pathlib import Path

from luminarycloud.meshing.sizing_strategy import Minimal, TargetCount

from app.luminary_pipeline import (
    CaseConfig,
    MESH_MODE_ADAPTIVE,
    MESH_MODE_FIXED,
    _cfd_mesh_params,
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
    fixed = _cfd_mesh_params(config, "geometry", bounds_min, bounds_max)
    assert isinstance(fixed.sizing_strategy, TargetCount)
    assert fixed.sizing_strategy.target_count == 10_000_000
    assert len(fixed.refinement_regions) == 2
    assert len(fixed.boundary_layer_params) == 1
    proto = fixed._to_proto()
    assert proto.mesh_complexity_params.target_cells == 10_000_000
    assert len(proto.refine_params) == 2
    assert len(proto.bl_params) == 1
    assert fixed.refinement_regions[1].shape.max.x > bounds_min[0]
