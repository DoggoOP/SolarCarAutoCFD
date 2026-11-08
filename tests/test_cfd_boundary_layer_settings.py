from pathlib import Path

import pytest
from google.protobuf.json_format import ParseDict
from luminarycloud._proto.client.simulation_pb2 import AdaptiveMeshRefinement

from app.luminary_pipeline import (
    MESH_MODE_ADAPTIVE, MESH_MODE_FIXED, SimulationTemplateBuilder, WheelMotionGroup,
)


@pytest.mark.parametrize("mesh_mode", [MESH_MODE_ADAPTIVE, MESH_MODE_FIXED])
@pytest.mark.parametrize("rotating_wheels", [False, True])
def test_manual_adaptation_boundary_layer_uses_only_walls(mesh_mode, rotating_wheels):

    template = Path(__file__).resolve().parents[1] / "data/base_simulation_params.json"
    builder = SimulationTemplateBuilder(template)
    payload = builder.build_payload(
        body_surfaces=["Body86"],
        floor_surfaces=["Surface 314"],
        farfield_surfaces=["farfield"],
        farfield_vector=(-24.59, 3.0, 0.0),
        farfield_speed=24.77,
        sound_speed=340.29,
        frontal_area=1.0,
        rotating_wheels=rotating_wheels,
        wheel_groups=[
            WheelMotionGroup(("Wheel_FL", "Wheel_FR"), (0.0, 0.0, 0.28), 0.28),
            WheelMotionGroup(("Wheel_Rear",), (-2.7, 0.0, 0.28), 0.28),
        ] if rotating_wheels else None,
        mesh_mode=mesh_mode,
    )

    amr = payload["adaptiveMeshRefinement"]
    # Parse through the SDK's protobuf schema, as create_simulation_template does.
    parsed = ParseDict(amr, AdaptiveMeshRefinement())
    assert amr["meshingMethod"] == ("MESH_METHOD_AUTO" if mesh_mode == MESH_MODE_ADAPTIVE else "MESH_METHOD_MANUAL")
    if mesh_mode == MESH_MODE_ADAPTIVE:
        assert parsed.target_cv_millions.value == 10
    else:
        assert "targetCvMillions" not in amr
    assert amr["allTet"] == "ALL_TET_OFF"
    assert len(parsed.boundary_layer_profile) == 1
    profile = parsed.boundary_layer_profile[0]
    assert profile.n_layers.value == 30
    assert profile.initial_size.value == pytest.approx(1e-5)
    assert profile.growth_rate.value == pytest.approx(1.2)
    expected = {"Body86", "Surface 314"}
    if rotating_wheels:
        expected.update({"Wheel_FL", "Wheel_FR", "Wheel_Rear"})
    assert set(profile.surfaces) == expected
    assert "farfield" not in profile.surfaces
