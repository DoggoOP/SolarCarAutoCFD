from pathlib import Path
from types import SimpleNamespace

import pytest
from google.protobuf.json_format import ParseDict
from luminarycloud._proto.client.simulation_pb2 import MotionData

from app.luminary_pipeline import LuminaryCFDPipeline, SimulationTemplateBuilder, _floor_height


def _surface_bounds(min_x, max_x, min_y, max_y, min_z=0.0, max_z=0.56):
    return SimpleNamespace(stats=SimpleNamespace(
        min_coord=SimpleNamespace(x=min_x, y=min_y, z=min_z),
        max_coord=SimpleNamespace(x=max_x, y=max_y, z=max_z),
    ))


@pytest.mark.parametrize("rear_x,rear_radius", [(-2.7, 0.28), (-2.45, 0.30)])
def test_wheel_axes_and_rolling_speed_match_mesh_and_floor(rear_x, rear_radius):
    surfaces = {
        "Wheel_FL": _surface_bounds(-0.28, 0.28, 0.51, 0.55),
        "Wheel_FR": _surface_bounds(-0.28, 0.28, -0.49, -0.45),
        "Wheel_Rear": _surface_bounds(
            rear_x - rear_radius, rear_x + rear_radius, 0.01, 0.05,
            0.0, 2 * rear_radius,
        ),
    }
    groups = LuminaryCFDPipeline._infer_wheel_motion_groups(surfaces, list(surfaces))
    assert len(groups) == 2
    assert set(groups[0].surfaces) == {"Wheel_FL", "Wheel_FR"}
    assert groups[0].center == pytest.approx((0.0, 0.0, 0.28))
    assert groups[1].surfaces == ("Wheel_Rear",)
    assert groups[1].center == pytest.approx((rear_x, 0.0, rear_radius))
    assert [group.radius for group in groups] == pytest.approx([0.28, rear_radius])

    template = Path(__file__).resolve().parents[1] / "data/base_simulation_params.json"
    payload = SimulationTemplateBuilder(template).build_payload(
        body_surfaces=["Body"],
        floor_surfaces=["Floor"],
        farfield_surfaces=["Farfield"],
        farfield_vector=(-24.59, 0.0, 0.0),
        farfield_speed=24.59,
        sound_speed=340.29,
        frontal_area=1.5,
        ground_speed=24.59,
        rotating_wheels=True,
        wheel_groups=groups,
    )
    frames = {
        frame.frame_id: frame
        for frame in (ParseDict(item, MotionData()) for item in payload["motionData"])
    }
    assert frames["moving_floor_frame"].motion_translation_velocity.x.value == pytest.approx(-24.59)
    for index, group in enumerate(groups, start=1):
        frame = frames[f"wheel_{index}_frame"]
        assert set(frame.attached_boundaries) == set(group.surfaces)
        assert frame.motion_angular_velocity.y.value == pytest.approx(24.59 / group.radius)
        origin = frame.frame_transforms[0].transform_translation
        assert (origin.x.value, origin.y.value, origin.z.value) == pytest.approx(group.center)


def test_floor_height_has_point_one_millimeter_default_clearance():
    assert _floor_height(0.0, 1.155, 0.0) == pytest.approx(-0.0001)
    assert _floor_height(0.2, 1.355, 0.0) == pytest.approx(0.1999)
