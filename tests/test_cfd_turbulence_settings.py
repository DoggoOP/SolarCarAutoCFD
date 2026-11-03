from pathlib import Path

import pytest

from app.luminary_pipeline import SimulationTemplateBuilder


@pytest.mark.parametrize(
    ("model", "expected_ratio"),
    [
        ("KOMEGA_SST", 5.0),
        ("SPALART_ALLMARAS", 0.21043825715555026),
    ],
)
def test_generated_farfield_uses_model_specific_viscosity_ratio(model, expected_ratio):
    template = Path(__file__).resolve().parents[1] / "data/base_simulation_params.json"
    builder = SimulationTemplateBuilder(template)
    payload = builder.build_payload(
        body_surfaces=["body"],
        floor_surfaces=["floor"],
        farfield_surfaces=["farfield"],
        farfield_vector=(-24.59, 3.0, 0.0),
        farfield_speed=24.77,
        sound_speed=340.29,
        frontal_area=1.0,
        turbulence_model=model,
    )

    fluid = payload["physics"][0]["fluid"]
    farfield = next(
        bc for bc in fluid["boundaryConditionsFluid"]
        if bc["physicalBoundary"] == "FARFIELD"
    )
    assert fluid["turbulence"]["turbulenceModel"] == model
    assert farfield["turbulentViscosityRatio"]["value"] == expected_ratio
    assert fluid["initializationFluid"]["initTurbulentViscosityRatio"]["value"] == expected_ratio
    assert farfield["turbulenceSpecificationKomega"] == "BC_TURBULENT_VISCOSITY_RATIO_AND_INTENSITY_KOMEGA"
