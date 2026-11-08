from types import SimpleNamespace

import pytest

from app.luminary_pipeline import (
    DRAG_SETTLING_AVERAGING_ITERATIONS,
    DRAG_SETTLING_START_ITERATION,
    DRAG_SETTLING_THRESHOLD_FRACTION,
    DRAG_SETTLING_THRESHOLD_PERCENT,
    DRAG_SETTLING_WINDOW_ITERATIONS,
    LuminaryCFDPipeline,
    MAX_CFD_ITERATIONS,
)


class FakeTemplate:
    def __init__(self, fail_drag=False):
        self.fail_drag = fail_drag
        self.general = None
        self.outputs = []
        self.conditions = []

    def update_general_stopping_conditions(self, **kwargs):
        self.general = kwargs

    def create_output_definition(self, definition):
        if definition.name == "Drag (Fx)" and self.fail_drag:
            raise RuntimeError("drag output unavailable")
        result = SimpleNamespace(id=f"output-{len(self.outputs)}")
        self.outputs.append((definition, result))
        return result

    def create_or_update_stopping_condition(self, **kwargs):
        self.conditions.append(SimpleNamespace(**kwargs))

    def list_stopping_conditions(self):
        return self.conditions


def test_only_drag_can_stop_before_iteration_limit():
    template = FakeTemplate()
    pipeline = object.__new__(LuminaryCFDPipeline)
    maximum, area_created = pipeline._setup_stopping_conditions(
        template, "physics", ["body", "wheel"], lambda _: None
    )

    assert (maximum, area_created) == (MAX_CFD_ITERATIONS, True)
    assert template.general == {"max_iterations": 15000, "stop_on_any": False}
    assert len(template.conditions) == 1
    drag_definition, drag_output = next(
        item for item in template.outputs if item[0].name == "Drag (Fx)"
    )
    assert drag_definition.surfaces == ["body", "wheel"]
    condition = template.conditions[0]
    assert condition.output_definition_id == drag_output.id
    assert DRAG_SETTLING_THRESHOLD_PERCENT == 0.5
    assert condition.threshold == DRAG_SETTLING_THRESHOLD_FRACTION == 0.005
    assert condition.start_at_iteration == DRAG_SETTLING_START_ITERATION == 8000
    assert condition.averaging_iterations == DRAG_SETTLING_AVERAGING_ITERATIONS == 250
    assert condition.iterations_to_consider == DRAG_SETTLING_WINDOW_ITERATIONS == 1000


def test_missing_drag_monitor_prevents_launch():
    template = FakeTemplate(fail_drag=True)
    pipeline = object.__new__(LuminaryCFDPipeline)
    with pytest.raises(RuntimeError, match="drag convergence output"):
        pipeline._setup_stopping_conditions(template, "physics", ["body"], lambda _: None)
    assert template.conditions == []


@pytest.mark.parametrize(
    ("drag_settled", "hit_limit", "expected"),
    [(True, False, "DRAG_SETTLED"), (False, True, "ITERATION_LIMIT")],
)
def test_reports_drag_settling_separately_from_iteration_limit(
    drag_settled, hit_limit, expected
):
    results = [
        SimpleNamespace(
            output_name="Force (%) - Max Deviation",
            satisfied=drag_settled,
            is_force_condition=False,
        ),
        SimpleNamespace(
            output_name="Iteration Index",
            satisfied=hit_limit,
            value=10000,
            is_force_condition=True,
        ),
    ]
    simulation = SimpleNamespace(
        get_stopping_condition_status=lambda: SimpleNamespace(
            condition_results=results,
            overall_success=drag_settled,
            force_stopped=hit_limit,
        )
    )
    messages = []
    assert LuminaryCFDPipeline._get_convergence_result(simulation, messages.append) == (
        expected,
        10000,
    )
    assert messages
