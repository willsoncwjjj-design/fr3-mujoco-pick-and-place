from types import SimpleNamespace

import warehouse_complex_run


class FinalStateStub:
    def destination_for(self, destination_id):
        assert destination_id == "buffer.slot_1"
        return SimpleNamespace(available=True)


def test_complex_run_uses_constraint_planner_and_checks_buffer(monkeypatch):
    report = SimpleNamespace(
        goal_satisfied=True,
        task_result=SimpleNamespace(
            items=[SimpleNamespace(operation="relocate")]
        ),
        final_state=FinalStateStub(),
    )
    received = {}

    def fake_run_observed_batch(**kwargs):
        received.update(kwargs)
        return report

    monkeypatch.setattr(
        warehouse_complex_run,
        "run_observed_batch",
        fake_run_observed_batch,
    )

    result = warehouse_complex_run.run_complex()

    assert received["scene_path"] == (
        "robot/franka_fr3/warehouse_complex_scene.xml"
    )
    assert received["scenario_path"].name == "warehouse_sorting_complex.json"
    assert received["planner_type"].__name__ == "ConstraintAwarePlanner"
    assert result.report is report
    assert result.relocation_performed
    assert result.buffer_cleared
    assert result.goal_satisfied
