from pathlib import Path

import mujoco
import pytest

from simulation.runtime import RobotRuntime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENE_PATH = PROJECT_ROOT / "robot" / "franka_fr3" / "warehouse_scene.xml"


@pytest.mark.integration
def test_runtime_assembles_controller_without_advancing_simulation():
    runtime = RobotRuntime(SCENE_PATH, settle_steps=0)

    with runtime as active:
        assert active is runtime
        assert runtime.is_open
        assert runtime.data.time == 0
        assert runtime.controller is not None
        assert runtime.body_id_for("cube_red") == _body_id(
            runtime.model, "cube_red"
        )
        assert runtime.body_id_for("cube_blue") == _body_id(
            runtime.model, "cube_blue"
        )
        runtime.set_free_body_xy("cube_red", (0.31, 0.16))
        assert tuple(runtime.data.body("cube_red").xpos[:2]) == pytest.approx(
            (0.31, 0.16)
        )

    assert not runtime.is_open
    assert runtime.camera is None
    assert runtime.viewer is None
    assert runtime.controller is None
    with pytest.raises(RuntimeError, match="RobotRuntime is not open"):
        runtime.body_id_for("cube_red")
    runtime.close()


def _body_id(model, body_name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
