from pathlib import Path

import mujoco
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "robot" / "franka_fr3" / "myscene.xml"


@pytest.fixture(scope="session")
def model():
    return mujoco.MjModel.from_xml_path(str(MODEL_PATH))


@pytest.fixture()
def data(model):
    simulation_data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    simulation_data.qpos[:7] = model.key_qpos[key_id][:7]
    simulation_data.ctrl[:7] = simulation_data.qpos[:7]
    mujoco.mj_forward(model, simulation_data)
    return simulation_data
