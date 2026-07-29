from pathlib import Path

import mujoco


class SimScene:
    """Load and advance a MuJoCo scene."""

    def __init__(self):
        self.model = None
        self.data = None

    def setup(self, xml_path):
        path = Path(xml_path)
        if not path.is_absolute():
            project_root = Path(__file__).resolve().parent.parent
            path = (project_root / path).resolve()

        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        return self.model, self.data

    def step(self):
        mujoco.mj_step(self.model, self.data)
