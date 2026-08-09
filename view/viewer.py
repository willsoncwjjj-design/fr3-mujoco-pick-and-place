import logging
import time

import mujoco.viewer

LOGGER = logging.getLogger(__name__)


class Viewer:
    """将被动 MuJoCo 查看器与 SimScene.step 同步。"""

    def __init__(
        self,
        model,
        data,
        realtime=True,
        lookat=(0.25, 0.0, 0.9),
        distance=1.6,
        azimuth=135,
        elevation=-25,
    ):
        self.data = data
        self.realtime = realtime
        self._handle = None
        try:
            self._handle = mujoco.viewer.launch_passive(model, data)
            camera = self._handle.cam
            camera.lookat[:] = lookat
            camera.distance = distance
            camera.azimuth = azimuth
            camera.elevation = elevation
        except RuntimeError as error:
            LOGGER.warning("Viewer unavailable; continuing headless: %s", error)

        self._last_sync_wall = time.perf_counter()
        self._last_sync_sim = data.time

    def attach(self, scene):
        original_step = scene.step

        def step_with_viewer():
            original_step()
            self.sync()

        scene.step = step_with_viewer

    def sync(self):
        if not self.is_running():
            return
        if self.realtime:
            simulation_delta = self.data.time - self._last_sync_sim
            wall_delta = time.perf_counter() - self._last_sync_wall
            if simulation_delta > wall_delta:
                time.sleep(simulation_delta - wall_delta)
        self._handle.sync()
        self._last_sync_wall = time.perf_counter()
        self._last_sync_sim = self.data.time

    def is_running(self):
        return self._handle is not None and self._handle.is_running()

    def wait_until_closed(self):
        while self.is_running():
            self._handle.sync()
            time.sleep(0.05)

    def close(self):
        if self.is_running():
            self._handle.close()
