import argparse
import logging

from simulation.runtime import (
    DEFAULT_SCENE_PATH,
    OBJECT_NAMES,
    RobotRuntime,
    build_object_catalog,
    initialize_home,
)

__all__ = ["build_object_catalog", "initialize_home", "run"]


def run(
    target_name="cube_red",
    use_viewer=False,
    scene_path=DEFAULT_SCENE_PATH,
    place_xy=None,
):
    with RobotRuntime(scene_path, use_viewer=use_viewer) as runtime:
        result = runtime.controller.run_cycle(
            target_body_id=runtime.body_id_for(target_name),
            place_xy=place_xy,
        )
        logging.info("Result: %s", result)
        if runtime.viewer and runtime.viewer.is_running():
            runtime.viewer.wait_until_closed()
        return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one FR3 tabletop pick-and-place cycle."
    )
    parser.add_argument("--target", choices=OBJECT_NAMES, default="cube_red")
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="open the passive MuJoCo viewer (use mjpython on macOS)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = run(target_name=args.target, use_viewer=args.viewer)
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
