import numpy as np


class Detector:
    """Simulation segmentation detector with a reserved real-detector adapter."""

    SIM_MODE = "simulation"
    REAL_MODE = "real"

    def __init__(self, model, mode=SIM_MODE, real_detector=None):
        self.model = model
        self.mode = mode
        self.real_detector = real_detector

    def detect(self, image, object_catalog=None):
        if self.mode == self.SIM_MODE:
            return self.detect_sim(image, object_catalog)
        return self.detect_real(image)

    def detect_sim(self, seg_mask, object_catalog):
        catalog_ids = {
            obj["body_id"] for obj in object_catalog
        } if object_catalog else set()
        detections = []

        # Segmentation channel 0 contains geom IDs, not body IDs.
        for geom_id in np.unique(seg_mask[..., 0]).astype(int):
            if not 0 <= geom_id < self.model.ngeom:
                continue
            body_id = int(self.model.geom_bodyid[geom_id])
            if body_id <= 0 or body_id not in catalog_ids:
                continue

            mask = seg_mask[..., 0] == geom_id
            ys, xs = np.where(mask)
            if not len(xs):
                continue
            detections.append(
                {
                    "class_name": self.get_class_name(body_id, object_catalog),
                    "bbox": (
                        int(xs.min()),
                        int(ys.min()),
                        int(xs.max()),
                        int(ys.max()),
                    ),
                    "body_id": body_id,
                    "center": (int(xs.mean()), int(ys.mean())),
                    "score": 1.0,
                }
            )
        return detections

    def detect_real(self, rgb):
        if self.real_detector is None:
            raise NotImplementedError("A real detector has not been configured")
        return self.real_detector(rgb)

    @staticmethod
    def get_class_name(body_id, object_catalog):
        for obj in object_catalog or []:
            if obj["body_id"] == body_id:
                return obj["class_name"]
        return "unknown"
