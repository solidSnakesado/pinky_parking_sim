"""P-S: learned segmentation (YOLO-seg) for robot detection on the ceiling image.

The trained model gives instance masks for 3 classes: 0 wheel, 1 line, 2 body.
wheels : the 2 most confident wheel instances
body   : the most confident body instance
The pose itself is computed by parking_geom.pose_from_masks (same as P-A), so only the
mask source differs between P-A and P-S.
"""
import numpy as np
from ultralytics import YOLO

import parking_geom as G

WHEEL, LINE, BODY = 0, 1, 2
CONF = 0.25
_model = None


def load(weights):
    global _model
    _model = YOLO(weights)
    # warm-up so the first image is not counted with model start-up time
    _model.predict(np.zeros((G.HEIGHT, G.WIDTH, 3), np.uint8), imgsz=640, verbose=False)


def detect(image_bgr, camera):
    """Returns (wheel_masks, body_mask) as boolean images. camera is unused (same signature as P-A)."""
    r = _model.predict(image_bgr, imgsz=640, conf=CONF, retina_masks=True, verbose=False)[0]
    if r.masks is None:
        return [], None
    masks = r.masks.data.cpu().numpy() > 0.5            # (N, 480, 640) at the original image size
    cls = r.boxes.cls.cpu().numpy().astype(int)
    conf = r.boxes.conf.cpu().numpy()
    order = np.argsort(-conf)
    wheels = [masks[i] for i in order if cls[i] == WHEEL][:2]
    bodies = [masks[i] for i in order if cls[i] == BODY]
    return wheels, (bodies[0] if bodies else None)
