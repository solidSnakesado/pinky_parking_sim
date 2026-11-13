"""P-S: 천장 영상에서 학습한 세그멘테이션(YOLO-seg)으로 로봇을 검출한다.

학습된 모델은 클래스 3개(0 바퀴, 1 라인, 2 몸통)의 인스턴스 마스크를 낸다.
몸통 : 신뢰도가 가장 높은 몸통 인스턴스 1개
바퀴 : 몸통 중심에서 바닥 거리 WHEEL_MAX_DIST 안에 있는 바퀴 중 신뢰도가 가장 높은 2개
       (실물에서 화면 구석의 파란 물체가 바퀴로 잡혀 수백 mm 오차가 나던 문제 대응.
        몸통이 없으면 예전처럼 신뢰도 순 2개)
자세 계산은 parking_geom.pose_from_masks가 한다(P-A와 같음). 따라서 P-A와 P-S의 차이는
마스크를 어떻게 얻는가뿐이다.
"""
import numpy as np
from ultralytics import YOLO

import parking_geom as G

WHEEL, LINE, BODY = 0, 1, 2
CONF = 0.25
WHEEL_MAX_DIST = 0.10      # m, 몸통 중심-바퀴 중심 바닥 거리 상한 (실제 약 0.05~0.06)

_model = None


def load(weights):
    global _model
    _model = YOLO(weights)
    # 예열: 첫 이미지 처리 시간에 모델 시작 시간이 섞이지 않게 함
    _model.predict(np.zeros((G.HEIGHT, G.WIDTH, 3), np.uint8), imgsz=640, verbose=False)


def detect(image_bgr, camera):
    """(바퀴 마스크 목록, 몸통 마스크)를 bool 이미지로 반환. camera는 바퀴 후보를 몸통 근처로 거를 때 쓴다."""
    r = _model.predict(image_bgr, imgsz=640, conf=CONF, retina_masks=True, verbose=False)[0]
    if r.masks is None:
        return [], None
    masks = r.masks.data.cpu().numpy() > 0.5            # (N, 480, 640), 원본 이미지 크기
    cls = r.boxes.cls.cpu().numpy().astype(int)
    conf = r.boxes.conf.cpu().numpy()
    order = np.argsort(-conf)
    bodies = [masks[i] for i in order if cls[i] == BODY]
    body = bodies[0] if bodies else None
    wheels = [masks[i] for i in order if cls[i] == WHEEL]
    if body is not None and camera is not None:
        bc = camera.pixels_to_plane(_centroid(body), G.BODY_Z)[0]
        wheels = [m for m in wheels
                  if np.linalg.norm(camera.pixels_to_plane(_centroid(m), G.WHEEL_Z)[0] - bc) < WHEEL_MAX_DIST]
    return wheels[:2], body


def _centroid(mask):
    ys, xs = np.nonzero(mask)
    return np.array([[xs.mean(), ys.mean()]])
