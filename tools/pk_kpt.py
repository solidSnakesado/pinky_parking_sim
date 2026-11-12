"""P-K: 천장 영상에서 학습한 키포인트 검출(YOLO-pose)로 로봇 자세를 구한다.

키포인트 4점 (로봇 좌표: x 앞, y 왼쪽, z 위, 미터):
  0 앞끝       (0.042, 0, 0.100)   몸통 윗면 앞 가장자리 중앙
  1 뒤끝       (-0.076, 0, 0.100)  몸통 윗면 뒤 가장자리 중앙
  2 왼쪽 바퀴  (0, +0.047, 0.030)  보이는 타이어 중심 (P-S와 같은 WHEEL_Y_VISIBLE, WHEEL_Z)
  3 오른쪽 바퀴 (0, -0.047, 0.030)
자세 계산: 신뢰도가 KPT_CONF 이상인 점을 각자의 높이로 바닥 평면에 되돌린 뒤, 4점 모델과
신뢰도 가중 2D 강체 정합(최소제곱)으로 x, y, yaw를 한 번에 구한다. 점이 2개 미만이면 실패.
앞끝·뒤끝을 서로 다른 점으로 배우므로 앞/뒤는 모델이 바로 구분한다.
"""
import math

import numpy as np

import parking_geom as G

KPTS = np.array([[0.042, 0.0, G.BODY_Z],
                 [-0.076, 0.0, G.BODY_Z],
                 [0.0, G.WHEEL_Y_VISIBLE, G.WHEEL_Z],
                 [0.0, -G.WHEEL_Y_VISIBLE, G.WHEEL_Z]])
FLIP_IDX = [0, 1, 3, 2]          # 좌우 뒤집기 증강 때 왼쪽·오른쪽 바퀴가 서로 바뀜
CONF = 0.25                      # 로봇(박스) 검출 신뢰도
KPT_CONF = 0.5                   # 자세 계산에 쓰는 점의 최소 신뢰도
_model = None


def load(weights):
    global _model
    from ultralytics import YOLO          # 라벨 생성 스크립트가 이 모듈을 쓸 때는 ultralytics가 필요 없게 함
    _model = YOLO(weights)
    # 예열: 첫 이미지 처리 시간에 모델 시작 시간이 섞이지 않게 함
    _model.predict(np.zeros((G.HEIGHT, G.WIDTH, 3), np.uint8), imgsz=640, verbose=False)


def pose_from_keypoints(camera, kp, min_conf=KPT_CONF):
    """kp: (4, 3) 픽셀 x, y, 신뢰도. 반환 (x, y, yaw, 사용한 점 수) 또는 None."""
    use = [j for j in range(len(KPTS)) if kp[j, 2] >= min_conf]
    if len(use) < 2:
        return None
    world = np.array([camera.pixels_to_plane(kp[j, :2], KPTS[j, 2])[0] for j in use])
    model = KPTS[use, :2]
    w = kp[use, 2]
    w = w / w.sum()
    pc, mc = w @ world, w @ model                       # 가중 중심
    p, m = world - pc, model - mc
    s = np.sum(w * (m[:, 0] * p[:, 1] - m[:, 1] * p[:, 0]))
    c = np.sum(w * (m[:, 0] * p[:, 0] + m[:, 1] * p[:, 1]))
    yaw = math.atan2(s, c)                              # world ≈ R(yaw) model + t
    t = pc - G.rot(yaw) @ mc                            # t = base_link 위치 (모델 원점)
    return t[0], t[1], G.wrap(yaw), len(use)


def detect(image_bgr, camera):
    """(자세 또는 None, 키포인트 (4, 3) 또는 None). 신뢰도가 가장 높은 로봇 하나만 씀."""
    r = _model.predict(image_bgr, imgsz=640, conf=CONF, verbose=False)[0]
    if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
        return None, None
    i = int(r.boxes.conf.argmax())
    kp = r.keypoints.data[i].cpu().numpy()             # (4, 3): x, y, 신뢰도
    return pose_from_keypoints(camera, kp), kp
