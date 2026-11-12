#!/usr/bin/env python3
"""실물 천장 웹캠을 시뮬 카메라와 같은 위치·방향으로 맞추기 위한 실시간 정렬 뷰어.

- 실시간 영상(640x480) 위에, 보정한 시뮬 카메라(gen_parking_dataset의 CAM_POS / CAM_QUAT,
  parking_geom의 HFOV·k1)로 투영한 테이프 외곽선을 초록 선으로 겹쳐 그린다.
  실물 테이프가 초록 선과 겹칠 때까지 카메라를 움직이면 시뮬과 같은 자세가 된다.
- 칸 중심(A~D)과 출발점은 노란 십자로 표시한다.
- 정렬 참고값: 투영 외곽선 점에서 가장 가까운 어두운 픽셀(실물 테이프)까지의 거리(px) 중앙값·p90.
  잘 맞으면 1~2 px 정도로 작아진다. 조명에 따라 --dark 값을 조정한다.
- 키: o = 시뮬 화면(--ref) 반투명 겹치기 켜기/끄기, l = 선 켜기/끄기, m = 어두운 픽셀 마스크 보기,
      s = 현재 프레임 저장(원본 + 겹친 화면), q / Esc = 종료.

사용법 (GUI가 되는 cv2 필요, 예: pinky_train 가상환경):
  python tools/camera_align_viewer.py --device /dev/video2
  python tools/camera_align_viewer.py --device /dev/video2 --calib config/ceiling_camera_calib.json
    --calib: calib_from_tape.py로 구한 실물 보정값으로 선을 그림 (보정 결과 확인용).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parking_geom as G  # noqa: E402

# 보정한 시뮬 카메라 자세 (gen_parking_dataset.py와 같은 값; mujoco 없이 실행되도록 여기 적음)
CAM_POS = [-0.2894, 0.5706, 1.815]
CAM_QUAT = [0.69746, 0.15253, -0.16265, -0.68105]
SLOT_U = (136.5, 345.5, 554.5, 763.5)
SLOT_V = 120.0                                  # 칸 중심 표시 높이 (칸 v 19~220의 가운데)


def put(img, text, y, color=(255, 255, 255)):
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='/dev/video2', help='camera device path or index')
    ap.add_argument('--ref', default=None, help='sim reference image to blend (e.g. ceiling_view.png)')
    ap.add_argument('--dark', type=int, default=80, help='gray level below which a pixel counts as tape')
    ap.add_argument('--out', default='results/camera_align', help='folder for saved frames')
    ap.add_argument('--calib', default=None, help='calibration json from calib_from_tape.py (default: sim camera)')
    args = ap.parse_args()

    dev = int(args.device) if args.device.isdigit() else args.device
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, G.WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, G.HEIGHT)
    if not cap.isOpened():
        raise SystemExit(f'cannot open camera {args.device}')

    if args.calib:                                  # 실물 보정값으로 그림
        c = json.load(open(args.calib))
        cam = G.Camera(c['pos'], c['quat_wxyz'])
        print(f"using calibration {args.calib} (residual median {c.get('residual_px_median')} px)")
    else:                                           # 시뮬 카메라 (학습 기준 자세)
        cam = G.Camera(CAM_POS, CAM_QUAT)
    polys = [np.round(p).astype(np.int32) for p in G.tape_polygons(cam)]
    outline = np.vstack(polys)
    inside = (outline[:, 0] >= 0) & (outline[:, 0] < G.WIDTH) & (outline[:, 1] >= 0) & (outline[:, 1] < G.HEIGHT)
    outline = outline[inside]                       # 화면 안의 외곽선 점만 정렬 참고값에 씀
    marks = [G.mm_to_world(u, SLOT_V) for u in SLOT_U] + [G.mm_to_world(G.START_U, G.START_V)]
    marks = np.round(cam.project([[x, y, 0.0] for x, y in marks])).astype(int)
    ref = None
    if args.ref:
        ref = cv2.imread(args.ref)
        if ref is None:
            raise SystemExit(f'cannot read {args.ref}')
        ref = cv2.resize(ref, (G.WIDTH, G.HEIGHT))
    show_ref, show_lines, show_mask = ref is not None, True, False
    out = Path(args.out)
    t_prev, fps, warned = time.perf_counter(), 0.0, False

    while True:
        ok, frame = cap.read()
        if not ok:
            print('frame grab failed')
            break
        h, w = frame.shape[:2]
        if (w, h) != (G.WIDTH, G.HEIGHT):            # 보정 해상도와 다르면 맞춰서 표시 (경고 1회)
            if not warned:
                print(f'warning: camera gives {w}x{h}, resized to {G.WIDTH}x{G.HEIGHT}')
                warned = True
            frame = cv2.resize(frame, (G.WIDTH, G.HEIGHT))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dark = (gray < args.dark).astype(np.uint8)
        # 어두운 픽셀까지의 거리 지도: 외곽선 점에서 값을 읽어 정렬 정도를 본다
        dist = cv2.distanceTransform(1 - dark, cv2.DIST_L2, 3)
        d = dist[outline[:, 1], outline[:, 0]] if len(outline) else np.array([np.nan])

        vis = cv2.cvtColor(dark * 255, cv2.COLOR_GRAY2BGR) if show_mask else frame.copy()
        if show_ref:
            vis = cv2.addWeighted(vis, 0.5, ref, 0.5, 0)
        if show_lines:
            cv2.polylines(vis, polys, True, (0, 255, 0), 1, cv2.LINE_AA)
            for u, v in marks:
                cv2.drawMarker(vis, (int(u), int(v)), (0, 255, 255), cv2.MARKER_CROSS, 14, 1)
        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 / max(1e-6, now - t_prev)
        t_prev = now
        put(vis, f'{w}x{h}  {fps:4.1f} fps   line->tape px: median {np.median(d):.1f}  p90 {np.percentile(d, 90):.1f}', 18)
        put(vis, 'o: sim overlay  l: lines  m: dark mask  s: save  q: quit', G.HEIGHT - 10, (200, 200, 200))
        cv2.imshow('camera align', vis)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord('o') and ref is not None:
            show_ref = not show_ref
        elif key == ord('l'):
            show_lines = not show_lines
        elif key == ord('m'):
            show_mask = not show_mask
        elif key == ord('s'):
            out.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime('%Y%m%d_%H%M%S')
            cv2.imwrite(str(out / f'raw_{stamp}.png'), frame)
            cv2.imwrite(str(out / f'overlay_{stamp}.png'), vis)
            print(f'saved {out}/raw_{stamp}.png  line->tape median {np.median(d):.1f} px')
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
