# P-S 재현 실험 (RTX 5070 Ti 노트북, 2026-11-14)

기본 개발 노트북(P-S 원래 가중치가 있는 유일한 기기)에 주말 동안 접근할 수 없어서, GitHub에 있는 코드와
설정만으로 5070 Ti에서 P-S(YOLO11n-seg)를 다시 만들고 원래 P-S와 비교한 기록이다.
이 브랜치(`exp/ps-repro-5070ti`)는 실험 기록용이며 main에 합치지 않는다.
**결과는 "5070 Ti 재현 P-S"이며, 원래 P-S(기본 노트북 `yolo11n_seed0~2`)의 결과로 쓰지 않는다.**

## 환경

| 항목 | 값 |
|---|---|
| 코드 커밋 | `8818e13` (main) |
| 기기 | RTX 5070 Ti Laptop GPU 12 GB, WSL Ubuntu 24.04, 드라이버 616.64 |
| venv | `~/venv/pinky_seg` (학습 전용, 강화학습용 `pinky_train`과 분리) |
| 패키지 | torch 2.11.0+cu128, ultralytics 8.4.208, mujoco 3.6.0 (torch·ultralytics·mujoco 모두 기본 노트북과 같은 버전) |
| 사전 학습 가중치 | `yolo11n-seg.pt` (ultralytics 8.4.208이 자동으로 내려받은 공식 COCO 가중치. 기본 노트북 파일과 같은지는 해시 미확인) |

## 데이터셋 (`datasets/parking_seg_v2_5070ti`, 이미지는 git에 없음)

```
python tools/gen_parking_dataset.py --count 2000 --seed 0 --out datasets/parking_seg_v2_5070ti            # train 1800 + val 200
python tools/gen_parking_dataset.py --count 300 --split test --seed 100 --out datasets/parking_seg_v2_5070ti
```

- 장수: train 1,800 / val 200 / test 300 (원래와 같음).
- test 300장의 정답 자세를 원래 데이터셋(`results/ps_test.csv`의 `gt_x, gt_y, gt_yaw`)과 비교: 이름·종류 300개 일치,
  **위치 차이 최대 0.000 mm, 방향 차이 최대 0.0000°**. 같은 생성 코드·seed라 자세·카메라 흔들림·조명이 같다.
  남는 차이는 렌더링 픽셀뿐 (GPU·OpenGL 환경 차이).
- `poses.csv`, `data.yaml`을 이 폴더에 함께 남김.

## 학습 (`runs/segment/runs/ps/yolo11n_5070ti_seed0`, 가중치는 git에 없음)

```
yolo segment train model=yolo11n-seg.pt data=datasets/parking_seg_v2_5070ti/data.yaml imgsz=640 batch=8 epochs=80 \
  patience=20 seed=0 deterministic=True project=$PWD/runs/segment/runs/ps name=yolo11n_5070ti_seed0
```

| 항목 | 값 |
|---|---|
| 옵티마이저 | `optimizer=auto` → **AdamW (lr 0.001429, momentum 0.9)**. 인수인계 문서에는 원래 1,800장 학습이 "SGD(lr 0.01)로 추정, 로그로 미확정"이라고 되어 있으나, 같은 ultralytics 버전·장수·설정에서 auto가 AdamW를 골랐으므로 원래도 AdamW였을 가능성이 큼 (원래 `args.yaml`로 확인 필요) |
| 종료 | 71 epoch에서 조기 종료, 최고 epoch 51 (`best.pt`) |
| val (best.pt) | 마스크 mAP50 **0.964**, mAP50-95 0.721 (원래 seed 0/1/2의 마스크 mAP50: 0.963 / 0.962 / 0.966) |
| 가중치 SHA-256 | `bbeb84e71d23dc054f3b4387e17816a9fb8717ecac7380bb8b03720dd3325171` (6.0 MB) |
| 학습 시간 | 약 35분 |

학습 기록: `train_log.txt`(학습 출력 중 val 줄), `args.yaml`, `results.csv`(epoch별 손실·지표).

## 평가 (같은 코드 `8818e13`의 `eval_pose.py --method ps`)

원래 P-S 비교 대상은 같은 코드 기준인 세그멘테이션 쪽 결과 `results/ps_fbfix/ps{0,1,2}_simtest.csv`(시뮬 test),
`results/real40_fbfix/ps{0,1,2}.csv`(실물 test).

### 시뮬 test 300장 (재현은 다시 만든 이미지, 원래는 원래 이미지)

| 모델 | 검출 | 위치 중앙값 | 위치 p95 | 방향 중앙값 | 앞/뒤 |
|---|---|---|---|---|---|
| **재현 5070 Ti** | 100% | **2.23 mm** | **11.2 mm** | **1.60°** | **99.7%** |
| 원래 seed 0 | 100% | 2.11 mm | 10.5 mm | 1.55° | 100% |
| 원래 seed 1 | 100% | 2.20 mm | 10.5 mm | 1.62° | 100% |
| 원래 seed 2 | 100% | 2.11 mm | 10.2 mm | 1.54° | 99.7% |

### 실물 test 49장 (`datasets/real_test`, 같은 사진으로 평가)

| 모델 | 검출 | 위치 중앙값 | 위치 p95 | 방향 중앙값 | 앞/뒤 |
|---|---|---|---|---|---|
| **재현 5070 Ti** | 100% | **10.19 mm** | **27.0 mm** | **2.39°** | **91.8%** |
| 원래 seed 0 | 100% | 10.64 mm | 23.8 mm | 2.54° | 95.9% |
| 원래 seed 1 | 100% | 9.03 mm | 23.3 mm | 2.84° | 91.8% |
| 원래 seed 2 | 100% | 9.47 mm | 28.2 mm | 2.93° | 87.8% |

### 판단

- 실물 test(같은 사진)에서 재현 모델은 원래 seed 3개의 범위 안이며 구분되지 않는다. 방향 중앙값은 범위보다 조금 좋다.
- 시뮬 test에서 위치 p95가 약 0.7 mm 크다. 렌더링 차이 또는 seed 흔들림 수준.
- 원래 모델과 마찬가지로 실물에서 오차가 시뮬의 약 5배 (위치 중앙값 약 2 mm → 약 10 mm). 시뮬→실물 격차는 원래와 같은 수준.
- **결론: 5070 Ti 재현 P-S는 원래 P-S와 통계적으로 같은 수준으로 재현되었다.** 인식 루프 주행 평가에 대신 쓸 수 있다.
  원래 가중치가 생기면 같은 평가(특히 인식 루프)를 원래 가중치로 다시 하고 비교한다.

## 파일

| 파일 | 내용 |
|---|---|
| `ps_repro_simtest.csv` | 시뮬 test 300장 장별 결과 |
| `ps_repro_real40.csv` | 실물 test 49장 장별 결과 |
| `args.yaml`, `results.csv` | 학습 설정·epoch별 기록 (ultralytics 출력) |
| `train_log.txt` | 학습 출력 (val 줄) |
| `data.yaml`, `poses.csv` | 재현 데이터셋 설정과 장별 정답 자세·카메라 자세·조명 |

## 이후 기록 (인식 루프)

(재현 P-S 인식 루프 40판 결과를 여기에 추가)
