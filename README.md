![logo](./assets/logo-256.png)

# SOKKANAEM

**S**patial-temporal **O**ptimized **K**ey-patch **K**ernel for **A**daptive **N**etwork **A**rchitecture and **E**fficient **M**amba

프레임 간 변화가 발생한 패치만 연산하는 실시간 비디오 깊이 추정 프레임워크. 설계 문서는 [IDEA.md](IDEA.md).

## 핵심 아이디어

Mamba 이산화 파라미터 Δ에 변화 마스크를 곱한다 (Δ-gating):
mask=0 이면 `exp(Δ·A)=I, Δ·B=0` 이므로 hidden state가 **정확히** 복사됨 — 연산 스킵 = 상태 유지가 수식 레벨에서 공짜.

## 이동 카메라 확장: Low-Res GMC + Feature-level Gating (IDEA.md §3.5)

고정 카메라를 넘어 블랙박스·드론 등 ego-motion 환경으로 확장하는 하이브리드 파이프라인.
센서(IMU) 없이 순수 RGB만 사용, 전처리 병목 없이 스킵 이득 유지:

1. **Low-Res GMC** (`gmc.py`) — 프레임을 저해상도(기본 128px)로 축소, 소수 특징점(≤50개) +
   LK 추적 + RANSAC 호모그래피로 프레임 t-1을 현재 시점으로 워핑 (~1-2ms). 추적 실패 시 항등 폴백.
   축소 이득은 HD 입력 기준 — PoC 기본 입력(128px)에서는 이미 저해상도라 무축소로 동작.
2. **Feature-level Gating** — 워핑된 t-1과 현재 프레임을 patch embedding(초기 인코더)에 통과,
   패치별 피처 상대 L1 차이로 active mask 확정. 정렬 잔차·조명 노이즈는 피처 레벨에서 흡수.
   임계값 로직(hysteresis/dilation/keyframe)은 기존 detector와 공유.

`SOKKANAEM(gmc=True)` 또는 `infer.py / eval.py --gmc`로 활성화. tau는 피처 스케일
(기본 0.1/0.05, 픽셀 MSE 스케일과 다름). opencv 필요: `uv pip install -e ".[video]"`.

## 설치 (conda + uv)

```bash
conda env create -f environment.yml   # python 3.11 + uv
conda activate sokkanaem
uv pip install -e ".[dev,video]"      # torch, numpy, pytest + opencv (GMC·비디오 입력)
```

## 사용

```bash
python -m pytest tests/ -q      # Δ-gating 정확성 + 데이터 파이프라인 검증

# 데이터셋 구성/검증
python scripts/prepare_data.py check --data scannet:/data/scannet
python scripts/prepare_data.py from-video cam.mp4 --out data/cctv

# 학습 — config가 기본값, CLI 플래그가 우선. 결과물은 work_dirs/<config명>/
python scripts/train.py --config configs/synthetic.toml
python scripts/train.py --config configs/scannet.toml --data scannet:/my/scannet
python scripts/train.py --config configs/mixed.toml   # 멀티 데이터셋 혼합

# 검증/추론 공통: ckpt 옆 config.toml(train.py가 기록한 실효 설정)에서 [model] 설정
# (dim, tau, keyframe, spatial/temporal cache)과 학습 해상도 size를 자동 복원 —
# CLI 플래그(--gmc, --tau-on, --size, --no-spatial-cache)를 준 경우에만 덮어씀.
# 검증 — 소스별 행 + MEAN(src)/POOLED(px) 통합행, --max-clips는 소스당 개수
python scripts/eval.py --ckpt work_dirs/scannet/latest.pt --data scannet:/data/scannet
python scripts/eval.py --ckpt work_dirs/scannet/latest.pt --data scannet:/data/scannet --sweep-tau  # Go/No-Go 곡선

# 추론 — depth PNG는 기본 <ckpt dir>/viz/ 저장 ('none'으로 비활성)
python scripts/infer.py --ckpt work_dirs/scannet/latest.pt --video cam.mp4
python scripts/infer.py --ckpt work_dirs/scannet/latest.pt --frames-dir data/cctv/seq0/rgb

# 실측 속도 — active ratio별 latency/FPS, cache on/off, peak VRAM, 스트림당 state
python scripts/bench.py --ckpt work_dirs/main_v8/latest.pt
python scripts/bench.py --ckpt work_dirs/main_v8/latest.pt --half --streams 4
# --bucket N: 모아온 active 토큰 수를 N의 배수로 올림 패딩(결과 불변, 패드는 Δ-gating으로
# 꺼짐). 희소 경로가 프레임마다 새 shape이던 문제를 없애 --compile이 의미를 갖게 한다.
python scripts/bench.py --ckpt work_dirs/main_v8/latest.pt --bucket 64 --compile
# 주의(REPORT §4.24, §4.27b): 융합 스캔 커널 이후 4090에서는 dense 경로가 항상 더 빠르다
# (fp32 1.29 ms vs 희소 2.0~2.2 ms). fp16+compile dense는 0.378 ms / 2646 FPS로
# DA v2 Small(0.816 ms / 1226 FPS)보다 2.2배 빠르고 VRAM은 2.4배 적다.
# 희소 경로의 이득은 연산량(37.0%)과 스트림당 state로 한정되며, 그것이 시간으로
# 환산되는지는 에지 기기 실측 전까지 미검증이다.

# head의 양자화 바닥을 깊이 구간별로 측정 — bin 개수/범위를 바꾸기 전에 먼저 볼 것
python scripts/bin_probe.py --data vkitti2:/data/vkitti2 --holdout Scene06

# 출력 구조의 상한 — GT를 같은 토큰 그리드 병목에 통과시켜 "완벽한 head"의 점수를 잰다.
# 모델이 상한 근처면 패치/해상도가 병목, 한참 아래면 용량·학습이 병목 (REPORT §4.27a)
python scripts/ceiling_probe.py --data tum:/data/tum --patch 16 8

# 검출기만 따로 측정 — infer.py는 '지불한 연산'을 보고하므로 dense 폴백이 가져간 프레임이
# 100%로 찍힌다. 폴백을 끄고 tau를 스윕해 게이팅 전략끼리 같은 축에서 비교 (REPORT §4.26)
python scripts/gate_probe.py --ckpt work_dirs/v9-60k/latest.pt \
    --frames-dir /data/kitti/2011_09_26_drive_0002_sync/image_02/data

# 이동 카메라 (ego-motion): Low-Res GMC + feature gating (IDEA.md §3.5)
python scripts/infer.py --ckpt work_dirs/kitti/latest.pt --video dashcam.mp4 --gmc
python scripts/eval.py --ckpt work_dirs/kitti/latest.pt --data kitti:/data/kitti --gmc --sweep-tau
```

## Configs (`configs/`)

데이터셋 특성별 최적화: 모션 양이 스킵 상한 결정, 실내/실외가 tau·keyframe 주기 결정.

| config | 대상 | max_skip | tau_on | 비고 |
|---|---|---|---|---|
| `cctv.toml` | 고정 카메라 (주 타깃) | 0.9 | 0.01 | clip 16, keyframe 60 |
| `scannet.toml` | 실내 핸드헬드 | 0.8 | 0.02 | |
| `nyu.toml` | 실내 (Kinect) | 0.8 | 0.02 | |
| `bonn.toml` | 실내 동적 객체 | 0.7 | 0.02 | tau_off 낮춤 — 동적 객체 유지 |
| `kitti.toml` | 주행 (전역 모션) | 0.5 | 0.05 | keyframe 10 |
| `vkitti2.toml` | 합성 주행 (dense GT) | 0.5 | 0.05 | kitti와 동일 프로파일 |
| `mixed.toml` | 전체 혼합 (generalist) | 0.7 | 0.02 | 이후 개별 config로 fine-tune |
| `main.toml` | **본 학습**: TartanAir V2 + PointOdyssey + vkitti2 | 0.5 | 0.05 | 256px, 100k steps, PoC ablation 결과 반영 (§4.5) |
| `main_v8.toml` | **본 학습**: 실촬(TUM/Bonn) + 합성 3종 | 0.5 | 0.05 | 4방향 scan, local conv, DPT, 64 bin + bin CE |
| `t1_binrange.toml` / `t1_bin128.toml` | T1-5 원거리 ablation | | | `d_max` 150→600 (+ bins 128) — PLAN.md T1-5 |

기본 꺼져 있는 손실 두 개(`--warp-weight` / `--edge-weight`): 전자는 RAFT flow로 워프한
log-depth 잔차를 GT 잔차에 맞추는 TCE의 학습판, 후자는 GT depth gradient 밴드에 가중한
log L1(전경 물체용). **확정 체크포인트 `work_dirs/v9-60k`는 둘 다 2.0으로 켜고 학습**했다
(REPORT §4.23):

```bash
python scripts/train.py --config configs/main_v8.toml \
    --resume work_dirs/main_v8/latest.pt --resume-partial \
    --steps 60000 --seed 0 --warp-weight 2.0 --edge-weight 2.0 \
    --work-dir work_dirs/v9-60k
```

## 산출물 (`work_dirs/<이름>/`)

`latest.pt`(가중치 + 재현 메타), `train.log`, `config.toml`(**실효 설정**: CLI+TOML 병합값,
`[model]`, `[meta]`에 git commit/seed/torch·CUDA 버전/실행 명령), `eval.txt`(검증 기록 누적),
`viz/`(depth PNG).

## 데이터셋

본 학습(`main.toml`)용 실 데이터셋은 `/archive/Dataset_SOKKANAEM`에 있음
(vkitti2, tartanair_v2, pointodyssey — `scripts/download_data.sh`로 받음).

모든 데이터셋은 정규 포맷 `(frames, depth[m], valid)` 클립으로 환원.
어댑터는 "(rgb, depth) 경로 쌍 시퀀스 + depth 스케일"만 제공 (`sokkanaem/data.py`).

| spec | 레이아웃 | 스케일 |
|---|---|---|
| `scannet:/root` | `scene*/color, depth` | 1000 |
| `tum:/root`, `bonn:/root` | `seq*/rgb, depth` | 5000 |
| `nyu:/root` | `seq*/rgb, depth` | 1000 |
| `kitti:/root` | `drive*/image_02/data, proj_depth/groundtruth/image_02` | 256 |
| `vkitti2:/root` | `Scene*/변형/frames/{rgb,depth}/Camera_*` (cm, 65535=하늘→invalid) | 100 |
| `folder:/root:SCALE` | `*/rgb, depth` (범용) | 지정 |

`--data` 반복 지정으로 혼합. WeightedRandomSampler가 데이터셋별 추출 확률 균등화
(대형 데이터셋이 소형을 잠식하지 않음). 스케일 상이(실내 10m vs KITTI 80m)해도
scale-invariant log loss라 혼합 안전. 새 데이터셋 = 어댑터 함수 ~5줄 + 레지스트리 1줄.

**계획된 학습 커리큘럼** (IDEA.md §3.4): Stage 1 — **TIMo**(고정형 ToF 실내 모니터링,
실측 GT) 지도 학습 → Stage 2 — **VIRAT**(감시 비디오, GT 없음)에 SOTA 교사 모델로
pseudo depth GT 생성 후 distillation. 두 데이터셋 모두 어댑터 추가 예정.

## 구조

```
sokkanaem/
  detector.py   변화 감지: MSE + hysteresis + dilation + keyframe refresh (§3.1)
  gmc.py        Low-Res GMC: 저해상도 특징점 + RANSAC 호모그래피 + full-res 워핑 (§3.5)
  ssm.py        Selective SSM + Δ-gating, 순수 PyTorch 레퍼런스 스캔 (§3.2)
  scan_triton.py  융합 selective-scan 커널 (추론 전용). 레퍼런스 스캔의 청크별
                (B,C,C,P,S) 쌍별 텐서를 없애 재귀를 레지스터에서 돈다 —
                Δ-gating bit-exactness·평가 지표 불변, dense 11.4→1.98 ms
  model.py      T-Mamba/S-Mamba 교차 백본 + 경량 디코더, 스트리밍 API (§3.0-3.3)
                스트림별 상태(SSM hidden/detector/prev frame)는 전부 state dict에 —
                모델 하나로 다중 스트림 처리 가능. from_checkpoint()가 config 복원 로드
  data.py       멀티 데이터셋 로더: 어댑터 레지스트리 + 정규 클립 포맷 + 균등 혼합 샘플러
  losses.py     SI-log + gradient + temporal consistency (validity 마스킹)
scripts/
  train.py      학습 (config 기반, 랜덤 마스크 스케줄링 §3.4)
  eval.py       검증 (소스별 정확도/시간 안정성/tau sweep/unscaled metric·scale drift)
  infer.py      스트리밍 추론 + depth PNG 저장
  bench.py      실측 wall-clock: active ratio별 latency/FPS, cache on/off, VRAM
  ceiling_probe.py  출력 구조 상한(GT를 토큰 그리드에 통과) — 다음 개선을 어디에 쓸지 결정
  gate_probe.py     폴백 끈 순수 검출기 active 비율, tau 스윕 (게이팅 전략 비교용)
  prepare_data.py  데이터셋 레이아웃 검증, 비디오 -> 프레임 추출
configs/        데이터셋별 최적화 학습 설정 (TOML)
tests/          핵심 주장 검증: mask=0 ⇒ bit-exact state copy, GMC 정렬 + 피처 게이팅,
                다중 스트림 독립성, ckpt config 복원
```

## PoC 범위 (로드맵 1단계)

포함: Δ-gating 수학 검증, 스킵 비율 측정, 스트리밍 추론, 시간 일관성 loss.
이후 실제 데이터셋 학습(§4)과 융합 스캔 커널(`scan_triton.py`, REPORT §4.24)까지 완료.
**남은 것은 에지 기기 실측**: 4090에서는 커널 이후 모델이 오버헤드에 묶여 FLOPs 절감이
시간으로 나타나지 않는다 — 스킵의 이득은 현재 연산량과 스트림당 state로만 입증돼 있다.
