# PLAN — 약점 해소 계획

기준일 2026-07-30. 근거는 [REPORT.md](REPORT.md) §4.15(외부 비교)·§4.19(감사 후속),
[reports/20260729.md](reports/20260729.md). 진행 기록은 [PROGRESS.md](PROGRESS.md).

비용은 실측 기준: 8k step 학습 ≈ 1.7h, 60k step ≈ 14.5h, 소스별 eval(100클립×5소스) ≈ 20분.
GPU는 RTX 4090 1장.

체크박스 규칙: `[ ]` 미착수 · `[~]` 진행 중 · `[x]` 완료 · `[-]` 폐기/불필요 판정.
완료 항목은 근거(REPORT 절 또는 work_dir 경로)를 한 줄로 남긴다.

## 약점 목록 (외부 모델 대비)

| # | 약점 | 처리 |
|---|---|---|
| W1 | 절대 δ1이 SOTA 대비 열위, 최신 모델은 baseline 재비교 자체가 없음 | T0-1 + 프레이밍 |
| W2 | 시간 우위가 t-delta 하나뿐, OPW/TCE는 DA3·VDA에 열위 | T1-6 |
| W3 | 움직임 많으면(active 70%) 효율 이득 소멸 | T2-12 (범위 명시) |
| W4 | 희소 경로 compile 불가 → compiled dense와 동률 | T2-9, T2-10 |
| W5 | Jetson·전력 실측 없음 | T2-11 |
| W6 | 원거리 RMSE 열위(전역 64 log bin) | T1-5 |
| W7 | 전경 물체 depth가 32px 상수 oracle보다 나쁨 | T1-7 |
| W8 | cache가 active 40~80%에서 U자형 악화 | T0-3 |
| W9 | sequence holdout뿐, cross-domain zero-shot 미검증 | T3-13 |
| W10 | metric scale 신뢰 낮음(median scaling 기준) | T0-4 |
| W11 | GMC가 합성 clean geometry에서만 검증 | T3-14 |
| W12 | seed 반복 없음, 전 결과 1회 실행 | T0-2 |

## Tier 0 — 즉시, 논문 차단 요소 (~1.5일)

- [x] **T0-3 · W8 고활동 dense 폴백 — 채택(기본 0.4).** detector active가 임계를 넘으면 그
  프레임은 dense 경로로 돌리고 cache를 전부 갱신, `active_ratio`도 1.0으로 정직하게 보고.
  실촬 AbsRel 0.1685→0.1633, δ1 0.8083→0.8211, TCE 0.0354→0.0351, 대가는 active 22.2→32.2
  및 t-delta 0.0881→0.0915. 합성은 정확도 동결에 연산만 +12.6pt. **속도 주장은 이제 실촬
  active 32%에서 인용해야 한다.** U자형 악화는 v6(추론 전용 캐시) 유물로 판정.
  → REPORT §4.20a, `sokkanaem/model.py`, `tests/test_arch.py::test_high_motion_frame_falls_back_to_dense`
- [x] **T0-1 · W1 최신 체크포인트로 baseline 공통표 재생성.** → REPORT §4.20c
  - [x] DA3 Base 5소스 — 실촬에서 우리 열위(AbsRel 0.1244 vs 0.1633), t-delta만 우위.
  - [x] DA v2 Small 5소스 — `baselines` env에 `transformers`가 사라져 `sokkanaem` env로 재실행
    (`work_dirs/t0-da2.sh`, 로그 `work_dirs/t0-da2.log`).
  - [-] VDA Small — 로컬 체크아웃 소실(`Video-Depth-Anything` 디렉터리 없음). 재클론 +
    metric 체크포인트 다운로드가 선행 조건이라 이번 라운드 제외, §4.15의 기존 수치 인용 유지.
- [x] **T0-2 · W12 seed 분산 분리 — bin CE는 분산 밖.** 실촬 AbsRel 0.1795±0.0026 (bin CE)
  vs 0.1963±0.0050 (없음), 차이 −8.6%로 seed 분산의 3~6배. 새 사실 둘: bin CE는 t-delta
  +8%·TCE +12%를 대가로 지불하고(→ T1-6), **합성 δ1의 seed 표준편차가 ±0.015**라 3pt 미만의
  합성 δ1 차이는 노이즈다. → REPORT §4.20b, `work_dirs/t0-seed*-bin*`
- [x] **T0-4 · W10 scale drift.** 8→32프레임에서 drift 실촬 0.0235→0.0444, 합성
  0.0934→0.1587 (√4=2배보다 낮아 체계적 드리프트가 아니라 확산). TartanAir는 배율 1.98로
  median scaling 없이는 무의미. metric scale 주장은 계속 하지 않는다. → REPORT §4.20d

## Tier 1 — 정확도 (~1주, 대부분 GPU 대기)

**측정된 seed 노이즈 기준선**(T0-2): 실촬 AbsRel ±0.005, 실촬 δ1 ±0.004, 합성 δ1 ±0.015.
이보다 작은 차이는 채택 근거로 쓰지 않는다. 대조군은 `work_dirs/t0-seed0-bin0.2`.

- [~] **T1-5 · W6 원거리 RMSE — bin 개수가 아니라 범위 문제로 판명.** `scripts/bin_probe.py`가
  head의 양자화 바닥을 깊이 구간별로 측정: 80 m 미만은 이미 AbsRel 0.0000(= bin 개수는
  병목이 아님), 반면 vkitti2 픽셀의 0.8%가 최상위 bin 중심(115 m, `d_max=150`에서 파생)을
  넘어가고 **그 0.8%가 제곱오차의 54%**를 낸다. 원래 계획의 128 bin·log-disparity·per-image
  adaptive는 근거가 사라져 폐기하고 2개 arm만 돌린다.
  - [-] log-disparity binning — log-depth와 부호만 다른 동일 분할이라 무의미.
  - [-] per-image adaptive bin / 데이터셋별 range 정규화 — 80 m 미만 바닥이 이미 0이라 불필요.
  - [~] `t1-binrange`: bins 64 유지, `d_max` 150 → 600.
  - [~] `t1-bin128`: bins 128 + `d_max` 600 (넓힌 로그 범위가 근거리 해상도를 깎는지 분리).
- [~] **T1-6 · W2 warp residual 손실.** `warp_residual_loss` 구현 — RAFT flow로 워프한
  log-depth 잔차를 GT의 잔차에 맞춘다(TCE의 학습판). 기존 `temporal_loss`는 정적 패치만 보고
  상수 출력이 전역 최소라 이걸 못 산다. arm `t1-warp0.5`, `t1-warp2.0`.
  기준: TCE 0.0344 → 0.030 이하이면서 실촬 AbsRel 열화 0.005 이내.
- [~] **T1-7 · W7 전경 물체.** `edge_weighted_loss` 구현 — GT log-depth gradient를
  max-pool로 넓힌 경계 밴드에 가중한 log L1. arm `t1-edge0.5`, `t1-edge2.0`.
  기준: 32px 블록 상수 oracle(AbsRel 0.1459) 추월.
- [ ] **T1-8 최종 60k 재학습** (14.5h): T1-5~7에서 seed 노이즈를 넘어선 조합만.

## Tier 2 — 시스템 (~2주). 여기가 실제 기여

- [~] **T2-9 · W4 버킷 패딩으로 static shape 확보.** 구현 완료: `pad_to_bucket`이 모아온
  active 토큰 수를 `bucket`의 배수로 올림하고, 패드는 Δ-gating으로 꺼서 **결과가 비트 수준
  동일**(역방향 스캔이 패드를 먼저 방문해도 상태가 안 움직임 —
  `tests/test_arch.py::test_bucket_padding_does_not_change_the_result`).
  `compile_sparse()`가 그 위에서 스캔만 컴파일한다(gather는 `nonzero`라 데이터 의존적이라
  eager 유지). 측정은 Tier 1이 GPU를 놓으면 자동 실행(`work_dirs/tier2-bench.sh`).
  기준: active 22%에서 compiled dense(4.67 ms)의 절반 이하.
- [ ] **T2-10 · W4 Triton 블록 희소 커널.** T2-9로 부족할 때만.
- [ ] **T2-11 · W5 Jetson Orin 실측.** `scripts/bench.py` 그대로 사용(기기 확보가 선행).
  latency, FPS, power, energy/frame, active별 곡선.
- [ ] **T2-12 · W3 운용 범위 명시.** active-vs-speedup 곡선(측정 완료)을 싣고 "active>50%는
  dense가 낫다"를 명문화. 자동 전환은 T0-3가 이미 수행.

## Tier 3 — 일반화 (선택, ~2일)

- [ ] **T3-13 · W9 cross-domain zero-shot.** 학습에 안 쓴 NYU/ScanNet/KITTI raw로 평가.
  어댑터는 이미 있음. 결과가 나빠도 기록한다.
- [ ] **T3-14 · W11 실 이동 카메라 GMC 검증.** KITTI raw 또는 TUM handheld.

## 고치지 않고 프레이밍으로 처리

- [ ] **W1 절대 δ1**: 4M 스트리밍 모델로 120M 제너럴리스트를 정확도에서 이기는 것은 목표가
  아니다. 주장을 efficiency-accuracy Pareto + 플리커로 좁히고, 동급 파라미터(24~28M) 대비
  4M의 AbsRel/RMSE 우위와 δ1 열위를 함께 적는다. v5(dim 384) 부활은 이 프레임을 스스로
  깨므로 보류 유지.
- [ ] **W2 확대 해석 금지**: T1-6이 성공해도 주장은 "후처리 없는 원시 플리커 억제"로 유지.
  OPW/TCE 1위는 목표가 아니다.

## 순서

T0 전체 → T1-5/6/7 순차(GPU 1장) → 이긴 조합으로 T1-8 → T2-9 → 기기 확보되면 T2-11 →
여유 시 T3.
