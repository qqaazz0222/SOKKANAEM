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

- [x] **T1-5 · W6 원거리 RMSE — bin 가설 기각, 목표는 edge 손실이 대신 달성.** `scripts/bin_probe.py`가
  head의 양자화 바닥을 깊이 구간별로 측정: 80 m 미만은 이미 AbsRel 0.0000(= bin 개수는
  병목이 아님), 반면 vkitti2 픽셀의 0.8%가 최상위 bin 중심(115 m, `d_max=150`에서 파생)을
  넘어가고 **그 0.8%가 제곱오차의 54%**를 낸다. 원래 계획의 128 bin·log-disparity·per-image
  adaptive는 근거가 사라져 폐기하고 2개 arm만 돌린다.
  - [-] log-disparity binning — log-depth와 부호만 다른 동일 분할이라 무의미.
  - [-] per-image adaptive bin / 데이터셋별 range 정규화 — 80 m 미만 바닥이 이미 0이라 불필요.
  - [-] `t1-binrange`(`d_max` 600): 어느 지표도 seed 노이즈를 못 넘음(합성 RMSE 15.53→15.37).
    상한을 풀어줘도 모델이 그 범위를 안 쓴다 — 원거리 열위는 head 표현력이 아니라 256px에서
    300~600 m를 못 보는 정보 한계로 판정. `d_max`는 kwarg로 남기고 기본 150 유지.
  - [-] `t1-bin128`: 양쪽 도메인 모두 악화. 기각.
  - [x] **목표치는 엉뚱한 곳에서 달성됐다**: T1-7의 edge 2.0이 bin을 안 건드리고 합성 RMSE
    15.53→**14.51**(목표 14.60)을 냈다. → REPORT §4.21a, c
- [x] **T1-6 · W2 warp residual 손실 — 2.0 채택.** 8k에서는 시간 지표 개선을 실촬 정확도로
  샀지만(TCE −12%·OPW −16% vs δ1 −0.0095), **60k에서 그 교환이 사라진다** — edge 2.0과 함께
  60k를 돌린 `v9-60k`가 실촬 AbsRel 0.1633→0.1595, t-delta 0.0915→0.0751로 양쪽 다 가져갔다.
  8k arm 스크리닝은 항의 부호는 정해도 크기는 못 정한다. → REPORT §4.21b, §4.23
- [x] **T1-7 · W7 edge 가중 손실 — 채택.** edge 2.0: 실촬 AbsRel 0.1773→**0.1745**,
  δ1 0.8082→**0.8107**, 합성 RMSE 15.53→**14.51**. 대가는 합성 t-delta +32%로 T1-6이
  상쇄할 수 있는 종류. TUM 단독 AbsRel 0.1447→0.1358. 32px oracle(0.1459)은 프로토콜이
  달라 직접 비교로 쓰지 않는다. → REPORT §4.21c
- [x] **T1-8 최종 60k 재학습 — `work_dirs/v9-60k` 확정.** tier1b 선택자가 edge 2.0 + warp 2.0을
  골랐고 60k(13h11m) 결과가 직전 확정 체크포인트를 **연산 증가 없이**(active 32.2% 동일) 전
  지표에서 앞선다: 실촬 AbsRel 0.1633→**0.1595**, δ1 0.8211→**0.8262**, t-delta
  0.0915→**0.0751**, TCE 0.0351→**0.0323**, 합성 RMSE 15.18→**14.22**. 단 대조군과 init 계보가
  달라 **손실 항의 인과 주장은 하지 않는다**(§4.23d). → REPORT §4.23
  - [-] `v9-edge-60k`(edge 단독 대조 arm): step 23100/60000에서 죽음, 근거였던 8k 순위가
    60k에서 뒤집혀 재개하지 않고 폐기. → REPORT §4.23e

## Tier 2 — 시스템 (~2주). 여기가 실제 기여

- [x] **T2-9 · W4 버킷 패딩 — 희소 경로가 compiled dense를 다시 앞선다(합격선은 미달).**
  `pad_to_bucket`이 active 토큰 수를 64의 배수로 올리고 패드를 Δ-gating으로 꺼서 **결과 불변**
  (`tests/test_arch.py::test_bucket_padding_does_not_change_the_result`), `compile_sparse()`가
  그 위에서 스캔만 컴파일한다. active 22%에서 4.87 → **2.99 ms**(334 FPS)로 compiled
  dense(4.70 ms) 대비 **1.57배**, 실촬 평균 32%에서 3.94 ms로 1.19배.
  기준(2.34 ms)에는 미달이고 active 50%에서 동률·70%에서는 dense가 낫다 — 그 오른쪽 끝은
  T0-3 폴백이 자동 처리. 버킷만 켜고 컴파일 안 하면 항상 느리다. → REPORT §4.22
- [x] **T2-10 · W4 융합 Triton 스캔 — 합격선 통과, 대신 희소 우위가 사라졌다.** 병목은
  gather가 아니라 스캔이었다(희소 프레임의 71%). `scan_triton.py`가 쌍별 감쇠 텐서를 없애고
  재귀를 레지스터에서 돌린다(추론 전용, Δ-게이팅 bit-exact 유지, 평가 지표 불변).
  dense eager 11.4→**1.98**, dense compiled 4.70→**1.29**, 희소+버킷+컴파일 2.99→**2.04 ms**.
  합격선 2.34 ms는 통과했으나 **모든 active에서 dense compiled가 더 빠르다**(1.29 vs 2.0~2.2)
  — 희소가 아끼던 스캔이 공짜가 되면서 고정 오버헤드(gather·argsort·clone·동기화)만 남았다.
  wall-clock 기준 시스템 기여 주장은 철회하고 연산량·메모리로 한정한다. → REPORT §4.24
- [ ] **T2-11 · W5 Jetson Orin 실측.** `scripts/bench.py` 그대로 사용(기기 확보가 선행).
  latency, FPS, power, energy/frame, active별 곡선.
- [x] **T2-12 · W3 운용 범위 명시 — 경계가 이동했다.** §4.22 기준의 "active>50%면 dense"는
  T2-10 이후 "**4090에서는 항상 dense**"가 됐다. 희소 경로는 연산량(37.0%)과 스트림당
  state에서만 이득이며, 그것이 시간으로 환산되는지는 T2-11이 가른다. → REPORT §4.24d

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

~~T0 전체 → T1-5/6/7 순차(GPU 1장) → 이긴 조합으로 T1-8 → T2-9~~ 완료(2026-08-07).
남은 것: T2-10 Triton 커널 → T3-13 zero-shot → 기기 확보되면 T2-11 → 여유 시 T3-14.
프레이밍 항목(W1/W2)은 논문 작성 시 처리.
