# PROGRESS

작업 상태 추적용. 실험 수치는 [REPORT.md](REPORT.md), 아이디어 배경은 [IDEA.md](IDEA.md) 참조.
로드맵 4단계는 IDEA.md §7 기준.

## 로드맵 상태

| 단계 | 상태 | 비고 |
|---|---|---|
| 1. PoC (4주) | ✅ 완료 (Go) | vkitti2, 스킵 55%에서 AbsRel 열화 +0.15% — 기준(5% 이내) 크게 상회 |
| 2. 본 학습 (8주) | 🟡 정확도 개선 중 (v5) | v3 완료(δ1 0.40) → 외부 SOTA 대비 정확도 격차 확인 → v4 distillation 시도 실패(효과 없음) → v5 모델 용량 증가(dim 192→384, 9.92M) 진행 중 |
| 3. 시스템 (4주) | ⬜ 미착수 | Triton 블록 희소 커널, Jetson Orin 실측, 데모 |
| 4. 논문화 | ⬜ 미착수 | CVPR/ICCV(efficiency) 또는 CoRL/IROS(실시간 시스템) |

## 현재 상태 (2026-07-24 기준)

- **PC 재설치로 conda 환경 소실 → 재구축 완료**: `conda env create -f environment.yml` + `uv pip install -e ".[dev,video]"`. pytest 14개 전부 통과.
  - **주의: `torch>=2.3` 무제한 상한 탓에 2.13.0+cu130이 잡힘 — `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`로 conv2d 자체가 안 됨.** `torch==2.8.0`(cu128, cudnn 9.10)으로 다운그레이드해 해결, 재발 방지로 `pyproject.toml`에 `torch<2.9` 상한 박음.
  - `work_dirs/*` 일부가 root 소유라 hyunsu 계정으로 못 쓰던 문제 있었음 — 사용자가 별도 터미널에서 `chown` 완료.
- **체크포인트 resume 기능 검증 후 사용**: `scripts/train.py`/`sokkanaem/model.py`의 `{model, optim, step}` 딕셔너리 저장 + `--resume` 복원 로직(이전 세션에서 미커밋 상태로 남아있던 것)을 합성 데이터로 스모크 테스트해 정상 동작 확인.
- **데이터셋 경로 버그 수정 (2026-07-24, 이전 세션)**: `configs/main.toml`, `configs/vkitti2.toml`가 존재하지 않는 로컬 경로를 가리키던 것을 `/archive/Dataset_SOKKANAEM` 하위로 통일.

### 사건: 본 학습 v1 (100k 완주) 붕괴 발견 → v2 데이터 정합성 버그 다수 발견 → v3 재학습

1. **v1 완주 (100,000/100,000 step)**: EMA/Kendall auto-loss-weight/normal-loss/해상도 커리큘럼/LR warmup+cosine/grad clip 전부 적용해 처음부터 학습, 정상 완료된 것처럼 보였음(로그상 loss 변동, 크래시 없음).
2. **holdout eval 돌리다 발견**: `scripts/eval.py` 결과 모든 tau에서 AbsRel 0.80, δ1 0.115, **t-delta 정확히 0.0000**. 직접 forward 찍어보니 **모든 픽셀·모든 프레임이 정확히 같은 상수(4.85)** — 완전 붕괴. raw model/EMA 둘 다 동일 증상(EMA 버그 아님).
   - **원인**: Kendall auto-loss-weight가 `temporal_loss`(연속 프레임 depth 동일해야 한다는 loss) 가중치를 clamp 상한(`exp(8)≈2981×`)까지 밀어붙임 — si_log(~0.5×)·grad(~0.6×) 대비 압도적. `temporal_loss`는 **상수 출력이 정확한 전역 최솟값(0)** 인 트리비얼 해가 있어서, 가중치가 그 정도로 쏠리면 실제 깊이 추정을 포기하고 상수로 붕괴하는 게 수학적으로 최적. IDEA.md §4.5의 모든 과거 ablation(고정가중치 1/0.5/0.1)은 이 문제 없었음 — auto_loss_weight 하나가 원인.
   - **v1 결과 전량 폐기**, `work_dirs/main-COLLAPSED-autolossweight-20260724`로 보관.
3. **holdout eval 재시도 중 데이터셋 버그 별도 발견 (v1 학습 자체와는 무관, 데이터 로더 단)**:
   - tartanair2 10개 env 중 **6개**(OldTownFall + Downtown/ModularNeighborhood/Office/SeasonalForestSpring/Supermarket)가 압축 해제 시 `env/env/Data_easy/...`로 이중 중첩되어 있어서 어댑터 glob이 전혀 못 찾음 — 학습·eval 양쪽에서 완전히 빠져있었음(오염 아니고 그냥 미사용). 전부 `mv`로 경로 정정, 이중 중첩이 없어질 때까지 재귀적으로 확인.
   - `ModularNeighborhood`의 P008은 image 프레임에 827개 구멍(3421 vs depth 4248) — `_pair_sorted`가 이름이 아니라 정렬 순서로 짝짓기 때문에 프레임이 밀려서 잘못 페어링될 위험. P009/P010은 image 자체가 없음(depth만). 셋 다 `Data_easy/_excluded/`로 이동해 어댑터 glob에서 제외.
   - vkitti2는 문제 없음. pointodyssey는 숨김 임시파일(`.depth_00411.png.BWMPW4`, 다운로드 중단 잔재) 하나 발견해 삭제 — Python glob이 dotfile을 안 잡아서 실제 영향은 없었음.
   - **결과**: tartanair2 정상 시퀀스 44→**74**개, 전체 학습 clip 수 160,973→**195,475**개로 증가. v1은 이 버그가 있는 상태로 학습했었음(즉 데이터의 일부 손실 상태로 진행됐던 것 — 붕괴 원인은 아니지만 별개로 고쳐야 했던 결함).
4. **수정 후 v3 재학습 착수**: `configs/main.toml`에서 `auto_loss_weight` 제거, 검증된 고정가중치(si_log + 0.5·grad + 0.1·temporal + 0.05·normal)로 복귀. **재발 방지로 `scripts/train.py`에 붕괴 자동감지 추가**(`sokkanaem/collapse.py`) — depth 예측 std가 `--collapse-eps`(기본 1e-4) 밑으로 `--collapse-patience`(기본 1000) step 연속 유지되면 즉시 중단·체크포인트 저장 후 종료, 8시간 날리기 전에 조기 발견. tmux 세션 `main`에서 처음부터 재학습 중, `mixed dataset: 195475 clips from 3 sources` 확인.
5. **v3 1차 시도, step ~1600에서 `OSError: image file is truncated`로 DataLoader worker 죽으면서 크래시**: 스크래핑 데이터셋(다운로드 중단 잔재) 특성상 손상 파일이 더 있을 수 있다고 보고, `ClipDataset.__getitem__`(`sokkanaem/data.py`) 자체를 방어적으로 고침 — `OSError`/`ValueError` 발생 시 경고 로그 남기고 다른 랜덤 클립으로 최대 10회 재시도 (개별 파일 하나 찾아 지우는 대신 로더 레벨에서 근본 수정, 향후 또 다른 손상 파일 나와도 학습 안 죽음). 체크포인트 저장 전(2000 step 이전) 크래시라 손실 미미 — 처음부터 재시작.
   - **후속 조사**: 실제 학습 중 걸린 손상 클립(`tartanair_v2/Hospital` P003/P004/P006)을 학습 밖에서 PIL로 직접 재검사(image 19950개, depth 19950개 전수) — **전부 정상 로드됨**. 즉 영구 손상이 아니라 `num_workers=4`가 `/archive`에 동시 접근하면서 생긴 **일시적 read 실패**로 추정. 로더를 "같은 파일 2회 재시도 → 그래도 실패하면 랜덤 클립 폴백"으로 개선(멀쩡한 데이터를 불필요하게 버리지 않도록). 이미 20%+ 진행된 현재 런은 안전하게 도는 중이라 재시작 안 하고 다음 런부터 적용.

## 완료된 마일스톤 (날짜순)

- **2026-07-06** — Δ-gated Mamba PoC 초기 구현 (`ab6014b`)
- **2026-07-07** — 이동 카메라 확장(GMC+feature gating) 설계 및 구현, vkitti2 PoC 결과 확보(Go 판정), wall-clock 베이스라인, 청크 segment-sum 스캔(3.6× FPS), static-patch 캐싱(phase 3, opt-in)
- **2026-07-08** — CUDA graph 캡처(2.4–2.9×), detector-driven 마스크 학습 플래그, eval-only ablation(keyframe 주기, MSE vs cosine), 본 학습 인프라(주기적 ckpt, holdout split, aspect crop), 256px OOM 수정(gradient checkpoint), 장기 스트림 드리프트/스트리밍 active% 조사, 마스크 분포 3-arm ablation(iid random @ 0.5 확정)
- **2026-07-08** — 본 학습 데이터셋 다운로드 스크립트 작성 (`/archive`로 VDA 믹스)
- **2026-07-09** — 스톨 워치독 추가, tartanair 다운로더를 `hf_hub` → `wget -c`로 교체 (다운로드 중 재시작 버그 실측·수정)
- **2026-07-10** — tartanair2 + pointodyssey 어댑터, `configs/main.toml` 작성 → 본 학습 착수
- **(런타임, 로그 기준) ~2026-07-14** — 본 학습 7500 step까지 진행 후 중단
- **2026-07-24** — 데이터셋 경로 버그 수정, README에 데이터 위치 명시, checkpoint resume 기능 작업 중
- **2026-07-24** — 본 학습 v1 100k step 완주 → 완전 붕괴(상수 출력) 확인·폐기, tartanair2 6개 env 데이터 미사용 버그 발견·수정(160,973→195,475 clips), 붕괴 자동감지 추가, v3 재학습 착수
- **2026-07-25** — v3 100k step 완주(붕괴 없음), 손상 프레임 방어 로더 추가(1차 재시도 중 크래시 겪고 수정), holdout eval 정상 확인(AbsRel 0.382/δ1 0.397, 스킵 100%→11%에도 정확도 유지+t-delta 0.214→0.154 개선), EMA vs raw 사실상 동일(cosine LR이 막판 0 근접이라 차이 안 남)
- **2026-07-25** — 외부 SOTA 3개(DA v2, DA3, Video Depth Anything metric) 같은 holdout으로 직접 실행·비교. `sokkanaem` env 보호 위해 별도 conda env 2개(`baselines`, `vda`) 신설. 결과: 단일 프레임 정확도는 파라미터 규모 순(전부 우리보다 큰 모델이 더 정확, 예상대로), **시간 안정성은 SOKKANAEM이 압도**(명시적 시간 모듈 있는 VDA조차 t-delta 17배 나쁨) — REPORT.md §4.8

### 2026-07-26 — 논문화 전 감사: 지표·회계 결함 4건 발견, v6/v7 계획 확정

논문 초고 착수 전에 주장-근거 정합성을 점검한 결과, 고쳐야 할 것이 4건 나왔다. 상세 수치는
REPORT.md §4.10–4.13.

1. **지표 프로토콜 불일치**: `eval.py`는 t-delta를 raw 출력에서, baseline 3종은 median-scaled
   출력에서 계산 중이었음(v3 median scale 0.77). 영향 0.95배로 결론은 유지되나 프로토콜 통일.
   재발 방지로 모든 per-clip 지표를 `sokkanaem/metrics.py:clip_scores` 한 곳으로 모아
   eval.py·baseline 3종이 같은 함수를 쓰게 정리.
2. **시간 지표 퇴화**: t-delta는 상수 출력이 전역 최적(§4.6 붕괴가 0.0000). OPW(flow-warp)도
   상수 필드엔 무력. → RAFT 기반 OPW + **TCE**(GT 자신의 워프 잔차 기준, 퇴화하지 않음) 추가,
   `--control`로 상수 예측 제어행 병기. TCE의 const 행 = 데이터셋 고유 워프 잔차 바닥으로,
   우리 모델은 그 바닥보다 위 → **절대적 시간 일관성 우위는 주장 불가, baseline 대비만 가능**.
3. **표본 편향**: 모든 홀드아웃 수치가 8,929 클립 중 100개(1.1%), 분산 미표기.
   1,000 클립 재평가에서 $\delta_1$ 0.397 → **0.546**, active% 16.6 → 31.6으로 크게 이동
   (클립별 AbsRel std 0.389). §4.7/§4.8 표는 폐기·대체. baseline 3종도 1,000 클립 재실행 필요.
4. **기여 3(연산량∝변화율) 기각**: FLOPs를 처음 계산해보니 Δ-gating만으로는 active 0%에서도
   풀연산의 **96.4%**. 디코더가 MAC의 67.8%(IDEA §3.3의 자체 예산 22배 위반), Δ-gating은
   static 토큰의 state 갱신만 없애고 readout(58.5%)은 남김. §3.3의 "active 56%에서 372→373 FPS"
   미해명 수치가 이걸로 설명됨(게이팅의 속도 기여 0). 4090에서는 launch-bound라 디코더 실측
   비중은 3%뿐.
   → **v6**(`configs/main_v6.toml`): `ShuffleDecoder`(디코더 비중 68%→7%) + **학습 단에서**
   spatial cache 활성화(지금까지 inference-only 근사였던 경로를 학습된 경로로 승격). 투영:
   active 16.6%에서 38.6%, 0%에서 26.3%. v5 완주 후 착수(사용자 결정).

또한 **실촬 고정카메라 첫 평가**(TUM `fr3/sitting_static`, zero-shot): active%는 $\tau$=0.05에서
**5.9%** 로 CCTV 주장 구간 실측 입증. 그러나 절대 성능은 **상수 예측기에 패배**
(AbsRel 0.371 vs 0.244, $\delta_1$ 0.388 vs 0.616, TCE 0.037 vs 0.011) — 합성 옥외 학습이
실촬 실내로 전이 안 됨. → **v7**: 실촬 실내 fine-tune(TUM static + Bonn Dynamic, 다운로드 중).
로더 결함도 수정: TUM/Bonn은 rgb·depth 타임스탬프·프레임 수가 달라 정렬순 페어링이 GT를 밀고
있었음 → `_pair_by_timestamp`(0.02s 창) + 회귀 테스트.

게이팅 위치 ablation(§4.4의 미실시 항목)도 구현: `gate_mode="drop"`(static 토큰 블록 우회)를
추가해 Δ-gating의 state readout 기여를 iso-active로 분리 측정(실행 중).

### 2026-07-27 — 학습 크래시, I/O 병목 발견, 정확도 최우선 전환

1. **학습 2건 네이티브 크래시** (원인 미확정): v5 `Illegal instruction` (step 45350),
   v7 `Segmentation fault` (step 7850), 약 1시간 간격·서로 독립(v7은 v5 사망 20분 후 시작).
   Python 예외 아님. dmesg 권한 없어 XID/MCE 확인 불가. 메모리 58GB 여유·디스크 정상.
   체크포인트(2000 step 주기)로 v7은 재개해 15k step 완주, v5는 step 44000에서 대기 중.
   **재발 시 원인 추적 필요.**
2. **v7 결과**: 실촬 미학습 홀드아웃에서 AbsRel 0.179 / $\delta_1$ 0.790 — 상수 제어행
   (0.288 / 0.556)을 명확히 이겨 §4.12의 전이 실패 해소. 합성 홀드아웃도 망각 없이 전면 개선
   (AbsRel 0.4292→0.4166, $\delta_1$ 0.5285→0.5327, t-delta 0.2455→0.2131, TCE 0.0879→0.0808).
   실촬+합성 혼합이 양쪽 도메인 모두에 이득.
3. **사용자 지시로 최우선 전환**: 절대 정확도를 최소 DA v2 수준으로. 진단 결과는 REPORT §4.17
   (요약: 용량이 아니라 일반화. 학습 클립에서 이미 $\delta_1$ 0.707). 이식한 관행: 클립 일관
   증강(기존에 증강이 전혀 없었음), DPT식 다중스케일 융합 디코더(파라미터 0.31M→0.13M),
   disparity 회귀 + MiDaS 다중스케일 gradient 손실. 프로브 3종 실행 중.
4. **I/O 병목 발견·해결**: `/archive`가 스피닝 HDD(ST8000DM004)라 208k 클립 랜덤 접근이
   seek-bound — GPU 사용률 0%, 해상도와 무관하게 ~48 frame/s. NVMe(990 PRO, `/`에 1.7T 여유)로
   333GB 전량 복사 후 **0.67s/step → 0.135s/step (5배)**. workers 4→16 + persistent_workers +
   prefetch_factor 4도 함께 적용. 모든 config·스크립트를 `/home/hyunsu/dataset_ssd`로 전환
   (`/archive`는 원본 보관). **이전까지의 모든 학습 시간 추정치는 5배 과대**였음.

### 2026-07-27 — v8 정밀화: Mamba-depth 계열 기법 4종 이식 + 희소 경로 확장

Mamba 기반 깊이 추정에서 쓰이는 고정밀화 기법을 **SOKKANAEM의 논지(변화 없는 영역은 연산을
내지 않는다)를 깨지 않는 위치에만** 넣었다. 원칙: 추가 용량은 active%에 비례하는 경로(공간
스캔)나 dense여도 무시 가능한 비용(depthwise 3x3, 1/2해상도 bin 로짓)에만 배치. 디코더(dense)에는
더 붓지 않는다 — §4.11에서 기여 3을 기각시킨 범인이 바로 dense 디코더였다.

| 기법 | 구현 | 배치 근거 |
|---|---|---|
| SS2D 4방향 cross-scan (Vim/VMamba) | `ssm.py:BiSpatialSSM(directions=4)` + `column_major_order` | raster flatten은 수직 이웃을 gw=16 스텝 떨어뜨려 수평 경계를 뭉갠다. 열우선 순서로 한 쌍 추가. 비용 2배지만 이 경로가 spatial_cache 아래에서 active%에 비례하는 유일한 경로 |
| CNN-Mamba 하이브리드 / local refinement | `model.py:SpatialBlock(local_conv=True)` — depthwise 3x3 x2, 스캔과 병렬 | dense지만 2·dim·9 MAC/token = 스캔의 0.6%. 입력이 dense readout이라 **희소 상태에서도 근사가 아니라 정확**(마스크는 쓰기만 게이팅) |
| 멀티스케일 융합 | 기존 `DPTDecoder`(2026-07-27 프로브)로 충족 | 주파수 영역 정렬은 미적용 — 멀티모달(이벤트 카메라) 입력이 없어 분리할 대상이 없음 |
| Continuous depth binning | `DPTDecoder(bins=64)` — 학습된 log-depth bin 중심에 softmax 기대값 | 스칼라 회귀는 폐색 경계에서 평균을 내지만 분포는 bimodal이어도 하나의 값으로 수렴. bin은 전역·학습형이라 추론 비용은 마지막 3x3의 16·64 vs 16·1 |

요청 범위 밖에서 추가로 넣은 2건:

- **`temporal_cache`** (`TemporalBlock.step_cached`): Δ-gating은 static 토큰의 *state* 갱신만
  없앴고 readout(in_proj/out_proj/C = active 토큰의 58.5%)은 여전히 dense로 냈다. static 패치는
  정의상 픽셀 변화가 τ 미만이고 state는 bit-identical이므로 블록 출력을 재사용. spatial cache와
  같은 트레이드(τ로 유계, 키프레임마다 갱신, 학습 경로로 승격). **기여 3에 남은 가장 큰 구멍**.
- **`--teacher-weight`** (`distill.py:load_frozen_teacher`/`affine_invariant_loss`): frozen DA v2
  Small의 상대 disparity를 affine-invariant L1로 증류. 학습 전용, 추론 비용 0. §4.17 진단(용량이
  아니라 일반화)을 DA v2 자신이 쓴 방법으로 정면 공격하고, Kinect GT가 비어 있는 픽셀에도 타깃을 준다.

**측정된 회계 변화** (256px, `scripts/flops.py --decoder dpt --scan-directions 4 --local-conv --bins 64`):

| | v7 구조 (conv 디코더, 2방향) | v8 |
|---|---|---|
| full compute | 2.337 GMAC | **1.644 GMAC** |
| dense 몫 (embed+decoder) | 69.4% | **25.4%** |
| active%에 비례하는 몫 | 21.8% | **62.0%** |
| active 16.6% (캐시 전부) | 1.841 GMAC (78.8%) | **0.623 GMAC (37.9%)** |
| active 0% | 1.743 GMAC (74.6%) | **0.420 GMAC (25.5%)** |

즉 full compute가 30% 싸지면서 동시에 정밀화 모듈이 들어갔고, 절감 곡선이 ideal에 2배 가까워졌다.
남은 바닥 25.5%는 dense embed(2.3%) + DPT 디코더(23.1%) — 다음 효율 레버는 **디코더 자체의 희소화**
(변하지 않은 패치는 이전 depth 타일 재사용)이며, DPT의 RGB stem이 dense conv라 타일 단위 재작성이 필요.

검증: `tests/test_arch.py` 8건 신설 — 열우선 순열의 정확성(전체/부분집합/가역), 4방향이 실제로
2방향과 다름, 모든 기능 ON에서 동일 프레임 active 0%·depth 안정·희소 경로가 full compute와 일치,
temporal cache가 hidden state를 bit-identical로 유지, bin 중심 단조·범위 내, affine-invariant
손실이 teacher gauge에 불변. 전체 52건 통과. v7→v8 가중치 승계는 `--resume-partial`(비엄격 로드,
스케줄 step 0 재시작) — 실측 84개 신규 텐서/8개 미사용(v7 conv 디코더).

또한 이전 세션의 미커밋 증강 테스트 1건이 실행 순서에 따라 실패하던 것 수정 — depth는 nearest
리샘플로 평탄 구간이 생겨 strict monotonic assert가 성립할 수 없음. 불변식을 "rgb/depth 방향 일치"로 교체.

### 2026-07-28 — v8 학습 완주 + 3-arm 처치 분리 (수치는 REPORT §4.18)

- **v8 60k step 완주** (12h58m, 크래시·붕괴 없음). `work_dirs/main_v8/latest.pt`.
- **eval 결과가 도메인별로 갈림**: 실촬 미학습에서 v7 대비 개선(AbsRel 0.179→0.1642,
  $\delta_1$ 0.790→0.8111), 합성 holdout에서는 후퇴($\delta_1$ 0.5327→0.4573).
- **3-arm 프로브로 원인 분리** (각 8k step): **teacher 출력 증류가 순손실**이었다 —
  끄기만 해서 실촬 $\delta_1$ 0.8156→0.8405, 합성 0.4155→0.5629. 대조군(arm0)이 v8과
  동일해서 추가 학습 효과가 아님이 확인됨. config에서 제거.
- **bin 감독 손실 신설** (`losses.bin_ce_loss`, `--bin-weight`): 감독 없는 bins=64는
  **퇴화**해 있었다(엔트로피 0.77, 경계 0.776 vs 평탄 0.773으로 구별 없음, 99.9% 픽셀에서
  최빈 bin 질량 <50%). 감독 후 0.408/0.354로 경계>평탄 관계가 처음 성립하고 실촬 AbsRel
  −6.5%. 최종 조합 = v8 구조 + teacher off + bin CE = `work_dirs/arm2-binloss`
  (실촬 AbsRel **0.1386** / $\delta_1$ **0.8472** @ active 17.9%).
- **정성 시각화** `scripts/viz.py` → `outputs/v8/`, `outputs/arm2-binloss/` (RGB|예측|GT 10장).
  전경 물체가 안 살아난다. 원인은 그리드 상한이 아님 — 모델(0.2511/0.7446)이 **32px 블록
  상수 오라클(0.1459/0.8141)보다도 나쁨**. 표현 품질이 다음 병목.
- **남은 판단**: arm2는 실촬에서만 v7을 이긴다. 합성 RMSE 21% 열위(38.49 vs 31.77),
  t-delta·TCE도 열위 — 원거리 bin 해상도 부족 후보.

**다음 후보** (사용자 논의 중): ① frozen DA v2 특징 → 우리 디코더 프로브로 백본 병목 여부
판정 ② v4 특징 증류 실패 원인 규명(단일 블록·cosine only였음) ③ 그래도 필요하면 자기지도
사전학습(IN-1k 지도 사전학습은 dense task 정렬·비용·시간축 블록 미사용 때문에 후순위).

## 다음 액션

1. v5(용량 증가, dim=384, ~16h 예상) 완주 대기 — 완주되면 holdout eval 재실행, v3/v4와 비교
2. v5도 효과 없으면: distill 위치를 초기 임베딩 단으로 옮기거나, 데이터 스케일(더 많은 step/데이터) 쪽 검토
3. `tartanair_v2/Hospital` 일부 파일의 간헐적 read 실패(추정: `/archive` 동시접근 경합) — 재발하면 `num_workers` 낮추거나 원인 더 깊이 조사
4. 3단계(시스템: Triton 희소 커널, Jetson Orin 실측) 착수 여부 결정 — Jetson 실기기 접근 여부 확인 필요
5. `baselines`/`vda` conda env는 재현용으로 남겨둠 — 정리하려면 `conda env remove -n baselines`/`-n vda`
