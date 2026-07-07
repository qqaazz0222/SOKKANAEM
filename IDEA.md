# SOKKANAEM

**S**patial-temporal **O**ptimized **K**ey-patch **K**ernel for **A**daptive **N**etwork **A**rchitecture and **E**fficient **M**amba

> 프레임 간 변화가 발생한 패치만 '솎아내어' 연산하는, 실시간 비디오 깊이 추정 프레임워크

---

## 1. 문제 정의 (Problem Statement)

### 1.1 기존의 한계

* 딥러닝 기반 비디오 깊이 추정(Video Depth Estimation)은 매 프레임 전체 영역(모든 픽셀/패치)을 독립적으로 재연산한다. Transformer 기반 모델(DPT, Depth Anything 계열)은 토큰 수 $N$에 대해 $O(N^2)$ attention 비용을 매 프레임 지불한다.
* 프레임 단위 독립 추론은 시간적 일관성(Temporal Consistency)도 보장하지 못해, 깊이 값이 프레임 간 튀는 플리커(flicker) 현상이 발생한다. 이를 후처리(NVDS 등)로 보정하면 지연(latency)이 추가된다.

### 1.2 핵심 관찰

* 비디오는 인접 프레임 간 **시공간적 중복성(Spatiotemporal Redundancy)** 이 매우 높다. 고정 카메라(CCTV) 환경에서 프레임 간 실제로 변하는 패치 비율은 통상 5–20% 수준이다.
* 즉, 정적 배경을 매 프레임 재계산하는 것은 순수한 낭비다. **"변한 것만 다시 본다"** 는 원칙을 아키텍처 레벨에서 강제하면 연산량을 변화율에 비례하도록 만들 수 있다.

### 1.3 왜 Mamba인가

* Transformer에서 토큰을 제거(pruning)하면 attention의 전역 컨텍스트가 깨지고, 제거된 위치의 피처를 별도 캐시로 관리해야 한다.
* Mamba(Selective SSM)는 순차 스캔으로 hidden state $h_t$를 갱신하는 $O(N)$ 모델이며, **이산화(discretization) 파라미터 $\Delta$가 "이 입력을 상태에 얼마나 반영할지"를 이미 제어하고 있다.** 여기에 마스크를 개입시키면 별도 모듈 없이 "상태 유지 = 연산 스킵"을 **수학적으로 정확하게(exact)** 구현할 수 있다. 이것이 본 프레임워크의 핵심 통찰이다 (§3.2).

---

## 2. 관련 연구 및 차별점 (Related Work & Positioning)

| 계열 | 대표 연구 | 한계 / SOKKANAEM과의 차이 |
|---|---|---|
| 단안 깊이 추정 | MiDaS, DPT, Depth Anything v1/v2 | 프레임 독립 추론. 시간 축 정보 미활용, 연산량 고정 |
| 비디오 깊이 추정 | NVDS, Video Depth Anything, DepthCrafter | 시간 일관성은 확보하나 전체 프레임 연산 유지. 실시간·에지 불가 |
| 토큰 감축 | ToMe, EViT, DynamicViT | 이미지 단일 프레임 내 감축. 프레임 간 중복성 미활용, dense 출력 복원 문제 |
| 효율적 비디오 추론 | Skip-Convolutions, DeltaCNN, Eventful Transformer | 변화 기반 스킵 개념은 공유. 단, CNN/Transformer 대상이며 상태 유지 메커니즘 부재 또는 캐시 관리가 복잡 |
| Vision Mamba | Vim, VMamba, VideoMamba | 전체 토큰 스캔. 입력 적응적(input-adaptive) 연산 스킵 없음 |

**차별점:** 변화 기반 스킵(Eventful Transformer 계열)과 SSM의 상태 유지 능력을 결합한 최초의 시도. 스킵된 패치의 정보는 캐시가 아니라 **Mamba hidden state 자체**가 보존하므로, 캐시 정합성 관리가 수식 레벨에서 공짜로 해결된다.

---

## 3. 방법론 (Method)

### 3.0 전체 파이프라인

```
Frame t ──> Patchify ──> Change Detector ──> Active Mask M_t ─┐
                │                                             ▼
                └──────> Patch Embedding ──> Masked Mamba Backbone (Δ-gating)
                                                    │
Feature Cache (static) ─────────── Fuse ◄──────────┘
                                    │
                                    ▼
                        Boundary Refinement Decoder ──> Dense Depth D_t
```

### 3.1 프론트엔드: 패치 레벨 변화 감지 (Change Detection)

* 프레임을 $16 \times 16$ 패치로 분할, 위치 $i$의 변화 점수:

$$s_i^{(t)} = \frac{\| P_i^{(t)} - P_i^{(t-1)} \|_2^2}{HW \cdot C}$$

* $s_i > \tau$ 이면 **Active (1)**, 아니면 **Static (0)** 으로 바이너리 마스크 $M_t \in \{0,1\}^N$ 생성.
* 안정화 기법 3종 (전부 픽셀 도메인 연산, 비용 무시 가능):
  1. **Hysteresis:** 활성/비활성 전환에 이중 임계값($\tau_{on} > \tau_{off}$) 적용 — 경계 패치의 마스크 플리커 방지.
  2. **Morphological Dilation:** 활성 영역을 1패치 팽창 — 움직이는 물체의 경계 누락 방지.
  3. **주기적 전체 갱신(Keyframe Refresh):** 매 $K$ 프레임(예: $K=30$)마다 전체 패치 강제 활성화 — 조명 변화 등 저속 드리프트 누적 차단.
* 조도 변화에 대한 강건성이 필요하면 MSE 대신 정규화된 코사인 유사도 또는 저해상도 피처 공간 거리로 교체 가능 (ablation 항목).

### 3.2 백본: $\Delta$-Gating을 통한 조건부 상태 제어 (핵심 기여)

Mamba의 ZOH(Zero-Order Hold) 이산화:

$$\bar{A}_i = \exp(\Delta_i A), \qquad \bar{B}_i = (\Delta_i A)^{-1}(\exp(\Delta_i A) - I) \cdot \Delta_i B$$

$$h_i = \bar{A}_i h_{i-1} + \bar{B}_i x_i$$

여기서 마스크를 $\Delta$에 직접 곱한다:

$$\tilde{\Delta}_i = M_i \cdot \Delta_i$$

* **Static ($M_i = 0$):** $\tilde{\Delta}_i = 0 \Rightarrow \bar{A}_i = I,\ \bar{B}_i = 0 \Rightarrow h_i = h_{i-1}$.
  → hidden state **정확한 항등 복사**. 근사가 아니라 이산화 수식의 극한값 그 자체이므로, 스킵으로 인한 상태 오염이 원천적으로 없다. 해당 토큰의 SSM 행렬 연산·게이트 연산은 커널에서 우회(bypass)한다.
* **Active ($M_i = 1$):** 표준 Mamba 갱신. 변화 패치만 hidden state에 새 정보를 기입.
* **학습:** $M_i$는 비미분 이진값이므로 Straight-Through Estimator(STE)로 역전파. 학습 시 랜덤 마스크 비율 스케줄링(0%→80% 스킵)으로 다양한 희소성에 강건하게 학습.

**상태 배치(State Layout):** 시간 축 상태 유지가 목적이므로, 각 패치 위치가 프레임을 가로질러 자신의 temporal state를 갖는 구조가 필요하다. 설계안:

* **T-Mamba(시간 축):** 패치 위치별로 프레임 축을 따라 스캔 — hidden state가 "그 위치의 시각적 기억"이 됨. Static 패치는 이 축에서 상태 복사.
* **S-Mamba(공간 축):** 프레임 내 2D 스캔(VMamba식 cross-scan)으로 공간 컨텍스트 혼합. Active 패치만 통과, Static 패치는 캐시된 출력 피처 재사용.
* 두 블록을 교차 적층(interleave). 시간 축이 상태를 유지하므로 공간 축의 피처 캐시는 stale해도 T-Mamba가 보정한다.

### 3.3 디코더: 피처 융합 및 경계 보정 (Boundary Refinement)

* Static 패치의 캐시 피처 + Active 패치의 갱신 피처를 위치대로 결합(scatter).
* 패치 경계 불연속(blocking artifact) 억제: 경량 $3\times3$ conv 2층 + RGB guided filter. 무거운 refinement 네트워크는 배제 — 전체 연산 절감 효과를 잠식하지 않도록 디코더 예산은 백본의 10% 이하로 제한.
* 손실 함수: scale-invariant depth loss + gradient matching loss(경계) + **temporal consistency loss** ($\| D_t - D_{t-1} \|$ on static regions — 정적 영역은 깊이도 불변이어야 함).

### 3.4 학습 전략

**2단계 커리큘럼 (고정 카메라 타깃):**

1. **Stage 1 — 지도 학습 (TIMo):** 고정형 ToF 실내 모니터링 데이터셋 TIMo의 실측 depth GT로 지도 학습. 배포 도메인(고정 CCTV)과 일치하는 유일한 실측 GT 소스.
2. **Stage 2 — Pseudo-GT Distillation (VIRAT):** depth GT가 없는 대규모 감시 비디오 VIRAT에 대해, SOTA 깊이 추정 모델(예: Depth Anything v2)로 의사 깊이 지도를 오프라인 생성 후 가상 GT로 distillation. 스케일 모호성은 scale-invariant loss가 흡수. 장면 다양성 확대 + 실 감시 도메인 적응.

**공통 기법:**

* **Teacher-Student 증류:** 마스크 없는 full-compute 모델(teacher)의 dense feature/depth를 masked 모델(student)이 모방 — 스킵으로 인한 성능 저하 최소화.
* **Budget Loss:** $\mathcal{L}_{budget} = \lambda \cdot \max(0, \bar{M} - \rho)$ — 평균 활성 비율 $\bar{M}$이 목표 예산 $\rho$를 넘지 않도록 유도 (학습형 임계값 사용 시).
* 사전학습 Depth Anything encoder에서 Mamba 백본으로 증류 후 fine-tuning하면 학습 비용 절감 가능.

### 3.5 이동 카메라 확장: Low-Res GMC + Feature-level Temporal Gating 하이브리드

센서 데이터가 없는 순수 비디오 입력(Pure RGB) 환경에서, 카메라가 움직일 때 발생하는 전처리 병목을 제거하고 실시간성을 사수하기 위한 하이브리드 구조.

**목적 (Objectives):**

* **이동형 카메라(Ego-motion)로의 도메인 확장:** 고정 카메라(CCTV)를 넘어 블랙박스, 드론, 로봇 등 카메라 자체가 움직이는 환경에서도 변화 기반 스킵이 무력화되지 않도록 범용성 확보.
* **'전처리 병목' 해결:** 카메라 움직임 보정에 무거운 딥러닝 Optical Flow나 고해상도 특징점 연산을 쓰면 배보다 배꼽이 커진다. 전처리 연산량을 최소화하여 SOKKANAEM 고유의 가속화 이득을 100% 보존.
* **하드웨어 의존성 탈피:** IMU 등 외부 센서 없이 단일 RGB 스트림만으로 구동 — 어떤 영상이 입력되어도 즉각 적용 가능한 소프트웨어 독립성.

**구동 메커니즘:**

```
[Frame t-1, t (원본)]
        │
        ├───> [1단계: Low-Res GMC] (1/4 해상도 축소 ──> 초고속 거시적 정렬 matrix 계산)
        │            │
        ▼            ▼
[Mamba 백본 진입] ──> [2단계: Feature-level Gating] (초기 임베딩 피처 차이 비교로 정적 패치 최종 판정)
        │
        ▼
[Selective Mamba Scan] (솎아진 핵심 활성 패치만 업데이트, 정적 패치는 이전 Hidden State 복사)
```

**1단계 — Low-Res GMC (거시적 전역 모션 보정):** 입력을 원본의 1/4 이하(예: $128 \times 128$)로 대폭 축소한 뒤, 극소수(30~50개)의 핵심 특징점만 추출하여 RANSAC 기반 호모그래피(Homography) 행렬을 계산. 이 행렬로 프레임 $t-1$을 현재 시점으로 워핑(Warping)하여 정렬한다. 저해상도라 연산 시간이 1~2ms 내외로 극히 짧으면서도, 카메라 이동/회전으로 인한 화면 전체의 거시적 픽셀 흐름을 상쇄한다. 추적 실패 시 항등 변환으로 폴백 — 활성 패치가 늘어날 뿐 오판은 없다.

**2단계 — Feature-level Temporal Gating (미시적 패치 솎아내기):** 1단계에서 정렬된 프레임을 Mamba 백본의 가벼운 초기 인코더(Linear Projection = patch embedding)에 통과시켜, 피처 맵 $F_t$와 워핑된 $F_{t-1}$의 패치별 차이(상대 L1 Norm)로 최종 Active Patch Mask를 확정한다. 맘바 내부의 초기 피처를 그대로 재활용하므로 전처리 오버헤드가 없고, 1단계의 미세한 정렬 오차나 조명 노이즈를 고차원 피처 레벨에서 유연하게 걸러내어 진짜 움직이는 객체가 포함된 패치만 정확하게 솎아낸다. 임계값 처리(hysteresis + dilation + keyframe refresh)는 §3.1의 게이트 로직을 그대로 공유하며, 점수 소스만 픽셀 MSE에서 피처 상대 L1으로 교체된다.

**기대 효과 및 학술적 의의:**

* **FLOPs 절감율 유지:** 카메라가 움직여도 배경·카메라 모션과 일치하는 정적 패치를 스킵하므로 이동형 영상에서도 50% 이상의 연산량 감소 및 FPS 향상 기대.
* **노이즈에 강인한 맘바 구조 입증 (Novelty):** 전처리 단에서 고의로 해상도를 낮춰 속도를 챙기는 대신 미세한 정렬 오차가 발생할 수 있지만, Mamba 고유의 순차적 Hidden State 전파가 이러한 노이즈를 스스로 흡수하며 매끄러운 깊이를 추정한다 — 논문의 핵심 기여로 어필 가능.

---

## 4. 실험 계획 (Evaluation Plan)

### 4.1 데이터셋

* **고정 카메라(주 타깃):**
  * **TIMo (Time-of-Flight Indoor Monitoring)** — 고정형 ToF 카메라 기반 실내 모니터링 RGB-Depth paired 데이터셋. 본 연구의 배포 시나리오(고정 CCTV 모니터링)와 도메인이 정확히 일치하며, 실측 depth GT를 제공하므로 **1단계 지도 학습(supervised)의 주 데이터셋**으로 사용.
  * **VIRAT Video Dataset** — 고정형 감시 카메라 대규모 비디오. Depth GT 부재 → **SOTA급 깊이 추정 모델(예: Depth Anything v2)을 교사 모델로 삼아 의사 깊이 지도(pseudo depth GT)를 오프라인 구축**하고, 이를 가상 GT로 한 distillation으로 **2단계 학습** 진행 (§3.4). TIMo 학습 완료 후 착수. 실 감시 도메인의 스케일·장면 다양성 확보 목적.
  * 자체 CCTV 시퀀스, ScanNet 정적 구간 — 보조.
* **일반 비디오:** KITTI(주행), NYUv2 video, Bonn RGB-D Dynamic, Sintel — 카메라 모션 존재 환경에서의 성능 하한 확인.

### 4.2 지표

* **정확도:** AbsRel, RMSE, $\delta < 1.25$.
* **시간 일관성:** TAE(Temporal Alignment Error), OPW(Optical-flow-based Warping error).
* **효율:** FLOPs(변화율 대비 곡선), FPS/latency — RTX 4090 + **Jetson Orin(에지 타깃)** 실측. 이론 FLOPs가 아닌 wall-clock 필수(희소 연산은 GPU 활용률이 관건).

### 4.3 베이스라인

Depth Anything v2(프레임 독립), Video Depth Anything, NVDS, VideoMamba(마스크 없는 동일 백본 = 본 연구의 upper-bound teacher).

### 4.4 Ablation

| 항목 | 변수 |
|---|---|
| 임계값 $\tau$ | 정확도–연산량 trade-off 곡선 (핵심 그림) |
| 패치 크기 | 8 / 16 / 32 |
| 게이팅 위치 | $\Delta$-gating vs 입력 토큰 drop vs 출력 캐시만 |
| 변화 감지기 | MSE vs cosine vs feature-space |
| Keyframe 주기 $K$ | 드리프트 vs 연산량 |
| Refinement | 없음 vs conv vs guided filter |
| 마스크 분포 | i.i.d. 랜덤 마스크 학습만 vs detector/GMC-driven 마스크로 fine-tune (학습-배포 분포 일치) |

### 4.5 PoC 결과 — Virtual KITTI 2 (2026-07-07)

PoC 모델(dim 기본, size 128, clip 4)을 vkitti2 전체(100 시퀀스, 42,520 프레임, 21,120 클립)에서 30k step 학습
(`configs/vkitti2.toml`, RTX 4090 ~111분, 최종 ckpt 11MB). 평가는 5,260 클립 중 100개 샘플.
전체 수치는 `work_dirs/vkitti2/eval.txt`.

**임계값 $\tau$ sweep — pixel gating (기본):**

| $\tau_{on}$ | active% | AbsRel | RMSE | $\delta_1$ | t-delta |
|---|---|---|---|---|---|
| 0 (풀 연산) | 100.0 | 0.2054 | 12.93 | 0.684 | 0.1033 |
| 0.05 | 68.1 | 0.2054 | 12.93 | 0.684 | 0.1002 |
| 0.1 | 44.6 | 0.2057 | 12.93 | 0.684 | 0.0962 |

**GMC + feature-level gating (§3.5, ego-motion 데이터에서):**

| $\tau_{on}$ | active% | AbsRel | RMSE | $\delta_1$ | t-delta |
|---|---|---|---|---|---|
| 0.2 | 54.1 | 0.2060 | 12.94 | 0.684 | 0.0985 |
| 0.4 | 23.7 | 0.2068 | 12.94 | 0.682 | 0.0918 |
| 0.8 | 2.3 | 0.2098 | 12.93 | 0.677 | 0.0762 |

**판정:**

1. **Go** — 로드맵 기준(스킵 50%에서 AbsRel 열화 5% 이내)을 크게 상회: active 44.6%(스킵 55%)에서
   AbsRel 열화 **+0.15% (상대)**. 주행(전역 모션) 데이터에서조차 스킵 비용이 사실상 0.
2. **§3.5 하이브리드 실증** — pixel gating의 스킵 바닥(~active 45%)을 GMC+feature gating이 돌파:
   active 23.7%에서 AbsRel +0.7%(상대), 극단(active 2.3%)에서도 $\delta_1$ 손실 0.007.
   호모그래피 정렬이 ego-motion 유발 변화를 흡수해 잔차 변화만 감지됨을 확인.
3. **스킵 = 시간 안정성 향상** — t-delta가 스킵과 단조 개선(0.1033 → 0.0762).
   $\Delta$-gating의 정확한 상태 복사가 후처리 없는 시간 일관성을 준다는 기여 2의 직접 증거.

한계: PoC 해상도(128px, 정사각 resize)와 30k step 제약으로 절대 정확도($\delta_1$ 0.68)는 낮음 —
trade-off 곡선의 형태가 검증 대상이며, 절대 성능은 본 학습(§7)에서 확보.

**Wall-clock 실측 (RTX 4090, 128px, batch 1, 스트리밍 300프레임):**

| 모드 | active% | ms/frame | FPS |
|---|---|---|---|
| pixel gating, $\tau$=0 (풀 연산) | 100.0 | 2.69 | 372 |
| pixel gating, $\tau$=0.1 | 56.2 | 2.68 | 373 |
| GMC, $\tau$=0.4 | 33.4 | 3.50 | 286 |
| GMC, $\tau$=0.8 | 12.1 | 3.47 | 288 |

- 최초 프로파일에서 spatial 블록의 Python 순차 스캔이 지연의 95–99%로 확인 →
  **청크 segment-sum 스캔**(Mamba-2 스타일, 수식 동일·수치 안정)으로 교체.
  104 → 372 FPS (**3.6×**), 512px에서는 137 → 19 ms/frame (**7×**). 학습 forward도 동일 경로라 함께 가속.
- GMC 오버헤드 +0.8 ms/frame — §3.5의 1–2ms 예산 내 (128px라 무축소 경로).

**Static-patch 출력 캐싱 (phase 3, `--spatial-cache`):** spatial 블록에서 active 패치만
gather→scan→scatter, static 패치는 직전 출력 토큰 재사용 (키프레임마다 전체 리프레시).
temporal $\Delta$-gating과 달리 **근사** — sparse scan이 static 패치의 문맥 기여를 생략.

| 지점 (GMC, vkitti2) | active% | FPS 128px | FPS 512px | AbsRel | t-delta |
|---|---|---|---|---|---|
| 캐시 없음, $\tau$=0.8 | ~12–33 | 292 | 40.2 | 0.2098 | 0.076 |
| 캐시 + $\tau$=0.8 | ~12–33 | **516** | **70.7** | 0.2185 | **0.068** |
| 캐시 + $\tau$=0.2–0.4 (중간 active) | 54–72 | 316–374 | 34–41 | 0.235–0.245 | 0.54–0.79 |

- **저-active 운영점(≤15%)에서 유효**: 1.8× 속도, AbsRel +4%(상대), t-delta는 오히려 개선
  (frozen 출력 = 플리커 0). **중간 active%(40–80%)는 U자형 악화** — fresh/stale 토큰 경계의
  문맥 불일치로 t-delta 6× 폭증. 속도 이득도 (1−active%)에 비례라 이 구간은 켤 이유 없음.
- vkitti2(주행)는 캐시에 최악 조건 — 주 타깃(고정 CCTV, active 5–10% 상시)이 정확히
  캐시가 무손실·최고속인 구간. 고정 카메라 데이터 확보 후 재검증.
- 학습은 항상 풀 spatial (inference-only 최적화). 잔여 개선 여지: 캐시 사용 시 keyframe 주기 단축.

**CUDA graph 캡처 (`enable_cuda_graphs()`):** full-compute 경로(embed→blocks→decoder)를
순수 텐서 함수로 분리, `torch.compile(mode="reduce-overhead")`로 그래프 캡처. Python
오케스트레이션·커널 런치 오버헤드 제거 — 128px 370→**1075 FPS** (2.9×), 256px 154→**366 FPS**
(2.4×), 512px는 compute-bound라 무이득. 수치: inductor 재배열로 상대 ~1e-3 노이즈, eval 지표
소수 4자리 동일 확인. sparse 캐시 경로는 동적 shape라 eager 유지 (두 최적화는 상보적:
그래프=풀연산·키프레임, 캐시=정적 구간).

**경량 모델 대비 (256px급, fp16+compile, batch 1, 4090):** DA V2 Small(24.8M) 380 FPS,
DPT-SwinV2-Tiny(40.9M) 371 FPS, DA V1 Small(24.8M) 280 FPS vs 본 모델(2.8M) 풀연산+그래프
**366 FPS**, 정적 스트림+캐시 **460 FPS**, 128px 그래프 **1075 FPS**. 단일 GPU FPS는 동급 —
차별점은 연산량∝변화율(에지 전력·멀티스트림)과 내장 시간 일관성. Jetson 실측이 결정적.

**Ablation — eval-only 항목 (§4.4, vkitti2 32프레임 클립, 60클립):**

- **Keyframe 주기 $K$**: $K$=5→1000에서 AbsRel 0.2888→0.2892, $\delta_1$ 변화 <0.001 —
  **32프레임 범위 드리프트 실질 0**. $\Delta$-gating 상태 복사 + hysteresis만으로 안정 유지,
  $K$=10(config)은 보수적, $K$=30이면 active 2.7%p 추가 절감 공짜. 수백 프레임 장기 스트림은 미검증.
- **변화 감지기 MSE vs cosine** (iso-active): MSE $\tau$=0.05 → active 69.2%, AbsRel 0.2890 /
  cosine $\tau$=0.3 → active 59.8%, AbsRel 0.2897. **동급 — 감지기 선택 둔감**, 더 싼 MSE 유지.
  cosine은 점수 분포가 압축돼 $\tau$ 스케일만 다름 (0.3 ≈ MSE 0.05).
- 학습 필요 항목(마스크 분포 3-arm: no-skip / detector-driven fine-tune / max_skip 0.8)은 진행 중.
- 패치 크기·게이팅 위치 변형은 모델 개조 필요로 보류 (Decoder patch 16 고정).
- (프로토콜 주: 32프레임 클립은 median scaling 1회/클립이라 8프레임 대비 절대 수치 낮음 —
  항목 내 상대 비교만 유효.)

---

## 5. 예상 리스크 및 대응 (Risks & Mitigations)

| 리스크 | 내용 | 대응 |
|---|---|---|
| **카메라 모션** | 이동 카메라에서는 전 패치가 활성화되어 이득 소멸 | (1) 주 타깃을 고정 카메라로 명시적 포지셔닝. (2) 확장: Low-Res GMC + feature-level gating 하이브리드 (§3.5) — 호모그래피 정렬 후 피처 레벨 잔차 변화만 감지 |
| **불규칙 희소성의 GPU 비효율** | 이론 FLOPs 절감이 실제 속도로 안 이어질 수 있음 | 패치 마스크는 블록 단위 희소성이므로 gather–compute–scatter로 dense 커널 재사용. Mamba 스캔은 선형 순차 구조라 Transformer 대비 스킵 커널 구현이 단순 |
| **저속 변화 드리프트** | 임계값 이하의 미세 변화 누적(조명 등) | Keyframe refresh + 변화 점수 누적 카운터(sub-threshold 누적치가 $\tau$ 초과 시 활성화) |
| **경계 아티팩트** | 활성/정적 경계의 깊이 불연속 | Dilation + refinement decoder + gradient loss (§3.1, §3.3) |
| **성능 상한** | teacher(full-compute) 대비 정확도 손실 | Budget–accuracy 곡선으로 운영점 선택 가능하게 제시. "정확도 1% 손실로 FLOPs 70% 절감" 형태의 주장 목표 |

---

## 6. 기여도 요약 (Contributions)

1. **$\Delta$-Gating:** Mamba 이산화 수식에 변화 마스크를 개입시켜 "연산 스킵 = 정확한 상태 유지"를 수식 레벨에서 달성하는 최초의 조건부 SSM 게이팅 기법.
2. **시간 일관성의 구조적 확보:** hidden state가 프레임 간 시각적 기억을 유지하므로, 후처리 없이 플리커 없는 비디오 깊이 추정 — 스킵할수록 오히려 정적 영역의 깊이가 안정.
3. **입력 적응적 연산량:** 연산 비용이 장면 변화율에 비례. 고정 카메라 환경에서 FLOPs 대폭 절감, 에지 디바이스 실시간 3D 인지(가상 공간 모니터링, 실시간 공간 복원) 병목 해소.
4. **센서리스 이동 카메라 확장 (§3.5):** Low-Res GMC + feature-level temporal gating 하이브리드로, IMU 없이 순수 RGB만으로 ego-motion 환경에서도 스킵 이득을 유지. 저해상도 정렬의 미세 오차는 Mamba hidden state 전파가 흡수 — 노이즈 강인성 자체가 기여점.

---

## 7. 로드맵 (Roadmap)

1. **PoC (4주):** 변화 감지기 + $\Delta$-gating을 기존 Vision Mamba(Vim/VMamba) 체크포인트에 주입, ScanNet 정적 구간에서 스킵 비율–정확도 곡선 확인. *Go/No-Go 지점: 스킵 50%에서 AbsRel 열화 5% 이내.* → **✅ Go 달성 (vkitti2, §4.5): 스킵 55%에서 AbsRel 열화 +0.15%.**
2. **본 학습 (8주):** T/S-Mamba 교차 백본. Stage 1: TIMo 지도 학습 → Stage 2: VIRAT pseudo-GT distillation (§3.4). 전체 벤치마크.
3. **시스템 (4주):** 블록 희소 커널(Triton) 구현, Jetson Orin 실측, 데모(CCTV 실시간 깊이 스트림).
4. **논문화:** 타깃 — CVPR/ICCV (efficiency track) 또는 실시간 시스템 강조 시 CoRL/IROS.
