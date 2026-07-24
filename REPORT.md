# REPORT

실험 결과 기술 보고서. 설계 배경·수식 유도는 [IDEA.md](IDEA.md), 차별점 요약은 [NOVELTY.md](NOVELTY.md),
현재 작업 상태는 [PROGRESS.md](PROGRESS.md) 참조. 원본 로그: `work_dirs/*/eval.txt`, `work_dirs/*/train.log`.

## 1. 방법 요약

프레임을 16×16 패치로 분할 → 변화 감지(MSE, hysteresis+dilation+keyframe) → 마스크 $M_t$를 Mamba
이산화 파라미터에 곱함($\tilde\Delta = M\cdot\Delta$) → $M=0$인 패치는 hidden state가 정확히
이전 값으로 복사됨(연산 스킵). T-Mamba(시간축)/S-Mamba(공간축) 교차 적층 백본 + 경량 conv 디코더.
전체 유도는 IDEA.md §3.

## 2. PoC 실험 설정

- 모델: `SOKKANAEM` 기본 dim, size 128, clip_len 4
- 데이터: Virtual KITTI 2 전체 (100 시퀀스, 42,520 프레임, 21,120 클립), holdout=Scene06
- 학습: `configs/vkitti2.toml`, 30k step, RTX 4090 ~111분, ckpt 11MB
- 평가: 5,260 val 클립 중 100 샘플 (§4.5 표), 별도 ablation은 32프레임 클립·660개 프로토콜(`work_dirs/vkitti2/eval.txt`) — **절대 수치가 다른 표와 다름**, median scaling 방식 차이 때문 (클립당 1회 스케일링, 프레임 수에 따라 절대값 이동). 항목 내 상대 비교만 유효.

## 3. 결과

### 3.1 Go/No-Go — pixel gating (기준 판정)

| $\tau_{on}$ | active% | AbsRel | RMSE | $\delta_1$ | t-delta |
|---|---|---|---|---|---|
| 0 (풀 연산) | 100.0 | 0.2054 | 12.93 | 0.684 | 0.1033 |
| 0.05 | 68.1 | 0.2054 | 12.93 | 0.684 | 0.1002 |
| 0.1 | 44.6 | 0.2057 | 12.93 | 0.684 | 0.0962 |

**판정: Go.** 로드맵 기준(스킵 50%에서 AbsRel 열화 5% 이내)을 크게 상회 — 스킵 55%(active 44.6%)에서
AbsRel 열화 +0.15%(상대). 주행(전역 모션) 데이터에서조차 스킵 비용이 사실상 0.
스킵률↑에 따라 t-delta(시간 불안정성)도 단조 개선(0.1033→0.0962) — Δ-gating의 정확한 상태 복사가
후처리 없는 시간 일관성을 준다는 근거.

### 3.2 GMC + Feature-level Gating (ego-motion 확장, §3.5)

| $\tau_{on}$ | active% | AbsRel | RMSE | $\delta_1$ | t-delta |
|---|---|---|---|---|---|
| 0.2 | 54.1 | 0.2060 | 12.94 | 0.684 | 0.0985 |
| 0.4 | 23.7 | 0.2068 | 12.94 | 0.682 | 0.0918 |
| 0.8 | 2.3 | 0.2098 | 12.93 | 0.677 | 0.0762 |

Pixel gating의 스킵 바닥(~active 45%)을 GMC+feature gating이 돌파 — active 23.7%에서 AbsRel +0.7%,
극단(active 2.3%)에서도 $\delta_1$ 손실 0.007. 호모그래피 정렬이 ego-motion 유발 변화를 흡수해
잔차 변화만 감지됨을 확인. GMC 자체 오버헤드는 +0.8ms/frame (§3.3 wall-clock 참조).

### 3.3 Wall-clock 성능 (RTX 4090)

**128px, batch 1, 스트리밍 300프레임:**

| 모드 | active% | ms/frame | FPS |
|---|---|---|---|
| pixel gating, $\tau$=0 (풀 연산) | 100.0 | 2.69 | 372 |
| pixel gating, $\tau$=0.1 | 56.2 | 2.68 | 373 |
| GMC, $\tau$=0.4 | 33.4 | 3.50 | 286 |
| GMC, $\tau$=0.8 | 12.1 | 3.47 | 288 |

- 최초 프로파일: spatial 블록 Python 순차 스캔이 지연의 95–99%. **청크 segment-sum 스캔**(Mamba-2
  스타일, 수식 동일)으로 교체 → 104 → 372 FPS (**3.6×**, 128px), 512px에서 137→19ms/frame (**7×**).
- **CUDA graph 캡처**(`enable_cuda_graphs()`, full-compute 경로만): 128px 370→**1075 FPS**(2.9×),
  256px 154→**366 FPS**(2.4×). 512px는 compute-bound라 무이득. eval 지표 소수 4자리까지 동일 확인(수치 안정성 검증됨).
- **Static-patch 출력 캐싱**(`--spatial-cache`, phase 3, opt-in, **근사** — temporal Δ-gating과 달리
  static 패치의 문맥 기여를 생략):

  | 지점 (GMC, vkitti2) | active% | FPS 128px | FPS 512px | AbsRel | t-delta |
  |---|---|---|---|---|---|
  | 캐시 없음, $\tau$=0.8 | ~12–33 | 292 | 40.2 | 0.2098 | 0.076 |
  | 캐시 + $\tau$=0.8 | ~12–33 | **516** | **70.7** | 0.2185 | **0.068** |
  | 캐시 + $\tau$=0.2–0.4 (중간 active) | 54–72 | 316–374 | 34–41 | 0.235–0.245 | 0.54–0.79 |

  저-active(≤15%) 구간에서만 유효(1.8× 속도, AbsRel +4%, t-delta 오히려 개선). 중간 active%(40–80%)는
  fresh/stale 토큰 경계 문맥 불일치로 t-delta 6× 폭증 — **U자형 악화**, 이 구간은 켜지 않음. 주 타깃인
  고정 CCTV(active 5–10% 상시)가 정확히 캐시 최적 구간. 학습은 항상 풀 spatial(inference-only 최적화).

### 3.4 경량 모델 대비 (256px급, fp16+compile, batch 1, RTX 4090)

| 모델 | 파라미터 | FPS |
|---|---|---|
| DA V2 Small | 24.8M | 380 |
| DPT-SwinV2-Tiny | 40.9M | 371 |
| DA V1 Small | 24.8M | 280 |
| **본 모델 (풀연산+그래프)** | **2.8M** | **366** |
| 본 모델 (정적 스트림+캐시) | 2.8M | 460 |
| 본 모델 (128px 그래프) | 2.8M | 1075 |

단일 GPU FPS는 동급 — 차별점은 연산량이 변화율에 비례한다는 점(에지 전력/멀티스트림 이득)과
내장 시간 일관성. Jetson 실측은 미실시(§5 한계 참조).

### 3.5 Ablation

**Keyframe 주기 $K$ / 변화 감지기 (eval-only, 32프레임 클립, 60클립, `work_dirs/vkitti2/eval.txt`):**

| 항목 | active% | AbsRel | $\delta_1$ | t-delta |
|---|---|---|---|---|
| K=5 | 73.7 | 0.2888 | 0.5338 | 0.1185 |
| K=10 (config 기본) | 69.2 | 0.2890 | 0.5332 | 0.1178 |
| K=30 | 66.5 | 0.2891 | 0.5332 | 0.1173 |
| K=1000 | 65.1 | 0.2892 | 0.5331 | 0.1166 |
| cosine $\tau$=0.3 (iso-active, MSE 0.05 대응) | 59.8 | 0.2897 | 0.5313 | 0.1180 |

K=5→1000 구간 AbsRel 변화 <0.001 — 32프레임 범위 드리프트 실질 0. MSE vs cosine도 동급(감지기 선택 둔감,
싼 MSE 유지 결정). K=10은 보수적, K=30이면 active 2.7%p 추가 절감 공짜.

**장기 스트림 드리프트** (270프레임×4 시퀀스, Scene06 스트리밍): keyframe 완전히 꺼도($K=\infty$)
late-frame AbsRel이 early 대비 **−3.8%(개선)** — 저속 드리프트 리스크(IDEA.md §5) vkitti2 스케일에서 미발현.
스트리밍 실측 active 20–28%($\tau$=0.05) vs 클립 평가 68% — **클립 평가가 스킵률을 심하게 과소평가**함(클립마다
keyframe 리셋). 실배포 이득은 클립 지표보다 큼.

**마스크 분포 3-arm** (각 30k step, Scene06 홀드아웃, $\tau$=0.05, `work_dirs/abl-*/eval.txt` 실측):

| 학습 마스크 | AbsRel | $\delta_1$ | t-delta |
|---|---|---|---|
| **iid random, max_skip 0.5 (채택)** | **0.3272** | **0.4271** | 0.0932 |
| 없음 (max_skip 0) | 0.3326 | 0.4290 | **0.9312** |
| detector-driven 10k fine-tune | 0.3614 | 0.4173 | 0.2732 |
| iid random, max_skip 0.8 | 0.3952 | 0.3613 | 0.0783 |

결론: (1) 마스크 학습의 진짜 기여는 시간 안정성(없으면 t-delta 10×, 정확도는 Δ-gating 수식이 그냥 보장).
(2) 과도한 스킵 학습(0.8)은 gradient 부족으로 정확도 붕괴. (3) "학습-배포 분포 일치" 가설 **기각**
— detector 마스크(공간 상관 blob)가 iid 무작위성보다 못함, iid 자체가 regularizer 역할.
**본 학습 레시피로 iid random @ max_skip 0.5 확정.**

## 4. 본 학습 (main.toml) 현황

- 데이터: vkitti2 + tartanair2 + pointodyssey 혼합, 256px, 100k step 목표
- **현재 진행: 7,500 / 100,000 step에서 정지** (`work_dirs/main/train.log`, 마지막 기록 2026-07-14 15:42)
- loss 0.2~3.7 사이 변동 (스텝별 로그값, 수렴 추세는 스무딩 없이 판단 불가 — 재개 후 곡선 재확인 필요)
- 재개 전 상태 점검: [PROGRESS.md](PROGRESS.md) 참조 (데이터셋 경로 버그, checkpoint 포맷 변경)
- 홀드아웃 평가 미실시 (100k step 미완주)

## 5. 한계 및 미검증 사항

- PoC 해상도(128px, 정사각 resize) + 30k step 제약으로 절대 정확도($\delta_1$ 0.68)는 낮음 — trade-off
  곡선의 **형태**가 검증 대상이었고, 절대 성능은 본 학습(§4)에서 확보 예정.
- §3.5 GMC+feature gating은 vkitti2(합성, clean geometry)에서만 실증 — 실 노이즈 있는 ego-motion 영상 미검증.
- Jetson Orin 실측 미실시 (로드맵 3단계, 미착수).
- 수백 프레임 초장기 스트림 드리프트는 270프레임 범위까지만 확인.
- 패치 크기(8/16/32), 게이팅 위치(Δ-gating vs 토큰 drop vs 출력 캐시만) ablation은 모델 개조 필요로 보류.
- 본 학습 완주 전이므로 §4.2 전체 지표(TAE, OPW, 베이스라인 대비 정량 비교)는 아직 없음.
