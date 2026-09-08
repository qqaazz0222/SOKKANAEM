# NOVELTY

SOKKANAEM이 기존 연구 대비 실제로 새로운 부분만 추림. 배경·실험은 [IDEA.md](IDEA.md),
진행 상황은 [PROGRESS.md](PROGRESS.md), 수치는 [REPORT.md](REPORT.md) 참조.

> **2026-09-06 개정.** 수학·구현 일치성과 주장 범위를 추가 정정했다. 아래 수치는 과거 개발
> 결과이며 미사용 최종 테스트의 결과가 아니다. [논문화 준비](paper/PREPARATION.md)와
> [평가 계약](paper/PROTOCOL.md)이 현재 보고 기준이다. 선명도 라운드(REPORT §4.44~4.47)의 결과로
> 주장 축이 바뀌었다. 기존
> 프레이밍("연산량 ∝ 변화율")은 §4.11에서 이미 정직한 형태로 축소됐고, 이번 라운드는 **측정
> 기여**와 **재현 가능한 부정 결과**를 앞세울 근거를 만들었다. 아래는 각 주장에 붙는 수치와
> 그 강도를 함께 적는다.

## 핵심 통찰 (one-liner)

이산화 파라미터 Δ에 binary 변화 마스크를 곱하면 (`Δ̃ = M · Δ`), 비활성 패치의 temporal state
갱신은 항등이 된다. 실제 입력 항은 정확 ZOH가 아닌 `Δ Bx` 근사지만 zero-step identity는 유지된다.
[Skip RNN](https://arxiv.org/pdf/1708.06834)도 binary update/copy를 쓰므로 이 성질 자체의 최초성은
주장하지 않는다. 차별화 후보는 변화 검출–selective SSM–dense depth readout의 결합과 실측 trade-off다.
State 보존은 출력 보존이나 ungated dense와의 동등성이 아니다. Temporal/spatial output cache에는
별도의 근사가 존재한다.

## A. 모델 기여

| # | 주장 | 근거 | 강도 |
|---|---|---|---|
| A1 | **Δ-Gating의 zero-step identity** — `M=0 → Ā=I, B̄=0 → h_t=h_{t-1}`. Binary update/copy와 동등 | `tests/test_gating.py` (state copy, update/copy 동등성, readout 분리) | 성질 확인; 단독 신규성 아님 |
| A2 | **후처리 없는 원시 프레임 차이 감소** — t-delta 0.2455 대 DA3 1.80 · VDA 2.18 · DA v2 9.47 | REPORT §4.15의 과거 프로토콜 | 해당 실험의 관찰. 상수 예측도 낮출 수 있으며 모션 보정 지표(OPW/TCE) 우위는 아님 |
| A3 | **D1: 학습형 전체 해상도 복원 경로** — 1/2 해상도 pixel-shuffle residual + 전체 해상도 RGB detail, 둘 다 zero-init(초기 출력 불변), +0.024 GMAC · +0.8k 파라미터 | `tests/test_full_res.py` | 확정 |
| A4 | **25.5k 보정 head의 GT-scale-aligned 오차 감소** — median gauge에서 DA2 0.6451/1.1188 → 0.1097/0.0871 (TUM/Bonn), 정합 실패 0 | REPORT §4.45 | 실제 metric calibration 성공은 no-GT-fit 평가 필요; native 효율과 분리 |
| A5 | **입력 적응적 연산량**(정직한 형태) — 시간축 state 경로 비용이 변화율에 비례하고, 종단 절감은 dense embed/decoder와 공간축에 의해 제한된다 | REPORT §4.11, §4.24 | 축소된 형태로 확정 |

## B. 측정 기여 (이번 라운드 신설, 다른 연구에 그대로 이식 가능)

| # | 주장 | 근거 |
|---|---|---|
| B1 | **선명도는 단일 지표로 잴 수 없다** — gradient ratio는 노이즈로도, 링잉으로도 살 수 있다. 경계 P/R/F1(멀티 임계) + flat TV + overshoot 동시 판정이 필요하다 | 실제 arm이 grad_ratio 1.154로 게이트를 "통과"하면서 overshoot 0.347 · edge AbsRel 0.187로 실패. 또 다른 arm은 2.25까지 갔다 |
| B2 | **고정 GT 재표본화 진단** — patch-P 평균 풀링·bilinear 복원의 grad_ratio 0.2612 대 모델 0.4323 | REPORT §4.44. 다채널 학습 decoder의 최적화 문제가 아니므로 엄밀 상한이나 용량 비병목의 증거가 아님 |
| B3 | **지역별 유사한 오차 배율 관찰** — Bonn edge 1.55 · flat 1.75 · dynamic 1.64 · near 1.77 · far 1.79 | REPORT §4.44a. 시험한 지역 가중 손실이 효과적이지 않았음; 모든 지역 손실의 불가능성을 증명하지 않음 |
| B4 | **GT 정합과 metric 성능의 분리** — 같은 체크포인트가 median 0.110/0.087, scaleshift 0.310/0.060(실패 10클립) | REPORT §4.45. GT fit은 절대 calibration을 측정하지 않음; no-fit / median / relative-shape를 분리 |
| B5 | **선명도 임계는 (기준 모델, 채점 해상도) 쌍이다** — DA2 grad_ratio 0.7559@256, 0.6454@384 | REPORT §4.45 |
| B6 | **단측 경계 손실** — 경계에서 GT 기울기 도달까지만 보상하고 초과는 무보상, 평탄 영역은 gradient L1로 억제 | `boundary_location_loss`, grad_ratio 0.431 → 0.578 |

## C. 재현 가능한 부정 결과

| # | 주장 | 근거 |
|---|---|---|
| C1 | **시험한 센서 GT 형상 학습 레시피가 효과적이지 않았음** — 합성 전용 형상 감독, 사전학습 형상 미세조정 3설정, teacher 증류, 시간 어댑터의 악화 관찰 | REPORT §4.44~4.46. 센서 GT의 일반적 불충분성이나 no-fit metric 성공으로 일반화하지 않음 |
| C2 | **시험한 확대 모델이 검증 성능을 개선하지 못함** — 10.8M이 4.19M보다 학습 손실은 낮고 채점은 열세 | REPORT §4.44. 특정 레시피의 결과이며 용량 비병목의 증명은 아님 |
| C3 | **디코더 용량은 손실이 요구하지 않으면 죽은 무게다** — D1의 shuffle 경로가 zero-init 그대로(norm 0.008) 남고, boundary 손실을 켜야 작동. 반대로 처음부터 학습하면 D1이 선명도의 대부분을 만든다(제거 시 grad_ratio 0.5624 → 0.2023, F1 0.4745 → 0.1183) | REPORT §4.47 |
| C4 | **현재 체크포인트에서 state carry의 정확도 이득이 확인되지 않음** — dense와 매 프레임 리셋이 소수 셋째 자리까지 동일. TemporalBlock 우회 시 0.1153 → 0.2518 | REPORT §4.43. 모든 학습 조건의 불가능성은 아님 |
| C5 | **해상도 상향(384px)은 채점 계약을 바꾸지 않으면 걷을 수 없다** | REPORT §4.45 |
| C6 | **사전 형상 없이 gradient를 요구하면 albedo가 깊이로 샌다** — from-scratch 10.8M 증류 4회 실패, grad_ratio 1.80·2.25, D1의 RGB 분기 norm이 fine-tune arm의 1.95 대비 4.84까지 성장 | REPORT §4.47 |

## D. 관련 연구 대비 차별점

| 계열 | 대표 연구 | 공유하는 것 | SOKKANAEM에서 검증할 차별화 |
|---|---|---|---|
| 단안/비디오 깊이 | MiDaS, DPT, Depth Anything (v1/v2/3), VDA, NVDS | 깊이 추정 태스크 | 프레임당 연산이 변화율에 반응(이들은 고정 비용), 후처리 없는 플리커 억제 |
| 토큰 감축 | ToMe, EViT, DynamicViT | 토큰 단위 절감 | 프레임 **간** 중복 활용, dense 출력 복원 문제 없음 |
| 변화 기반 스킵 | Skip-Convolutions, DeltaCNN, Eventful Transformer | "변한 것만 재계산" | SSM state 보존과 dense depth readout의 결합; 출력 캐시는 별도 근사 |
| Vision Mamba | Vim, VMamba, VideoMamba | SSM 백본, O(N) 스캔 | 입력 적응적 갱신을 비디오 깊이에 결합; 최초성 미확정 |
| 선명도 평가 | DA-2K(DA V2), boundary F1 계열 | 경계 품질을 따로 잰다 | 여러 실패 양상을 함께 점검하고 정합·해상도 의존성을 명시; non-gameable 증명은 없음 |

## E. 주장하면 안 되는 것

- **"DA2급 정확도"** — DA2 자신이 P1을 못 넘는다(TUM 평균 0.3464, 정합 실패 15클립). "DA2급"은
  선명도 축에서만 쓸 수 있다.
- **효율 예산 안의 DA2급 품질** — 10.8M student는 네 번 실패했다. 현재 DA2급 선명도(94~106%)는
  Q0(24.81M, 57.6 GMAC)에서만 나온다.
- **장스트림(P3)** — 최고값이 L8→L256 +19.3%(목표 5%). 미해결.
- **"연산량 ∝ 변화율"의 원래 형태** — §4.11에서 실측으로 기각됨. A5의 축소된 형태만 유효.
