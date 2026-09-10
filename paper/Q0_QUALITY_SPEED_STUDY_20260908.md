# Q0 품질·가속 동시 확보: 순차 고도화 실험

2026-09-08. 기존 [통합 1차 실험](Q0_SOKKANAEM_INTEGRATION_20260908.md)의 후속이다.

**요약: 1–5단계를 구현·실험하고 6단계의 L256 개발 검증까지 수행했다.
L32에서 품질·가속을 함께 통과한 후보는 있으나 L256까지 양쪽을 통과한 후보는 없다.**
최종 논문 모델과 원본 Q0 가중치는 교체하지 않았다. 품질 허용치를 완화하지 않았다.

후속 [평탄 영역·refresh-state 실험](Q0_FLAT_STATE_STUDY_20260908.md)에서
명시적 GT-flat 손실과 갱신 시 h 유지를2×2 비교했다. 이 문서의 당시 결과는 유지한다.

## 1. 범위와 비교 계약

- 모든 Q0 추론은 같은 518px, 점수 계산은 256px, RTX4090 FP32다.
- Q0는 Depth Anything V2 기반 encoder/decoder와 기존 metric calibration을 포함한다.
  Q0의 사전학습 품질 자체를 SOKKANAEM의 독창적 성과로 주장하지 않는다.
- SSM 학습은 이전 600step adapter에서 동일하게 시작하여 각 600step 추가했다.
  특징 손실/출력 손실 × 정렬 없음/있음의 네 실행은 같은 seed736·클립 순서·학습량이다.
- 학습은 TUM/Bonn/VKITTI2 각16클립, 총48클립·192프레임. 기존 FP16 stage-feature 캐시와
  FP32 pooled feature를 사용한다. 새 출력 target은 정확한 Q0 경로에서 다시 생성했다.
- L32: 기존 개발 4장면×2클립×32프레임 =256프레임.
- L256: 같은 개발 4장면에서 각1클립×256프레임 =1,024프레임.
  선택을 평가 전에 기록했고 L32와 프레임 경로 교집합은 **0**이다. 학습과도 교집합0이다.
- L256은 **새 장면의 독립 시험이 아니다**. 기존 장면의 다른 구간이며, 이전 native 연구에서도
  사용한 개발 자료다. L32/L256의 절대 지표 차이를 길이 때문에 발생한 누적 오차로 해석하지 않는다.
- 원본 final manifest를 사용하지 않았다. risk 정책은 L256 진단을 본 뒤 수행한 후속 개발이다.

기존 Q0-relative gate 유지:
raw/경계 AbsRel·overshoot·flat-TV 상대 증가≤1%, 경계 F1 절대 감소≤.005,
source별 raw 증가≤2%. 모든 품질 조건과 평균 지연시간≥5% 감소를 함께 요구한다.
이는 개발 선별 기준이지 통계적 비열등성 또는 장치별 품질 보증이 아니다.

## 2. 1단계 — 실행시간 분해

80회 중 첫16회를 제외한64회 구성 요소별 동기화 wall time 평균이다.
중간 동기화로 실행 스케줄이 바뀌므로 아래 수치를 단순 합산해 전체 속도를 주장하지 않는다.

| 구성 요소 | 평균 ms |
|---|---:|
| 전처리 | 0.035 |
| Q0 encoder | 6.216 |
| 전체 특징·상태 캐시 초기화 | 0.046 |
| SSM 갱신: 활성 패치25% | 0.628 |
| SSM 갱신: 활성 패치100% | 0.633 |
| Q0 decoder | 1.107 |
| Metric calibration | 0.341 |
| 시험용 RGB 움직임 추정 | 2.048 |
| 특징·SSM 상태 warp | 0.729 |
| Q0 전체 경로 별도 측정 | 7.520 |

Encoder 생략의 여지는 크지만 선택적 SSM의 활성률 감소가 이 구현·장치에서
비례하는 지연 감소로 이어지지는 않았다. 경량 RGB embedding/context 연산과
메모리 복사 등은 활성 패치 수와 무관하게 남는다. 움직임 정렬도 무료가 아니다.

## 3. 2–4단계 — 출력 보존, 움직임 정렬, RGB 보정

### 구현

- `sokkanaem/qquality.py`: frozen Q0 decoder를 거쳐 예측 특징에 기울기를 전달하는 별도 경로.
  원본 `Q0ExactStream.decode`의 추론용 no-grad 계약은 바꾸지 않았다.
- 출력 손실: 기존 featureL1+.1 pooledL1에 **5 log-depth L1 +10 다중 스케일 log-gradient L1** 추가.
  Teacher 경계의 가중치를 높였으나 GT 감독·명시적 평탄 영역 손실·시간 정렬 손실은 아직 없다.
- 움직임: 74px RGB에서 반경2 local block matching, 3×3 window, 작은 zero-motion 우선값.
  이전 프레임→현재 좌표의 backward sampling grid로 4단계 patch features와 h를 bilinear 이동한다.
  CLS와 pooled는 전역 값이므로 이동하지 않는다. 가림/새 물체를 보증해서 검출하는 구조가 아니다.
  비활성 state copy는 **정렬된 좌표의 상태** 기준이며 이전 원시 좌표의 bitwise 보존이 아니다.
- RGB 보정: RGB/log-depth/깊이 기울기를 입력받는 3,217parameter CNN, ±.06 log-depth residual.
  생략 프레임에만 적용하며 Q0 전체 갱신 출력은 그대로 둔다. Adapter는 동결하고600step 학습했다.
  보정량 제한은 Q0의 절대 출력 범위 준수를 보증하는 것과는 다르다.

### L32 품질 비교

아래 모든 K2는 encoder50% 생략, K4는75% 생략이다.
각 mode는 한 seed의 작은 개발 학습이므로 방식 일반의 가능/불가능을 결론내리지 않는다.

| 추가 학습/구조 | 정책 | Raw AbsRel↓ | 경계 F1↑ | 품질 gate |
|---|---|---:|---:|---|
| 원본 Q0 | Dense | .128127 | .660219 | 기준 |
| 특징 손실 추가 학습 | K2 | .128866 | .660168 | 통과 |
| 특징 손실 추가 학습 | K4 | .131381 | .634385 | 실패 |
| 최종 깊이·경계 손실 추가 | K2 | .129182 | .658569 | source별 raw 실패 |
| 최종 깊이·경계 손실 추가 | K4 | .132533 | .628986 | 실패 |
| 움직임 정렬+특징 손실 | K2 | .128684 | .651207 | 경계 실패 |
| 움직임 정렬+특징 손실 | K4 | .130140 | .626149 | 실패 |
| 움직임 정렬+출력 손실 | K2 | .128551 | .650630 | 경계 실패 |
| 움직임 정렬+출력 손실 | K4 | .129568 | .624078 | 실패 |
| 특징 후보+RGB 보정 | K2 | .128460 | .658295 | 통과 |
| 특징 후보+RGB 보정 | K4 | .130771 | .631697 | 실패 |

현재 출력 손실은 경계를 개선하지 못했다. 작은 출력 손실 가중치 탐색이나 더 많은 학습을
실행하지 않았으므로 최적화를 끝냈다는 뜻은 아니다. Bilinear 정렬 후 경계가 약해지는 결과는
나왔지만, matching 오차와 interpolation 자체의 영향을 별도로 식별하지는 않았다.
RGB 보정도 일부 raw/overshoot를 개선하는 대신 경계 F1은 소폭 낮췄다.

## 4. 통과 후보의 제거 대조와 반복 시간 측정

L32 256프레임, 정책마다64 warmup, 순서를 회전한3회 재측정 평균이다.
GPU-resident 입력의 동기화 Python 호출 시간이며 detector·전처리·모델은 포함,
파일 IO/H2D와 실제 카메라 pipeline은 제외한다.

| 후보 | 평균 ms | Q0 대비 가속 | 경계 F1 | L32 품질 |
|---|---:|---:|---:|---|
| Dense Q0 | 7.471 | 1.00× | .660219 | 기준 |
| 특징 추가 학습 SSM, K2 | 4.744 | 1.575× | .660168 | 통과 |
| SSM 없는 전체 출력 hold, K2 | 3.817 | 1.957× | .660049 | 통과 |
| 특징 SSM+RGB 보정, K2 | 4.803 | 1.556× | .658295 | 통과 |

단순 hold도 이 짧은 개발 구간에서 통과한다. SSM의 독립적인 우위를 입증한 결과가 아니다.
전체 갱신을 짝수/홀수 프레임 중 어느 쪽에서 하는지 바꾸는 phase 진단도 L32 품질은 통과했다.
Phase-shift는 초기 두 프레임을 전체 갱신하므로 기본 K2와 호출 수가 완전히 같지는 않다.

**현재 K2는 매 전체 갱신에서 h를0으로 초기화하고 그 사이에 SSM update가 한 번뿐이다.**
따라서 이 정책에서 과거 update의 h가 다음 update에 기여하는 장기 기억 효과는 구조상 없다.
K4 h-reset 대조도 수행했으나 큰/일관된 이득을 확인하지 못했다.

평균 지연시간 개선과 p95 개선은 다르다. 반복 측정 p95는 Q0 약7.50ms,
K2 특징 모델 약7.55–7.58ms였다. 전체 갱신 프레임의 지연은 여전히 남는다.

## 5. 6단계 일부 — L256 확대 검증

| 후보 | Raw AbsRel↓ | 경계 F1↑ | Flat-TV 상대 증가 | 평균 ms | 판정 |
|---|---:|---:|---:|---:|---|
| Dense Q0 | .159183 | .614665 | 기준 | 7.504 | 기준 |
| 특징 SSM K2 | .159068 | .611582 | +1.513% | 4.809 | Flat-TV 실패 |
| 특징 SSM K2, phase 변경 | .158933 | .611286 | +1.654% | 4.807 | Flat-TV 실패 |
| 전체 출력 hold K2 | .159210 | .612048 | +1.603% | 3.851 | Flat-TV 실패 |
| 특징 SSM+RGB K2 | .159371 | .609847 | +1.524% | 4.865 | Flat-TV 실패 |

Flat-TV는 GT 평탄 영역의 공간적 깊이 거칠기 지표다. 이것만으로 시간 깜빡임을 측정했다고
주장하지 않는다. 어느 후보도 기준≤1%를 만족하지 않아 L32 통과만으로 모델을 채택하지 않았다.

## 6. 5단계 — 학습된 예상 복원 오차에 따른 갱신

L256의 실패를 확인한 뒤, RGB 변화 통계와 캐시 나이를 입력으로 teacher 복원 오차를
예측하는 1,025parameter MLP를 추가했다. `sokkanaem/qrisk.py`는 개발용 wrapper다.

- Adapter/Q0 동결. Target은 실제 생략 프레임의 Q0 log-depth 오차+2×log-gradient 오차.
- 학습 자료 중 source별 앞12클립의108개 update로 학습, 뒤4클립의36개 update로 예측 오차 진단.
  Adapter는 앞서 전체48클립을 보았으므로 이 split을 모델 전체의 독립 검증이라고 부르지 않는다.
- 1,000step, seed736. 임계값 .02/.04/.06을 평가 전에 기록하고 모두 보고했다.
- 가장 긴 전체 갱신 간격은4프레임. 위험도가 크면 Q0 전체 갱신한다.
- Held-out training-clip proxy MAE .02744로, 정밀하게 보정된 uncertainty가 아니다.
  값이 작다고 안전이 보증되지 않는다. 새 장면/가림/장면 전환 전용 detector도 없다.

| 임계값 | L32 encoder 생략 | L32 가속 | L32 품질 | L256 encoder 생략 | L256 가속 | L256 품질 |
|---|---:|---:|---|---:|---:|---|
| .02 | 0% | .971× | 통과 | 0% | .970× | 통과 |
| .04 | 18.75% | 1.122× | 통과 | 6.74% | 1.019× | 통과 |
| .06 | 55.86% | 1.633× | 실패 | 36.33% | 1.315× | 경계 실패 |

.04는 L256 raw .159038, 경계 F1 .614132, flat-TV .030985로 품질 기준은 모두 통과한다.
그러나 평균 시간 7.501→7.364ms, 약1.82% 감소에 그쳐 **5% 속도 조건은 실패**한다.
L256에서는955/1,024프레임이 전체 갱신이다. h를 이어받는 age2/3 update는 각각13/5프레임으로
적으며, 이 정책의 별도 h-reset 효과는 아직 평가하지 않았다.

## 7. 완료/남은 작업과 다음 결정

| 순차 작업 | 상태 |
|---|---|
| 1. 구성 요소별 프로파일 | 완료 |
| 2. 출력 깊이·경계 보존 학습과 matched feature control | 완료: 이번 설정 미채택 |
| 3. 움직임 정렬 및 2×2 비교 | 완료: 이번 정렬 방식 미채택 |
| 4. 제한형 RGB 경계 보정 | 완료: L32 통과, L256 실패 |
| 5. 학습형 갱신 위험도 proxy | 완료: 품질 통과점은 L256 가속 미달 |
| 6. 긴 영상/독립 장면/다중 seed/실기기 | L256 개발 진단만 완료, 나머지는 미실행 |

다음 우선순위는 임계값 완화가 아니다.

1. 평탄 영역의 오차를 teacher 출력 수준에서 직접 억제하고, 생략 프레임의 새 경계/가림 해제
   영역을 구분하는 학습을 설계한다. 이번 loss는 이를 명시적으로 다루지 않았다.
2. 특징/상태의 bilinear 이동과 local matching을 분리 대조한다. 현재 정렬 방식을 그대로
   확대하거나 더 큰 flow 모델을 추가해 가속을 희생하지 않는다.
3. 전체 갱신 때도 유효한 h를 이어가는 별도 정책을 먼저 검증해야 SSM 장기 상태 기여를
   평가할 수 있다. Q0 refresh 출력 보존과 h 보존은 별개의 계약으로 검사한다.
4. 같은 비용의 hold/비순환 predictor와 비교해 품질·가속 및 상태 기여가 확보된 후,
   아직 사용하지 않은 개발 장면·다중 seed·Nano B01 5W/10W·Pi4로 확대한다.

이는 후속 과제다. 현재 결과를 완성된 제안 모델의 실험 표로 원고에 승격하지 않는다.
하드웨어 이름을 알고 있다는 것과 해당 새 모델의 장치 실측이 끝났다는 것은 다르다.

## 8. 검증·산출물·재현

- 관련 회귀 테스트 **63 passed**.
- 실제 Q0 학습 경로 진단1프레임: 추론 decoder와 최대 출력 차이0, 네 단계 feature gradient
  norm .002075/.002751/.003348/.017405, pooled gradient norm .185122, Q0 가중치 gradient 없음.
  앞선48개 원본/stream parity 검사는 이전 통합 기록이며 이번에 다시 실행했다고 주장하지 않는다.
- 이번8개 출력 디렉터리의 source/원본 Q0/초기 adapter/refiner hash 계약 총99개를 재검사했다.
- 원본 `qmodel.py`, `qstream.py`, `qdelta_ssm.py`, `ssm.py`, `detector.py`는 수정하지 않았다.
- 새 코드: `sokkanaem/qquality.py`, `qrgb_refiner.py`, `qrisk.py`, `tests/test_qquality.py`.
- 실행별 protocol, 체크포인트, 예측 tensor, 지표, 프레임별 호출수/시간은 아래 폴더에 있다.
  각 재실행은 **새 출력 경로**가 필요하며 기존 결과를 덮어쓰지 않는다.

```text
work_dirs/qquality_profile_20260908/
work_dirs/qquality_feature_20260908/
work_dirs/qquality_output_20260908/
work_dirs/qquality_warp_feature_20260908/
work_dirs/qquality_warp_output_20260908/
work_dirs/qquality_rgb_20260908/
work_dirs/qquality_audit_20260908/
work_dirs/qquality_risk_20260908/
```

```bash
HF_HUB_OFFLINE=1 python scripts/qquality_study.py --out work_dirs/qquality_profile_repro --mode profile
HF_HUB_OFFLINE=1 python scripts/qquality_study.py --out work_dirs/qquality_feature_repro --mode feature
HF_HUB_OFFLINE=1 python scripts/qquality_study.py --out work_dirs/qquality_output_repro --mode output
HF_HUB_OFFLINE=1 python scripts/qquality_study.py --out work_dirs/qquality_warp_feature_repro --mode feature --warp
HF_HUB_OFFLINE=1 python scripts/qquality_study.py --out work_dirs/qquality_warp_output_repro --mode output --warp
HF_HUB_OFFLINE=1 python scripts/qrgb_study.py --out work_dirs/qquality_rgb_repro
HF_HUB_OFFLINE=1 python scripts/qquality_audit.py --out work_dirs/qquality_audit_repro
HF_HUB_OFFLINE=1 python scripts/qrisk_study.py --out work_dirs/qquality_risk_repro
```

마지막3개 스크립트는 기록된20260908 부모 출력을 명시적으로 참조한다.
따라서 위 명령은 현재 보존된 부모 아티팩트에서 각 단계를 재현하며,
새 `_repro` 부모 출력을 자동으로 연결하는 전체 재실행 파이프라인은 아니다.
