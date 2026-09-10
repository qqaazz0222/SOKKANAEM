# 이동 정렬 특징의 SSM·비순환 MLP 동등 예산 학습

후속: [K4 실제 상태 전달·기억 제거 대조](Q0_RECURRENT_FLOW_STUDY_20260909.md).
실제 기억 학습/출력효과를 확인했지만 품질 이득은 입증하지 못했다. 아래 K2 결과는 유지한다.

2026-09-09. [RGB 흐름·특징 캐시·SSM 결합 대조](Q0_RGB_FLOW_FEATURES_STUDY_20260909.md)의 후속.

**동일 초기 공통 가중치·자료 순서·손실·1,200step으로 SSM/MLP를 각각 학습했다.
두 후보 모두 거칠기는 줄였지만 경계F1이 악화되어 L32/L256 품질 기준을 통과하지 못했다.**
새 학습 후보는 미채택이며, 이전에 통과한 출력-flow 및 동결 특징 이동 후보를 덮어쓰지 않았다.
SSM의 추가 우위도 확인하지 못했다.

## 1. 실행 조건과 공정성 범위

| 항목 | SSM | 비순환 MLP |
|---|---|---|
| 공통 입력 | 현재 RGB, 이동 정렬된4단계 Q0 feature | 동일 |
| 공통 경로 | RGB→32, context384→32, fuse,4단계 residual readout | 동일 초기 가중치 |
| Core | SelectiveSSM32, state8 | Linear32→77→32, GELU |
| 전체 추가 파라미터 | 89,040 | 89,037 |
| Core 파라미터 | 5,040 | 5,037 |
| 초기 출력 | Residual head0, 정렬 특징 그대로 | 동일 |
| 학습 | 1,200step, batch1, AdamW .0003, wd .0001 | 동일 |
| Seed/표본 순서 | 738, 저장된 동일1,200개 index | 동일 |

- 모두 새로 초기화했다. SSM만 과거 학습 가중치를 가져오는 불균형은 없다.
- 기존 Q0 encoder/decoder/metric calibration은 동결했다. Post-encoder CLS/pooled delta와 미사용 global head는 새 adapter에서 제외했다.
- 파라미터 수는 거의 같지만 core 함수와 활성 파라미터 역할까지 같지는 않다.
  특히 현재 K2에서는 SSM의 transition A256개가 데이터 의존 출력에 기여하지 않는다(§4).
- Training cache 전체192클립·768프레임, TUM/Bonn/VKITTI2 각64클립을 사용했다.
  직전 전체 갱신→다음 생략 프레임의384쌍이다. 새 데이터 수집은 아니다.
- DIS nearest 흐름으로 Q0 feature를 먼저 정렬하고, 같은 RGB 변화 mask를 두 모델에 적용했다.
- Training cache의 정렬/teacher feature는 FP16 저장 후 FP32로 복원했다. RGB/depth/pooled는 FP32다.
  Dev 추론은 기존 FP32 실제 경로이며, training feature 양자화는 실험의 한계다.
- Training과 개발 파일 경로 교집합0 및 reserved final sequence 제외를 확인했다.
  개발 L32 256프레임/L256 1024프레임은 앞 실험과 같으며 독립 검증이 아니다.

공통 손실은 다음과 같다.

`1×patch feature L1 + 5×Q0 log-depth L1 + 10×Q0 multiscale log-gradient L1 + 20×training-GT-flat excess`

개발 GT는 평가에만 쓴다. 생략 추론에서 현재 Q0 teacher 또는 GT를 사용하지 않는다.
최종1,200step checkpoint를 평가했으며 개발 점수로 중간 checkpoint를 선택하지 않았다.

## 2. 개발 결과

### L256

| 구성 | Raw AbsRel↓ | 경계 F1↑ | Flat-TV 상대 변화 | 평균 ms¹ | 품질·가속 동시 gate |
|---|---:|---:|---:|---:|---|
| Q0 dense | .159183 | .614665 | 기준 | 7.527 | 기준 |
| 기존 출력-flow, SSM 없음 | .159120 | .620225 | +0.852% | 4.379 | 통과 |
| 정렬 특징, update 없음 | .159719 | .610922 | +0.972% | 5.279 | 통과 |
| 새 정렬 학습 SSM | .159483 | .595872 | −2.696% | 5.524 | 경계 실패 |
| 새 정렬 학습 MLP | .159326 | .599388 | −2.509% | 5.446 | 경계 실패 |

¹ 이번 실행의 단일 동기화 평가값이다. 초기 IO/H2D를 제외한 GPU-resident 호출 시간이며,
CPU DIS·내부 전송·특징 이동·모델을 포함한다. 다른 GPU 프로세스가 있는 RTX4090 환경이다.
작은 방법 간 지연 차이의 통계적 의미나 Pi4/Nano 실기기 속도를 주장하지 않는다.

SSM과 MLP 모두 raw·edge AbsRel, overshoot, flat-TV, source별 raw 제한을 통과했지만,
경계F1 감소는 각각 .018793/.015278로 허용 .005를 크게 초과했다.
지연만으로는 약1.36×/1.38× 가속이지만 **품질을 함께 확보한 새 모델로 볼 수 없다**.

### L32

| 구성 | Raw AbsRel↓ | 경계 F1↑ | Flat-TV 상대 변화 | 품질 gate |
|---|---:|---:|---:|---|
| Q0 dense | .128127 | .660219 | 기준 | 기준 |
| 새 정렬 학습 SSM | .128133 | .648856 | −5.092% | 경계 실패 |
| 새 정렬 학습 MLP | .127939 | .651378 | −5.150% | 경계 실패 |

MLP가 SSM보다 raw/F1에서 조금 좋았지만 두 후보 모두 실패다.
한 seed·한 손실·한 학습 예산의 결과를 MLP 또는 SSM 일반의 우열로 확대하지 않는다.

## 3. 같은 학습 자료 전체를 이용한 손실 감사

서로 다른 표본의 step1/200/400 손실을 곧바로 학습 추세로 해석하지 않았다.
동일384쌍 전체에서 update 없음과 학습 완료 모델의 평균을 다시 계산했다.

| 학습 자료 평균 | Update 없음 | SSM 학습 후 | MLP 학습 후 |
|---|---:|---:|---:|
| Feature L1 | .718476 | .720948 | .720455 |
| Q0 log-depth L1 | .025734 | .026577 | .026473 |
| Q0 log-gradient L1 | .011085 | .010913 | .010932 |
| GT-flat excess | .005007 | .003709 | .003828 |
| 가중 총손실 | 1.058143 | 1.037155 | 1.038701 |

총손실은 SSM−1.984%, MLP−1.837%로 줄었다. 그러나 feature와 Q0 depth 차이는 오히려 증가했고,
감소분은 주로 강하게 가중된 flat excess에서 나왔다. SSM의 flat excess는−25.92%다.
즉 학습 실행 자체가 멈춘 것은 아니며, **현재 가중 손실의 감소가 원하는 경계 보존과 일치하지 않았다.**
이것은 현재 loss의 trade-off를 보여주는 근거다. 별도 손실 ablation 없이 flat 항 하나만을 유일 원인으로 확정하지 않는다.
학습 GT-flat 손실과 개발 balanced flat-TV는 집계·정의가 다르므로 같은 수치로 취급하지 않는다.

## 4. K2에서는 장기 기억을 검증할 수 없는 이유

현재 구현은 전체 갱신에서 h를0으로 만들고 생략 프레임에서 한 번만 업데이트한다.

`h₁ = exp(ΔA) · 0 + (Δx)B = (Δx)B`

이 조건의 SSM 출력은 현재 입력으로 구한 B/C/Δ와 gate를 사용하는 비순환 식으로 계산할 수 있다.
따라서 이번 SSM/MLP 비교는 **서로 다른 한 단계 보정 함수**의 비교다.
SSM이 이기더라도 시간 기억의 장점이라고 할 수 없다.

- 실제 학습한 SSM과 CUDA training 입력8쌍에서 비순환 식을 대조했다.
- Core 출력 최대 차이≤4.77×10⁻⁷, 새 h 차이0, transition A의 데이터 gradient 최대값0을 확인했다.
- CPU 단위 검사에서는0이 아닌 이전 h를 넣으면 출력이 달라짐도 확인했다.
- A의 optimizer weight decay 가능성과 데이터 gradient 부재는 별개다. 여기서 “gradient0”은 데이터 의존 항에 대한 주장이다.
- 캐시된 Q0 특징에는 직전 영상 정보가 있으므로 **모델 전체가 이전 영상 정보를 전혀 쓰지 않는다**는 의미는 아니다.
  별도의 SSM recurrent h 누적 기여가 이 주기에서 식별되지 않는다는 뜻이다.

## 5. 검증·결정·다음 우선순위

- 신규4개 포함 관련 테스트 **88 passed**.
- 전체 갱신2560회에서 해당 dense Q0와 출력 차이0. 여러 정책의 반복 비교 횟수다.
- 공통 초기 가중치 동일성, 파라미터 수, 초기 residual0, 비활성 patch/h copy,
  CLS/pooled 유지, gradient 및 비순환 식 대조 확인.
- 주 실험과 training audit의 source·Q0·training/development/cache hash 계약 검증 완료.
- 출력-flow 및 update 없음의 정량 결과가 앞 단계와 동일하게 재현됐다.
- 원래 논문 모델·원고·최종 결과, 이전 통과 checkpoint를 수정하지 않았다.

**새 학습 SSM/MLP는 미채택이다.** 기존 출력-flow nearest를 품질·속도 기준 대조군으로 유지한다.
이전 동결 특징+SSM의 기준 통과 기록은 유효하지만 고유 우위 미입증 상태도 그대로다.

다음 작업에서는 K2 단일 업데이트 학습을 반복하기보다 다음을 우선한다.

1. 실제 h가 다음 업데이트로 전달되는 K≥3 또는 명시적 refresh-state carry 조건을 만들고,
   같은 가중치의 state 유지/초기화와 별도 학습 비순환 대조를 준비한다.
2. 원래 training sequence 안에서 흐름·가림·경계를 보존하는 목적함수를 검증한다.
   총손실 감소만 보지 않고 depth/경계/flat의 개별 보존을 확인한다. Dev gate는 완화하지 않는다.
3. 기존 출력-flow의 품질을 훼손하지 않는 결합 경로를 우선 비교한다. 효과가 없는 SSM을 논문 기여를 위해 억지로 남기지 않는다.
4. 후보 고정 뒤 독립 장면·시간 일관성·반복 지연·Pi4/Nano5W·10W 실측으로 확대한다.

## 산출물 / 재현

- 구현 `sokkanaem/qaligned_adapter.py`, 검사 `tests/test_qaligned_adapter.py`.
- 학습·평가 `scripts/qaligned_study.py --out <새 경로>` (기본 각1,200step).
- 감사 `scripts/audit_qaligned.py --out <새 경로>` (이번 학습 출력 경로를 입력으로 고정).
- Python `/home/hyunsu/miniforge3/envs/sokkanaem/bin/python`.
- `work_dirs/qaligned_20260909/`: protocol, training_order, cache_record, 두 checkpoint/history,
  results, parity, integrity 및 실제 예측 tensor.
- `work_dirs/qaligned_audit_20260909/audit.json`: 같은384쌍의 손실·실제 SSM 대수 검사.

이번에는 새 정성 그림을 만들지 않았다. 이전 출력-flow 그림을 이번 학습 후보의 결과로 사용하면 안 된다.
