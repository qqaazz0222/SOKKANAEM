# Q0 평탄 영역 보존·전체 갱신 시 SSM 상태 유지 실험

2026-09-08. [품질·가속 순차 실험](Q0_QUALITY_SPEED_STUDY_20260908.md)의 후속.

**평탄 영역 손실과 refresh-state carry를 구현하고 2×2 matched 학습 및 제거 대조를 완료했다.
거칠기 개선은 확인했으나 경계/일부 source 정확도 손실로 채택 가능한 후보는 없다.**
기존 논문 모델과 Q0 가중치, final 데이터는 변경하지 않았다.

후속: [거리 보정 입력 분리·국소 보정](Q0_LOCAL_CALIBRATION_STUDY_20260909.md).
Pooled delta의 정확도 영향과 현재 decoder가 post-encoder CLS delta를 사용하지 않는 점을
확인했다. 이 문서의 당시 실험 수치는 유지한다.

## 1. 실험 설계

- 동일 초기 checkpoint: `work_dirs/qquality_feature_20260908/adapter.pt`.
- 네 조건: 상태 초기화/유지 × 평탄 영역 손실 없음/있음.
- 각각600step, seed736, AdamW lr1e-4, 동일48클립·192프레임과 동일 무작위 클립 순서.
  총2,400step 추가 학습. 네 조건 모두 train/inference K2로 맞췄다.
- Q0 encoder/decoder/calibration 가중치는 동결하고 152,400parameter adapter만 학습했다.
- Training clip은 실제 연속4프레임. t0/t2는 Q0 특징 갱신, t1/t3는 SSM 예측이다.
  Carry 조건은 t1의 h가 t2 갱신을 거쳐 t3 손실까지 전달된다.
  더 긴 시퀀스의 BPTT 학습은 하지 않았으며 서로 다른 clip을 임의로 이어붙이지 않았다.
- 기존 training feature cache는 FP16, pooled는 FP32. 현재 Q0에서 최종 teacher 깊이를 다시 생성했다.
- 학습 GT는 기존 `architecture_readout_20260908/train_cache.pt`에서 **경로 전체가 같은 clip**을
  연결하고 RGB tensor의 정확한 동일성을 확인했다. 개발 GT는 학습이나 추론에 전달하지 않았다.
- L32 개발256프레임, L256 개발1,024프레임. 이전에 이미 사용한 개발 자료를 재사용한다.
  두 개발 묶음 간 프레임 경로 교집합0, 학습/개발 교집합0을 확인했다.
  같은4장면의 다른 구간이므로 독립 장면 검증이나 길이만 달라진 통제 실험은 아니다.

## 2. 평탄 영역 손실

기본 목적함수는 featureL1+.1 pooledL1+5 log-depth L1+10 다중 스케일 log-gradient L1이다.
새 조건은 다음 항을 가중치20으로 추가한다.

1. Training GT의 normalized-disparity gradient 하위50%에서 유효한 평탄 영역을 정한다.
2. GT 경계와 그 주변을 제외한다. `sharpness.flat_tv`와 같은 mask 정의를 사용한다.
3. 해당 영역에서 **예측의 gradient 크기가 Q0 teacher보다 커진 부분만** 벌점으로 준다.
4. 유효 픽셀이 부족한 경우0을 반환하고 손실을 NaN으로 만들지 않는다.

Evaluation 기준을 바꾸지 않았다. 공간적 거칠기를 억제하는 손실이며 시간 깜빡임 손실은 아니다.
손실 mask가 경계를 제외하더라도, 특징을 공유하는 decoder 출력의 경계까지 불변인 것은 아니다.

## 3. 전체 갱신 시 상태 유지 계약

`sokkanaem/qflat_state.py`의 `refresh_memory`/`PersistentStream` 구현:

- Q0 전체 갱신 때 **4단계 특징과 pooled calibration 입력은 항상 새 Q0 값으로 교체**한다.
- 직전 RGB와 현재 RGB의 patch-14 MSE≤.002인 위치만 이전 h를.9배 감쇠해 유지한다.
- 나머지 위치, stream 시작, 매32프레임 강제 초기화 지점에서는 h를0으로 둔다.
- 갱신 프레임 출력은 h를 사용하지 않으므로 Q0 출력 보존과 h 유지는 별개의 계약이다.
- 이 조건은 위치별 RGB 변화 heuristic이다. 움직임 정렬, 물체 대응, 가림 해제의 안전성을
  보증하지 않는다. 프레임 크기/장치 변경에 대한 일반 배포 API로 확장한 것도 아니다.

각 학습 checkpoint를 reset/carry 두 방식으로 모두 평가하여, 재학습 차이와 같은 가중치에서의
상태 제거 효과를 분리했다. 마지막 조건처럼 carry를 학습한 모델에도 reset 대조가 있다.

## 4. L256 결과 — 학습과 동일한 상태 정책

원본 Q0의 raw AbsRel .159183, 경계 F1 .614665, flat-TV .030988이다.
모든 후보는 encoder 호출을50% 생략한다. 평탄 영역 상대 증가 허용치는 기존≤1%를 유지했다.

| 학습/운용 조건 | Raw AbsRel↓ | 경계 F1↑ | Flat-TV 상대 변화 | 평균 ms | 품질 gate |
|---|---:|---:|---:|---:|---|
| Q0 Dense | .159183 | .614665 | 기준 | 약7.50 | 기준 |
| Reset, 기본 출력 손실 | .160377 | .609931 | +1.148% | 4.822 | 실패 |
| Reset, 평탄 손실 추가 | .159884 | .602991 | −1.094% | 4.821 | 실패 |
| Carry, 기본 출력 손실 | .159666 | .610143 | +1.185% | 4.853 | 실패 |
| Carry, 평탄 손실 추가 | .160001 | .604630 | −0.846% | 4.841 | 실패 |

평탄 손실은 거칠기 조건을 통과하게 했지만 경계 F1 감소가 허용치.005를 초과했다.
결합 후보의 경계 F1 감소는.010035다. Bonn raw AbsRel도 원본 Q0 대비+3.313%로
source별 허용치2%를 넘었다. 균형 평균 raw만 보면 통과하므로 source별 확인이 중요하다.

평탄 손실이 없는 조건도 flat-TV 또는 source별 정확도 등에서 실패했다.
시간은 GPU-resident FP32, Python 호출 동기화, IO/H2D 제외이며 실제 장치 pipeline 지연이 아니다.
이번 시간은 평가1회 값이다. 작은 reset/carry 시간 차이를 속도 기여로 해석하지 않는다.

## 5. L32와 같은 가중치의 상태 제거

| 학습/운용 조건 | L32 Raw AbsRel | L32 경계 F1 | L32 판정 |
|---|---:|---:|---|
| Q0 Dense | .128127 | .660219 | 기준 |
| Reset, 기본 출력 손실 | .132061 | .659804 | raw/source 등 실패 |
| Reset, 평탄 손실 | .130760 | .655431 | raw/source 실패 |
| Carry, 기본 출력 손실 | .130750 | .659831 | raw/source 실패 |
| Carry, 평탄 손실 | .128922 | .655689 | Bonn raw+2.579%로 실패 |

결합 모델은 L32의 균형 평균 지표들은 통과하지만 source별 조건까지 통과하지 못했다.

같은 **carry+flat checkpoint**에서 h를 매 갱신 초기화한 대조:

| 평가 | Reset raw | Carry raw | Reset 경계 F1 | Carry 경계 F1 |
|---|---:|---:|---:|---:|
| L32 | .129766 | .128922 | .656481 | .655689 |
| L256 | .159751 | .160001 | .605053 | .604630 |

L32에서는 raw가 개선되지만 L256에서는 악화하고, 경계 F1은 두 경우 모두 낮아졌다.
상태가 전달되는 사실과 상태가 일관된 품질 이득을 주는 사실을 구분해야 한다.
현재 결과는 후자를 입증하지 못했다. 단일 seed·짧은 학습 horizon의 개발 screen이므로
시간 상태 일반이 무효하다고 결론내리지 않는다.

## 6. 검증

- 관련 회귀 테스트 **66 passed**.
- 네 실행×두 inference 정책의 전체 갱신 출력 **5,120프레임**을 이전에 저장한 정확한
  Q0 dense 출력과 비교: **최대 절대 차이0**, rtol=0/atol=0 검사 통과.
- 각 실행에서 encoder 호출 수가 전체 frame 수의 절반임을 확인했다.
- Fresh Q0 features/pooled 불변, h 감쇠·gradient 전달, 큰 RGB 변화/강제 초기화,
  GT-flat mask와 empty-mask loss 처리 테스트를 추가했다.
- 원본 Q0/초기 adapter/실행 source/training cache/GT cache/development cache의
  기록된 hash 계약을 완료 후 다시 확인했다.
- 원본 `qmodel.py`, `qdelta_ssm.py`, `qstream.py`, `ssm.py`, `detector.py`, `sharpness.py`는 변경하지 않았다.

## 7. 결정과 다음 우선순위

이번 두 가설의 구현·대조·개발 평가는 완료했다. **원고 모델 교체나 배포 후보 승격은 보류한다.**
다음 작업은 이미 실패한 기준을 완화하는 것이 아니다.

1. 경계와 평탄 영역이 같은 특징 보정 때문에 함께 바뀌는 문제를 분리한다.
   전체 특징 delta가 아니라 공간적으로 제한된 출력/특징 보정 경로와 경계 보존을 대조한다.
2. Source별 metric 오류를 조사한다. CLS 변화와 raw pooled calibration 입력 변화를
   각각 제거하는 대조로 원인을 확인해야 하며, 현재 결과만으로 calibration이 원인이라고 단정하지 않는다.
3. 상태 유지의 유효성을 더 긴 실제 학습 clip과 물체/움직임 대응으로 검증한다.
   이번4프레임 학습과 좌표 고정 heuristic을 장기 기억 모델의 완성으로 취급하지 않는다.
4. 독립 개발 장면·다중 seed·Nano B01 5W/10W·Pi4는 미실행이며, 현재 후보가 핵심 품질 gate를
   통과하지 못한 상태에서 실기기 속도만 논문 성과로 추가하지 않는다.

## 산출물 / 재현

```text
sokkanaem/qflat_state.py
scripts/qflat_state_study.py
scripts/audit_qflat_state.py
tests/test_qflat_state.py
work_dirs/qflat_reset_control_20260908/
work_dirs/qflat_reset_flat_20260908/
work_dirs/qflat_carry_control_20260908/
work_dirs/qflat_carry_flat_20260908/
work_dirs/qflat_audit_20260908/audit.json
```

각 학습 폴더에 protocol, targets/GT-flat masks, adapter, history, L32/L256 predictions와
지표/프레임별 telemetry를 보존했다. 재실행은 기존 결과를 덮어쓰지 않는 새 출력 경로를 사용한다.

```bash
HF_HUB_OFFLINE=1 python scripts/qflat_state_study.py --out work_dirs/qflat_reset_control_repro
HF_HUB_OFFLINE=1 python scripts/qflat_state_study.py --out work_dirs/qflat_reset_flat_repro --flat
HF_HUB_OFFLINE=1 python scripts/qflat_state_study.py --out work_dirs/qflat_carry_control_repro --carry
HF_HUB_OFFLINE=1 python scripts/qflat_state_study.py --out work_dirs/qflat_carry_flat_repro --carry --flat
python scripts/audit_qflat_state.py --out work_dirs/qflat_audit_repro
```

학습 스크립트의 부모 checkpoint/data와 audit 스크립트의 대상은 보존된20260908 출력을
명시 참조한다. 위 audit 명령은 새 `_repro` 학습 결과를 자동으로 연결하는 명령이 아니다.
