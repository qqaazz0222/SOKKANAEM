# Q0 + SOKKANAEM 스트리밍 통합 1차 결과

2026-09-08. **Q0 출력 동등성 복구와 선택적 SSM 연결을 구현했다.**
품질과 속도를 동시에 통과한 배포 후보는 아직 없으며 기존 논문 모델은 유지한다.

후속: [품질·가속 순차 고도화 실험](Q0_QUALITY_SPEED_STUDY_20260908.md)에서
출력 보존/움직임 정렬/RGB 보정/학습형 갱신과 L256 개발 진단까지 진행했다.
아래 표는 이 문서 작성 당시의 1차 결과이며 후속 결과로 덮어쓰지 않는다.

## 1. 원본 Q0와 정확히 같은 스트리밍 경로

기존 `qmodel.py`는 수정하지 않고 `sokkanaem/qstream.py`에 `Q0ExactStream`을 추가했다.
기존 일반 실행은 calibration에 마지막 **raw hidden state의 CLS 제외 평균**을 사용하지만,
이전 스트리밍 경로는 decoder용 normalized feature map을 평균하여 다른 입력을 전달했다.
CLS 처리를 token 수의 홀짝으로 추정하는 방식도 사용하지 않도록 했다.

새 경로는 backbone을 **프레임당 한 번만 실행**하여 normalized decoder features와
raw calibration features를 동시에 얻는다. 별도로 Q0 전체 추론을 두 번 하는 우회 구현이 아니다.

- 개발 8클립의 t=0/15/31, 해상도 224/518에서 총 **48개 출력 비교**.
- 평균 상대 차이와 최대 절대 차이 모두 **0**.
- 각 비교에서 backbone 호출 1회 확인.
- 이것은 검사한 입력·환경에서의 결과이며 모든 장치/정밀도의 비트 동등성 증명이 아니다.
- 기존 QT 체크포인트를 재학습하거나 그 과거 지표를 수정하지 않았다.
  이번 수정으로 QT의 모든 실패 원인을 설명했다고 주장하지 않는다.

## 2. 변화 감지 + 전체 출력 재사용 대조군

먼저 학습 없이 실제 encoder 호출을 줄일 수 있는지 검사했다.
원본 입력을 Q0의 patch-14/37×37 grid에 맞춰 변화량을 계산한다.
직전 프레임뿐 아니라 **마지막 전체 갱신 프레임**과 비교하여 작은 변화의 누적을 놓치지 않도록 했다.
히스테리시스·마스크 팽창·강제 전체 갱신을 사용한다.

| 정책 | Q0 전체 호출/256프레임 | Raw AbsRel | 경계 F1 | 평균 ms | 판정 |
|---|---:|---:|---:|---:|---|
| Dense Q0 | 256 | 0.128127 | 0.660219 | 7.512 | 품질 기준 |
| 보수적 변화 감지 | 256 | 0.128127 | 0.660219 | 7.644 | 품질 유지, 가속 없음 |
| 중간 변화 감지 | 256 | 0.128127 | 0.660219 | 7.645 | 품질 유지, 가속 없음 |
| 4프레임마다 전체 갱신·출력 유지 | 64 | 0.129990 | 0.634581 | 1.985 | 빠르지만 품질 실패 |

이 단계는 **전체 출력 hold 대조군**이며 선택적 SSM 모델이라고 부르지 않는다.
움직이는 장면에서 프레임 전체가 충분히 안정적인 경우가 드물어 변화 감지 정책은 스킵하지 못했다.

## 3. 실제 선택적 SSM 특징 갱신

`sokkanaem/qdelta_ssm.py`의 `QDeltaSSM`은 현재 RGB의 경량 patch embedding,
캐시된 Q0 특징, 패치별 시간 상태로 Q0 특징의 변화량을 예측한다.

```text
현재 RGB → patch 변화 감지
  ├─ keyframe / 변화율 >40% → Q0 encoder 전체 실행 → 특징·상태 초기화
  └─ 그 외 → RGB 경량 embedding + 선택적 SSM → active patch 특징 갱신
                               ↓
                  Q0 decoder + 원본 거리 보정기
                               ↓
                           metric depth
```

- Q0 encoder/decoder/calibration 가중치는 모두 동결한다.
- 네 단계의 Q0 patch features를 캐시하고 활성 패치만 SSM에 gather한다.
- 원본 `SelectiveSSM` 구현 사용. 비활성 hidden state는 bitwise copy한다.
  binary zero-step의 update/copy 동작을 사용하지만 native 전체 backbone을 복사한 것은 아니다.
- 비활성 **patch token 특징**도 그대로 유지한다. CLS와 calibration pooled feature는
  활성 패치의 평균 출력으로 별도 갱신하므로, 모든 비활성 영역의 최종 깊이까지 불변인 것은 아니다.
- update 프레임은 Q0 encoder를 호출하지 않는다. 경량 RGB embedding과 Q0 decoder는 실행한다.
- adapter는 152,400 파라미터다. native의 공간 SSM 경로는 이식하지 않았으며,
  Q0(Depth Anything V2 기반)의 공간 encoder를 그대로 사용한다.
- 초기 readout은 0이므로 새 특징 delta는 0이다. 이것은 **캐시를 보존하는 초기화**이지,
  바뀐 현재 프레임의 dense Q0와 같다는 보장이 아니다.
- 카메라 움직임 정렬, occlusion warp, 학습된 uncertainty refresh는 아직 없다.

### 학습

- 이전 training cache에서 TUM/Bonn/VKITTI2 각 16클립, 총 48클립·192프레임.
- 개발 프레임과 교집합 없음 확인. GT 깊이는 학습 손실에 사용하지 않았다.
- Q0 teacher features와 raw pooled features를 생성하여 저장했다.
  normalized stage features는 FP16 캐시, pooled는 FP32다.
- 600 step, seed 736, AdamW lr 3e-4. 매번 4프레임 clip 하나를 사용한다.
- t0는 teacher 전체 갱신, t1–3은 mask에 따라 예측한다. 학습 중에는 높은 변화율의
  전체 갱신 fallback을 끄고 update 예측을 학습한다.
- 손실은 stage-feature L1 + 0.1 pooled-feature L1이다.
  아직 decoder를 거친 깊이/경계 보존 손실은 적용하지 않았다.
- 실제 평가에서는 encoder 전체 갱신 시 FP32 원본 특징을 사용한다.

## 4. Q0 품질 보존과 속도 판정

이번 목표는 native를 능가하는 것이 아니라 **같은 518px Q0 대비 품질을 유지하며 계산을 줄이는 것**이다.
별도 실험 계약을 평가 전에 저장했다. 기존 논문 기준을 바꾼 것은 아니다.

- Raw/경계 AbsRel, overshoot, flat-TV: Q0 대비 상대 증가 ≤1%.
- 경계 F1: 절대 감소 ≤0.005.
- source별 raw AbsRel: 상대 증가 ≤2%.
- 위 품질 조건 전부와 평균 GPU 모델 호출 시간 ≥5% 감소를 함께 요구한다.
- 표본: 개발 `acc_real_L32.json`에서 각 4장면의 균등 선택 2클립,
  총 8클립·256프레임. GT score resolution 256, inference 518.
- 시간은 RTX4090 FP32, GPU-resident 입력, 호출 전후 동기화,
  detector·전처리·모델 포함, IO/H2D 제외다. Pi 4/Nano 또는 실제 비디오 pipeline 속도가 아니다.

| 정책 | Q0 encoder 호출 | 스킵률 | Raw AbsRel ↓ | 경계 F1 ↑ | 평균 ms | 품질 | 품질+속도 |
|---|---:|---:|---:|---:|---:|---|---|
| Dense Q0 재측정 | 256 | 0% | 0.128127 | 0.660219 | 7.534 | 기준 | 기준 |
| 주기적 SSM, K4 | 64 | 75% | 0.131095 | 0.634806 | 4.717 | 실패 | 실패 |
| 적응형 SSM, K4·변화율 40% fallback | 252 | 1.5625% | 0.128166 | 0.660233 | 7.561 | 통과 | 실패 |

**적응형의 품질 통과는 256프레임 중 252프레임에서 원본 encoder를 실행한 결과**다.
적은 계산으로 Q0 품질을 유지했다고 해석하지 않는다. 주기적 SSM은 약 1.60× 빨랐지만
경계·정확도 등의 품질 조건을 실패했다. 출력 hold보다 비용이 더 드는 것은 update 때도
경량 SSM과 Q0 decoder를 실행하기 때문이다.

## 5. 시간 상태 기여도 제거 검사

동일한 학습 가중치·K4 정책에서 **특징 캐시는 유지하고 SSM hidden state만 매 프레임 0으로
초기화**했다. 별도로 재학습한 비순환 모델의 대조가 아니라 사후 진단이다.

- state carry − reset의 Raw AbsRel 차이: −0.0000722.
- 경계 F1 차이: −0.0000566.
- 이 작은 표본에서 시간 상태의 유의미한 품질 이득은 아직 입증되지 않았다.
  초기 통합이 동작한다는 사실과 SSM의 성능 기여를 구분한다.

## 6. 현재 완료와 남은 작업

| 작업 | 상태 |
|---|---|
| Q0 일반/스트리밍 출력 동등성 | 구현·48개 실제 비교 완료 |
| 변화 감지·전체 갱신·출력 hold 대조 | 구현·평가 완료 |
| 선택적 SSM state + 특징 cache + 실제 encoder skip | 구현·600 step 학습·개발 평가 완료 |
| Q0 품질 + 유효한 가속의 동시 확보 | 미달성 |
| SSM 시간 상태의 독립 기여 입증 | 미달성 |
| 움직임 정렬/가림/깊이 출력 보존 학습 | 미실행 |
| L256·다중 seed·별도 개발 장면 | 미실행 |
| Pi 4/Nano B01 5W·10W·원고 모델 교체 | 미실행 / 보류 |

다음 우선순위는 **비갱신 프레임의 특징 예측 품질**이다. 지금은 임계만 높이면
깊이 경계가 손상된다. 캐시의 움직임 정렬과 decoder 출력 수준의 Q0 보존 학습을 각각
대조하고, 품질이 확보된 범위에서만 refresh 빈도를 낮춰야 한다.
기존 Qt의 단순 EMA 추가나 낮은 프레임 차이만으로 시간 안정성을 주장하지 않는다.

## 산출물 / 검증

- `sokkanaem/qstream.py`, `sokkanaem/qdelta_ssm.py`
- `scripts/qstream_study.py`, `scripts/qdelta_study.py`, `scripts/qdelta_memory_ablation.py`
- `tests/test_qstream.py`, `tests/test_qdelta_ssm.py`
- `work_dirs/qstream_20260908/`: parity, protocol, L32 데이터·예측·정책별 telemetry.
- `work_dirs/qdelta_20260908/`: teacher cache, adapter, 평가·시간 상태 ablation.
- 관련 회귀 테스트 **55개 통과**. 실제 Q0 parity 48개 비교는 별도 실행했다.
- 원본 Q0 가중치, 기존 qmodel/ssm/detector 구현과 논문 final은 수정하지 않았다.

```bash
HF_HUB_OFFLINE=1 python scripts/qstream_study.py --out work_dirs/qstream_repro
HF_HUB_OFFLINE=1 python scripts/qdelta_study.py --parent work_dirs/qstream_repro --out work_dirs/qdelta_repro
HF_HUB_OFFLINE=1 python scripts/qdelta_memory_ablation.py --run work_dirs/qdelta_repro
python -m pytest -q tests/test_qstream.py tests/test_qdelta_ssm.py
```

새 모델은 개발 프로토타입이다. 현재 결과를 논문의 완료된 제안 모델 성능으로 사용하지 않는다.
