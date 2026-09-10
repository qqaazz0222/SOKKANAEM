# 두 프레임 기반 캐시 오류 보정 실험

2026-09-09. [국소 보정 실험](Q0_LOCAL_CALIBRATION_STUDY_20260909.md)의 후속이다.

**현재·이전 RGB와 이전 깊이를 사용하는 보정 경로를 구현하고 matched 대조군과 학습했다.
오류 위치 탐지에는 개선이 있었지만, 깊이 품질의 일관된 개선이나 L256 gate 통과는 달성하지 못했다.**
기존 논문 모델과 최종 평가 결과는 유지한다.

후속: [이전 깊이 위치 선택 실험](Q0_TRANSPORT_STUDY_20260909.md)에서 학습 자료를 확대하고
위치 선택/혼합/제거 대조를 수행했다. 이 문서의 당시 결과는 유지한다.

## 1. 구현과 비교 계약

- Base: `qquality_feature_20260908`의 동결 adapter+Q0, K2 encoder50% 생략.
- 두 조건 모두 post-encoder CLS delta와 raw pooled delta를 사용하지 않는다.
  Pooled 입력은 마지막 Q0 전체 갱신 값을 유지한다. 이전 flat-trained checkpoint의 pooled
  제거 효과가 이 checkpoint에도 그대로 성립한다고 가정하지 않고 새 base를 함께 평가했다.
- 새 경로는 현재 RGB3ch, 이전 RGB3ch, RGB 절대차3ch, coarse log-depth1ch,
  이전 log-depth1ch의11채널을 사용한다.
- Full/half-resolution convolution과 dilation을 결합한 **22,450parameter** 네트워크로
  보정 delta와 오류 위치 gate를 출력한다. 추가 optical-flow 모델은 없다.
- 보정은 `coarse * exp(.1*tanh(delta)*sigmoid(gate))`, 이후 Q0 거리 범위로 clamp한다.
  강제적인 경계 보호 mask는 제거했다. 전체 갱신 출력에는 보정하지 않는다.
- **현재 정보 대조군:** 이전 RGB를 현재 RGB로, 이전 깊이를 coarse로 대체한다.
  채널 수·구조·파라미터·초기화·학습량은 같다. Base 자체는 캐시를 쓰므로 모든 시간 정보를
  없앤 모델이 아니라, 보정기에 추가된 명시적 이전 프레임 입력을 제거한 대조군이다.
- Gate는 Q0와 coarse의 log-depth 차이>.015 또는 log-gradient 차이 합>.02인 training 위치를
  target으로 학습한다. GT 경계, 실제 물체 이동, 가림 해제를 직접 판별하는 정답 label이 아니다.
- Gate가 계산 생략을 수행하는 것은 아니다. CNN은 생략 프레임 전체에서 실행되고 gate는
  출력 보정량만 조절한다.

## 2. 학습·평가

- 동일48개 실제 training clip·192프레임 중 K2 생략96프레임을 사용했다.
  인접한 실제 이전 RGB/깊이를 연결했고 임의 clip을 이어붙이지 않았다.
- 두 조건 각각1,200step, seed736, batch4, AdamW lr.0005. 총2,400step.
- 손실: 5 Q0log-depth+10 Q0log-gradient+20 trainingGT-flat excess+.1 오류 위치BCE.
- Q0와 adapter는 동결. 개발 GT는 학습 및 추론에 전달하지 않았다.
- Training/development 경로 교집합0을 확인했다.
- 기존 L32 개발256프레임과 L256 개발1,024프레임을 재사용했다.
  이미 관찰한4장면의 개발 결과이며 독립 장면/final-test/다중 seed 검증은 아니다.
- Q0 inference518/score256, RTX4090 FP32. 기존 품질 기준과 평균 시간≥5% 감소 조건 유지.

## 3. 결과

L256에서 Q0 raw AbsRel .159183, 경계F1 .614665, flat-TV .030988이 기준이다.

| 모델 | Raw AbsRel↓ | 경계 F1↑ | Flat-TV 상대 증가 | 평균 ms | 품질 gate |
|---|---:|---:|---:|---:|---|
| Q0 Dense | .159183 | .614665 | 기준 | 약7.52 | 기준 |
| Pooled 고정 SSM K2 base | .159216 | .611583 | +1.513% | 4.832 | Flat-TV 실패 |
| 현재 정보 보정기 | .159416 | .611432 | +1.404% | 4.927 | Flat-TV 실패 |
| 두 프레임 보정기 | .159425 | .611536 | +1.459% | 4.928 | Flat-TV 실패 |
| 두 프레임 학습 가중치에서 이전 입력 제거 | .159347 | .611499 | +1.452% | 4.925 | Flat-TV 실패 |
| SSM 없는 hold+두 프레임 보정기 | .159420 | .611986 | +1.549% | 3.968 | Flat-TV 실패 |

두 프레임 보정기는 평균 약1.53× 빠르고 경계 감소 허용치.005도 만족하지만,
flat-TV 증가 허용치1%를 초과하므로 전체 성공으로 판정하지 않는다.
현재 정보 대조군보다 raw/flat-TV가 좋지 않고 같은 가중치의 이전 입력 제거 효과도 혼재한다.
Hold 대조는 같은 보정 가중치를 사후 적용한 것으로, hold 전용 재학습 결과는 아니다.

L32에서는 현재 정보/두 프레임 보정 모두 품질·가속 screen을 통과했다.
두 프레임 후보의 L32 raw .128395, 경계F1 .659781, 평균4.922ms다.
L32 통과만으로 모델을 승격하지 않는다.

시간은 동기화된 GPU-resident Python 호출, 전처리·detector·모델 포함, IO/H2D 제외다.
이번 단일 평가의 작은 시간 차이에 통계적 의미를 부여하지 않으며 실기기 pipeline 속도가 아니다.

## 4. 오류 위치 탐지와 실제 복원은 다르다

L256 생략 프레임에서 Q0-error label과 gate를 사후 비교했다.
아래 값은 전체 픽셀 가중 집계이며 GT-boundary metric 또는 calibrated uncertainty가 아니다.

| 오류 위치 gate 지표 | 현재 정보 | 두 프레임 |
|---|---:|---:|
| Precision, threshold .5 | .6808 | .7259 |
| Recall, threshold .5 | .4128 | .4805 |
| Brier score↓ | .2492 | .2366 |
| 평균 sigmoid gate | .4536 | .4647 |

두 프레임 입력은 이 오류 위치 label의 분류에는 도움이 됐지만,
그 위치에서 필요한 깊이 값을 복원하는 개선으로 이어지지 않았다.
Label이 정확한 motion/occlusion/현재 경계인지 별도로 검증한 것은 아니다.

### 보정량 제한 진단 — 배포 불가능한 oracle

생략 프레임에서도 dense Q0를 알고 있다고 가정하고,
각 픽셀을±.1 log-depth 범위 안에서 Q0에 가장 가깝게 이동시켰다.
이는 모델 성능이나 실현 가능한 가속이 아니라 **정답 참조 진단**이다. 속도를 부여하지 않았다.

- L256 생략 픽셀의6.60%는 Q0 값까지 이동하는 데±.1보다 큰 변화가 필요했다.
- 이 제한형 oracle도 L256 경계F1 .609114와 flat-TV에서 기존 gate를 실패했다.
- 픽셀별 Q0 거리 오차를 줄이는 것과 GT 경계/거칠기를 유지하는 것은 같지 않다.
  이 oracle은 GT-gradient metric의 최적해가 아니므로, 결과만으로 같은 보정 범위의
  모든 모델이 불가능하다고 증명한 것은 아니다.

현재 구조가 필요한 경계 이동을 단순 깊이 값의 제한적 증감으로 표현하는 데 불리할 수 있다는
진단 근거다. 보정량만 무조건 키우면 안전성이 개선된다는 뜻도 아니다.

## 5. 검증과 다음 결정

- 관련 회귀 테스트 **73 passed**.
- 두 학습 모델의 전체 갱신 출력 총1,280프레임에서 저장된 Q0와 차이0.
- 초기 보정 identity, 보정량 bound, 이전 입력 제거의 불변성, 이전 입력 사용 시 출력 변화,
  오류 target 생성, stream 상태와 반환 tensor의 독립성 테스트 완료.
- Source/원본 Q0/초기 adapter/training cache/GT/development hash 계약 재검사 완료.
- 원본 모델·decoder·SSM·검증 metric 구현은 수정하지 않았다.

이번 단계는 구현·학습·대조·진단까지 완료했지만 모델 승격은 보류한다.
다음 우선순위는 **오류 위치를 찾는 경로와 그 위치를 복원하는 표현의 분리**다.

1. 깊이 값만 증감하는 residual과, 이전 깊이의 위치 이동/국소 후보 선택을 비교한다.
   신규 영역은 이전 깊이만으로 복구할 수 없으므로 현재 RGB 재예측 또는 Q0 refresh가 필요하다.
2. 실제 현재 경계/잘못 재사용한 경계에 대한 대응 정보와 더 다양한 학습 장면을 확보한다.
   현재96개 생략 프레임을 반복 학습한 것으로 시간 대응을 충분히 학습했다고 보지 않는다.
3. 같은 비용의 cache-only/비순환 보정 대조를 유지한다. 현재 K2 base는 매 갱신 h를 초기화하므로
   이번 두 프레임 보정 효과를 SSM 장기 기억의 기여로 주장할 수 없다.

독립 개발 장면·다중 seed·Pi4·Nano B01 5W/10W 실측은 아직 남아 있다.

## 산출물 / 재현

```text
sokkanaem/qtemporal_refiner.py
scripts/qtemporal_refiner_study.py
scripts/audit_qtemporal_refiner.py
tests/test_qtemporal_refiner.py
work_dirs/qtemporal_current_20260909/
work_dirs/qtemporal_pair_20260909/
work_dirs/qtemporal_audit_20260909/audit.json
```

학습 폴더에는 protocol, 실제 training cache, checkpoint, history, L32/L256 prediction과
프레임별 호출수/시간/지표를 보존했다. 항상 새 output 경로로 실행한다.

```bash
HF_HUB_OFFLINE=1 python scripts/qtemporal_refiner_study.py --out work_dirs/qtemporal_current_repro
HF_HUB_OFFLINE=1 python scripts/qtemporal_refiner_study.py --out work_dirs/qtemporal_pair_repro --temporal
python scripts/audit_qtemporal_refiner.py --out work_dirs/qtemporal_audit_repro
```

부모 모델/data와 audit 대상은 보존된 날짜별 폴더를 명시 참조한다.
Audit은 새 `_repro` 학습 폴더로 자동 연결되지 않는다.
