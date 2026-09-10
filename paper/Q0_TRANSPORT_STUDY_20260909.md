# 이전 깊이의 국소 위치 선택·이동 실험

후속: [RGB 대응 기반 깊이 캐시 이동](Q0_RGB_FLOW_STUDY_20260909.md).
학습 없는 nearest 출력 이동 대조가 개발 품질·가속 기준을 통과했으며, 아래 transport 실패 기록은 유지한다.

2026-09-09. [두 프레임 보정 실험](Q0_TWOFRAME_REPAIR_STUDY_20260909.md)의 후속이다.

**깊이 값의 제한적 증감에서 이전 깊이의 위치 선택으로 표현을 확장했다.
실제 위치 선택은 발생했지만 정확도·거칠기가 악화되어 현재 후보는 미채택이다.**
단일 선택/혼합 선택/선택 제거 대조와 교사 참조 진단까지 완료했다.

## 1. 학습 자료 확대와 공통 조건

- 이전96개 생략 training frame 대신 기존 training cache 전체의192클립·768프레임을 사용했다.
  TUM/Bonn/VKITTI2 각64클립이며, K2 생략384프레임으로 보정기를 학습했다.
- Training과 L32/L256 development의 파일 경로 교집합0, reserved final sequence 제외를 확인했다.
- 단순 residual 대조군과 transport 후보에 같은 확대 자료·클립 순서·seed736·1,200step·batch4·lr.0005를 적용했다.
  총2,400step 추가 학습. 입력은 현재/이전 RGB와 현재 coarse/이전 깊이다.
- Base는 같은 동결 Q0+feature adapter K2이며 pooled/미사용 post-encoder CLS delta를 사용하지 않는다.
  매 전체 갱신에서 SSM h는 초기화한다. 이번 효과는 SSM 장기 기억의 기여가 아니다.
- 공통 손실은5 Q0log-depth+10 Q0log-gradient+20 trainingGT-flat excess다.
  Residual은.1 오류 위치BCE, transport는.1 후보CE를 추가한다.
- 출력 표현과 보조 감독이 달라 **정확한 단일 요인 대조는 아니다**.
  동일 backbone 계열이며 파라미터는 residual22,450, transport23,283이다.
- 같은 개발 L32 256프레임/L256 1,024프레임을 재사용했다. 독립 장면/final 검증이 아니다.
- Q0 inference518/score256, RTX4090 FP32. 기존 모든 품질 gate와 평균 시간≥5% 감소 기준 유지.

## 2. 새 출력 표현

Transport는 현재 coarse 예측1개와 이전 깊이의49개 위치를 합쳐 픽셀마다50개 후보를 만든다.
이동 offset은 각 축 `(-4,-2,-1,0,1,2,4)`px이며, 경계 밖은 replicate padding한다.

- **Hard:** 가장 높은 logit의 후보 하나를 직접 선택한다. 후보를 평균내지 않는다.
- **Soft:** 같은 checkpoint에서 확률 가중 log-depth 평균을 출력하는 사후 대조다.
  별도로 soft 모델을 학습한 것은 아니다.
- **Disabled:** 같은 checkpoint에서 후보 선택만 제거하고 coarse를 사용한다.
- 선택 뒤±.03 log-depth residual을 허용하며 최종 출력은 Q0 거리 범위로 clamp한다.
- 초기 hard 출력은 coarse와 정확히 같다. 전체 갱신 프레임은 보정하지 않는다.
- Hard 학습은 soft surrogate의 straight-through gradient를 사용한다.
  Argmax의 정확한 미분이 아니며 추정 편향이 있을 수 있다.

Training 후보 label은 **현재 Q0 깊이와 log-depth 차이가 가장 작은 후보**다.
Coarse보다 오차를.005 이상 줄이지 못하면 coarse label을 유지한다.
이 label은 실제 optical flow 또는 같은 물체의 대응 위치 정답이 아니다.
서로 다른 물체/표면도 비슷한 깊이를 가질 수 있으며, 새로 드러난 표면은 후보에 없을 수 있다.

## 3. L256 결과

| 구성 | Raw AbsRel↓ | 경계 F1↑ | Flat-TV 상대 증가 | 평균 ms | 판정 |
|---|---:|---:|---:|---:|---|
| Q0 Dense | .159183 | .614665 | 기준 | 약7.5 | 기준 |
| 같은 SSM K2 base | .159216 | .611583 | +1.513% | 4.825 | Flat-TV 실패 |
| 확대 자료 residual | .159210 | .611401 | +1.405% | 4.919 | Flat-TV 실패 |
| Hard 위치 선택 | .165526 | .609516 | +25.730% | 4.993 | 정확도·경계·Flat-TV 등 실패 |
| 같은 가중치 Soft 혼합 | .160159 | .516117 | +11.588% | 4.998 | 경계·Flat-TV 등 실패 |
| 같은 가중치 위치 선택 제거 | .159115 | .611353 | +1.483% | 4.979 | Flat-TV 실패 |
| SSM 없는 hold+Hard 선택 | .165501 | .609808 | +25.683% | 4.023 | 정확도·Flat-TV 등 실패 |

Hard 선택의 경계F1 감소도.005를 조금 초과한다. 위치 선택을 제거하면 손상이 크게 줄어,
현재 선택 경로가 오차 증가에 기여하는 것을 확인했다.
Soft 혼합은 정확도를 일부 회복하지만 경계를 크게 흐리게 했다.
이는 같은 hard 학습 가중치의 대조이며 soft 학습 일반의 한계를 증명한 것은 아니다.

L32도 hard 선택은 raw .133617, 경계F1 .650974, flat-TV+16.307%로 실패했다.
확대 자료 residual과 위치 선택 제거 대조는 L32를 통과하지만 L256은 통과하지 못했다.
이번 자료 확대만으로 품질이 해결되거나 데이터 양의 효과가 독립 입증됐다고 주장하지 않는다.

시간은 GPU-resident 동기화 Python 호출의 단일 평가값으로 전처리/모델을 포함하고 IO/H2D를
제외한다. 단일 평가의 작은 방법 간 시간 차이와 실기기 pipeline 가속을 주장하지 않는다.

## 4. 실제 선택과 교사 참조 진단

| Hard 선택 진단 | L32 | L256 |
|---|---:|---:|
| 생략 픽셀 중 coarse 이외 후보 선택 | 6.30% | 11.98% |
| 생략 픽셀 중 0이 아닌 위치 이동 선택 | 6.30% | 11.98% |
| Training과 같은 교사 후보 label 일치율 | 32.62% | 24.80% |

Training label은60.01%가 coarse 이외 후보였다. Inference에서 모든 픽셀이 coarse로
붕괴한 것은 아니지만, 현재 label 규칙과 선택의 일치율은 낮다.
여러 후보가 비슷한 깊이를 갖기 때문에 label 일치율을 실제 motion 정확도로 해석하지 않는다.

### 배포할 수 없는 교사 참조 선택

개발 생략 프레임에서도 dense Q0를 알고 있다고 가정하여 같은 training label 규칙으로
후보를 선택하는 oracle 진단을 추가했다. 실행 가능한 모델이 아니므로 속도를 부여하지 않았다.

- L256 raw .158795, 경계F1 .624089로 Q0 대비 일부 지표가 개선됐다.
- 그러나 flat-TV .033070, Q0 대비약+6.72%로 품질 gate는 실패했다.
- 즉 픽셀별 깊이 오차가 작은 후보를 독립적으로 고르는 것만으로 공간적으로 일관된
  깊이장을 보장할 수 없다. 이 선택은 GT 기반 metric의 최적해도 아니다.
- 이 결과는 이동 구조 일반이 불가능하다는 증명이 아니라, 현재 후보/감독 규칙이
  실제 영상 대응과 표면 일관성을 충분히 표현하지 못한다는 진단 근거다.

## 5. 검증·결정

- 관련 회귀 테스트 **76 passed**.
- Hard 후보의 전체 갱신640프레임을 저장된 Q0와 비교하여 출력 차이0 확인.
- 후보 offset 방향/경계 padding, tie 시 coarse 우선, hard 초기 출력 동일성,
  학습 gradient, hard/disabled 선택 동작을 검사했다.
- 두 실행의 source/원본 Q0/adapter/training/development hash 계약 재검사 완료.
- 개발 GT는 점수와 사후 진단에만 사용했다. 학습 GT는 원래 training cache의 flat mask에만 사용했다.
- 원본 모델·논문 원고·final 평가 결과는 수정하지 않았다.

이번 위치 선택 단계의 구현·학습·대조·진단은 완료했으며 **채택은 보류**한다.
다음 우선순위는 후보 수 확대가 아니라 대응의 타당성과 공간적 일관성 검증이다.

1. RGB/특징의 실제 대응 정보로 이동을 추정하고, 깊이 유사도만으로 정한 label과 분리 비교한다.
2. 같은 표면 안에서는 이동이 일관되도록 하되 물체 경계에서는 다른 이동을 허용하는 제약을 검증한다.
3. 가림 해제·범위 밖 이동·낮은 대응 신뢰도에서 현재 RGB 재예측 또는 Q0 전체 갱신이 필요하다.
4. 고정 비용의 cache-only/비순환 대조를 유지하고, 핵심 품질 gate 통과 후 독립 장면/다중 seed/실기기로 확대한다.

## 산출물 / 재현

```text
sokkanaem/qtransport.py
scripts/qtransport_study.py
scripts/audit_qtransport.py
tests/test_qtransport.py
work_dirs/qtransport_residual_20260909/
work_dirs/qtransport_hard_20260909/
work_dirs/qtransport_audit_20260909/audit.json
```

Residual 실행 폴더에 공통 확대 training cache, 각 학습 폴더에 protocol/checkpoint/history와
L32/L256 prediction·시간·호출수·지표를 저장했다. 새 출력 경로로만 재실행한다.

```bash
HF_HUB_OFFLINE=1 python scripts/qtransport_study.py --out work_dirs/qtransport_residual_repro --mode residual
HF_HUB_OFFLINE=1 python scripts/qtransport_study.py --out work_dirs/qtransport_hard_repro --mode transport --cache work_dirs/qtransport_residual_repro/training_cache.pt
python scripts/audit_qtransport.py --out work_dirs/qtransport_audit_repro
```

Audit 대상은 보존된20260909 출력 폴더를 명시 참조하며 `_repro` 결과로 자동 변경되지 않는다.
