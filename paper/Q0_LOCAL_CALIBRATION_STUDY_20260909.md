# Q0 거리 보정 입력 분리·경계 보호 국소 보정

2026-09-09. [평탄 영역·상태 유지 실험](Q0_FLAT_STATE_STUDY_20260908.md)의 후속.

**거리 보정용 pooled delta가 일부 정확도 손실에 기여하는 것을 분리 대조로 확인했다.
경계 보호 국소 보정도 구현·학습했으나 L256 품질 gate는 여전히 실패하여 모델은 미승격이다.**

후속: [두 프레임 오류 보정](Q0_TWOFRAME_REPAIR_STUDY_20260909.md)에서 현재/이전 RGB와
이전 깊이를 사용하는 경로를 matched 학습했다. 이 문서의 당시 결과는 유지한다.

## 1. 비교 범위

- 기존 Q0와 논문 모델을 수정하지 않고 새 module/script만 추가했다.
- Q0 inference518/score256, RTX4090 FP32, K2 encoder50% 생략.
- L32 개발256프레임, L256 개발1,024프레임. 이미 여러 번 사용한 개발 자료다.
  새 장면 또는 독립 final-test 성과가 아니며 통계적 비열등성 검정도 아니다.
- 기존 raw/edge AbsRel·overshoot·flat-TV 상대 증가≤1%, 경계F1 감소≤.005,
  source별 raw 증가≤2%, 평균 latency≥5% 감소 조건을 유지했다.

## 2. CLS와 pooled calibration 입력 분리

`qflat_carry_flat_20260908`의 **동일한 가중치**에서 post-encoder CLS delta,
raw pooled delta를 각각 제거했다. Patch delta와 K2 h-carry 정책은 동일하다.
재학습된 네 모델의 비교가 아니라 같은 모델의 사후 제거 대조다.

| 조건 | L32 Raw AbsRel↓ | L32 품질 | L256 Raw AbsRel↓ | L256 경계 F1↑ | L256 Bonn raw 변화 |
|---|---:|---|---:|---:|---:|
| Q0 Dense | .128127 | 기준 | .159183 | .614665 | 기준 |
| 원래 delta 모두 사용 | .128922 | source 실패 | .160001 | .604630 | +3.313% |
| CLS delta만 제거 | .128922 | source 실패 | .160001 | .604630 | +3.313% |
| pooled delta만 제거 | .127985 | 통과 | .158584 | .604632 | +1.984% |
| CLS·pooled 모두 제거 | .127985 | 통과 | .158584 | .604632 | +1.984% |

Pooled 입력을 마지막 전체 갱신 값으로 유지하면 이번 checkpoint의 raw/source 문제는
개선된다. 하지만 L256 경계F1은 허용 범위 아래라 전체 gate는 실패한다.
이 결과를 pooled 입력 갱신이 항상 잘못이라는 결론으로 일반화하지 않는다.
정규화된 decoder disparity도 metric depth에 영향을 주므로 pooled 고정은 모든 거리 변화를
고정하는 것과 다르다.

### CLS 관련 정정

실제 import된 `DepthAnythingReassembleStage.forward`는 `hidden_state[:, 1:]`로
CLS를 제외한다. 현재 adapter가 **encoder 이후**의 CLS에 더하는 delta는 이 decoder에서
사용되지 않는다. 네 stage의 post-encoder CLS를 제거한 경우 전체1,280프레임에서 출력 차이0을
확인했고, pooled 제거 조건과 둘 다 제거한 조건 사이도 차이0이었다.

따라서 이전 실험 문서의 'CLS delta가 전역 출력을 바꿀 수 있다'는 일반적 설명은
**현재 사용 중인 Q0 decoder에 대해서는 적용되지 않는다.** Pooled delta는 사용된다.
이것은 Q0 encoder 내부 attention의 CLS 토큰이 불필요하다는 뜻은 아니다.
설치된 decoder source 경로와 hash는 `qlocal_audit_20260909/audit.json`에 기록했다.
기존 모델 구조/checkpoint를 즉시 잘라내거나 과거 실험 수치를 바꾸지는 않았다.

## 3. 국소 보정 설계와 matched 학습

경계가 이미 약해진 flat-trained feature 후보에 후처리를 붙이는 대신,
경계가 더 잘 유지됐던 `qquality_feature_20260908` K2를 **공통 동결 base**로 사용했다.

- RGB/log-depth/깊이 gradient를 입력받는3,217parameter CNN.
- 보정량은±.03 log-depth로 제한, 전체 갱신 프레임은 보정하지 않는다.
- 두 조건: 모든 픽셀 보정 vs 경계 보호 국소 보정. 동일 초기화/seed736/600step/lr.001.
- 동일48개 training clip의 실제 K2 생략96프레임으로 학습했다. Q0와 adapter는 동결했다.
- 손실은5 Q0log-depth+10 Q0log-gradient+20 trainingGT-flat excess gradient.
- Training cache 경로를 GT와 연결하고 training/development 경로 교집합0을 확인했다.
- 보호 mask는 **현재 RGB와 현재 coarse prediction만** 사용한다. 개발 GT는 추론에 사용하지 않는다.
  RGB gradient>.12 또는 predicted log-depth gradient>.02를 경계 후보로 삼고5×5 팽창한다.
  해당 위치의 출력은 `torch.where`로 원래 coarse depth를 정확히 복사한다.
- GT 경계를 정확히 판별하는 mask는 아니다. 보호된 raw depth 값이 같더라도
  전역 정규화가 포함된 metric까지 동일하다고 보장하지 않는다.
- 같은 학습 refiner를 SSM 없는 K2 whole-output hold에 붙인 사후 대조도 수행했다.
  별도로 hold용 refiner를 재학습한 공정한 최적 성능 비교는 아니다.

## 4. L256 결과

| 구성 | Raw AbsRel↓ | 경계 F1↑ | Flat-TV 상대 변화 | 평균 ms | 판정 |
|---|---:|---:|---:|---:|---|
| Q0 Dense | .159183 | .614665 | 기준 | 약7.51 | 기준 |
| 기존 feature SSM K2 | .159068 | .611582 | +1.513% | 4.816 | Flat-TV 실패 |
| 전체 픽셀 보정 | .159113 | .609262 | +1.252% | 4.869 | 경계·Flat-TV 실패 |
| 경계 보호 국소 보정 | .159071 | .611580 | +1.508% | 4.943 | Flat-TV 실패 |
| Hold K2+국소 보정 | .159214 | .612038 | +1.597% | 3.996 | Flat-TV 실패 |

국소 보정은 경계를 거의 그대로 유지했으나 거칠기도 거의 바꾸지 못했다.
전체 픽셀 보정은 거칠기를 일부 줄이면서 경계 손실을 더했다.
L32에서는 두 보정 모두 품질·가속 screen을 통과하지만 L256까지 통과하지는 못했다.
이번에는 평균 latency 단일 평가값을 기록했으며 작은 방법 간 시간 차이에 통계적 의미를
부여하지 않는다. GPU-resident Python 호출이며 IO/H2D와 실제 카메라 pipeline은 제외한다.

## 5. 왜 국소 보정의 효과가 작은가: mask 감사

실제 저장된 prediction을 대상으로 다음 사후 진단을 수행했다.
GT는 이 원인 진단에만 사용했으며 inference mask를 바꾸지 않았다.

| 생략 프레임 픽셀 단위 진단 | L32 | L256 |
|---|---:|---:|
| 전체 픽셀 중 수정 허용 비율 | 57.01% | 55.40% |
| GT-flat 픽셀 중 수정 허용 비율 | 76.74% | 74.86% |
| GT-flat에서 Q0보다 큰 gradient 초과량 중 보호 위치의 비중 | 86.89% | 90.49% |

전 영역을 막은 것은 아니다. 수정할 수 있는 픽셀은 절반 이상이지만,
해결할 초과 gradient가 보호 위치에 집중돼 있었다.
위 초과량은 normalized-disparity의 양의 gradient 차이를 픽셀별 합산한 사후 진단이며
source-balanced flat-TV 점수와 동일한 집계가 아니다. Forward-difference 위치에 귀속된
gradient이므로 보호 위치 주변의 변경이 전혀 영향을 줄 수 없다는 수학적 주장도 아니다.

오래된 경계, motion mismatch, 현재 RGB 무늬, coarse 깊이의 오경계 등을 이 진단 하나로
구분하지는 못한다. 다만 '예측에 경계가 있으면 그대로 보호'하는 규칙만으로 해결하기
어려운 위치에 오차가 집중돼 있음을 보여준다.

## 6. 검증과 결정

- 관련 회귀 테스트 **69 passed**.
- 국소 보정 후보의 전체 갱신640프레임은 저장된 Q0와 출력 차이0.
- 실제 생략640프레임의 모든 보호 픽셀은 coarse prediction과 차이0.
- CLS 제거의 출력 동일성, train/development/원본 모델/source hash 계약 확인.
- 추가 학습은 두 refiner 각각600step, 총1,200step. 분리 대조는 재학습 없이 수행했다.
- 기존 논문 모델, manuscript/draft, final 결과는 변경하지 않았다. GPU 학습·평가 작업은 종료했다.

다음 우선순위는 **현재 프레임의 실제 경계와 재사용된 잘못된 경계를 구분하는 보정 경로**다.
현재 RGB+이전 RGB+이전 깊이를 함께 보고 대응 신뢰도를 학습하는 방법과,
거리 보정용 pooled 입력을 안정적으로 유지하는 정책을 분리해서 검증할 필요가 있다.
Pooled 고정과 국소 보정을 서로 다른 checkpoint에서 시험했으므로 조합 효과는 아직 미검증이다.
단순 마스크 임계 완화나 품질 허용치 변경으로 통과시키지 않는다.
독립 장면/다중 seed/실기기 검증은 여전히 남아 있으며 이번 결과를 SSM 고유 이득으로 주장하지 않는다.

## 산출물 / 재현

```text
sokkanaem/qseparated.py
sokkanaem/qlocal.py
scripts/qseparated_study.py
scripts/qlocal_study.py
scripts/audit_qlocal.py
tests/test_qlocal.py
work_dirs/qseparated_20260909/
work_dirs/qlocal_unrestricted_20260909/
work_dirs/qlocal_protected_20260909/
work_dirs/qlocal_audit_20260909/audit.json
```

각 study 폴더에는 protocol, predictions, 전체 지표/시간/telemetry가 있으며
학습 run에는 refiner checkpoint와 history도 있다.

```bash
HF_HUB_OFFLINE=1 python scripts/qseparated_study.py --out work_dirs/qseparated_repro
HF_HUB_OFFLINE=1 python scripts/qlocal_study.py --out work_dirs/qlocal_unrestricted_repro
HF_HUB_OFFLINE=1 python scripts/qlocal_study.py --out work_dirs/qlocal_protected_repro --protected
python scripts/audit_qlocal.py --out work_dirs/qlocal_audit_repro
```

부모 모델/data와 audit 대상은 보존된 날짜별 폴더를 명시 참조한다.
새 `_repro` 결과로 audit 대상을 자동 변경하는 전체 파이프라인은 아니다.
