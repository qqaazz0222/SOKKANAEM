# 깊이 품질 고도화: 단계별 실행 기록

날짜: 2026-09-08. **개발 실험이며, 투고 논문의 결과를 대체하지 않는다.**

## 1. 이전 기록에서 확인한 사항

- `REPORT.md` §§4.44–4.47, `PLAN.md`, `PROGRESS.md`, `NOVELTY.md`,
  `outputs/sharpness/suite.jsonl`, queue24–26 및 실제 체크포인트를 검토했다.
- D1 경계 손실 강화는 경계 검출과 함께 overshoot/평탄부 잡음을 증가시켰다.
- v11 교사 학습 4k, QT-balanced, M0 장기 학습을 이미 시도했다.
  같은 처방을 새로운 성공 방법처럼 제시하지 않는다.
- M0-dense는 6400/8000에서 중단된 실행이므로 완료된 실패 실험이 아니다.
- 기존 Q0 결과는 518px 추론이며, native 256px와 연산량·정확도를
  동일 조건인 것처럼 비교하지 않는다. 이번 재평가는 native 후보만 수행했다.
- 다운샘플링 GT 재구성은 학습형 디코더의 성능 상한이 아니다.
  spread 손실이 과도한 선명화의 원인이라는 해석도 아직 가설이다.

## 2. 실행 단계 및 판정

| 단계 | 실제 수행 내용 | 결과 / 다음 조건 |
|---|---|---|
| 0. 평가 조건 통일 | v11, D1-boundary3, v11-teacher를 동일 개발 16클립·128프레임에서 재평가 | 완료. 과거 집계와 다른 표본이므로 절대 수치를 혼합하지 않음 |
| 1. 기본 예측 보존형 보정 | v11 동결, RGB+디코더 특징 기반 2,634파라미터 모듈, 두 손실 구성 각각 400 step | 완료. 두 후보 모두 품질 gate 실패 |
| 2. 경계 국소 보정 | 저주파 보정 제거, coarse 깊이 경계로 게이트, 두 구성 각각 800 step | 완료. 경계 F1 개선, overshoot gate 실패 |
| 3. 경계 안전성 개선 | 신뢰할 수 있는 GT 경계에서 보정 방향·범위 제약을 추가하는 대조 실험 | 미실행. 단순 경계 손실 증가는 반복하지 않음 |
| 4. 검증 확대 | 더 큰 개발 표본, L32/L256, 여러 seed, 경계/가림별 분석 | 미실행. 후보 고정 후 실시 |
| 5. 배포 검증 | 실제 FP32 특징으로 스트리밍 통합, 지연·메모리·시간 일관성, Pi 4 및 Nano B01 5W/10W | 미실행. 작은 파라미터 수만으로 속도·전력 우위를 주장하지 않음 |
| 6. 논문 반영 | 새 모델 고정 및 미사용 평가 계획 수립 후 정량·정성 결과 갱신 | 보류. 기존 테스트를 새 모델 선택에 사용하지 않음 |

## 3. 이번 실험의 고정 조건

- 원본: `work_dirs/v11-longclip-spread-s0/latest.pt`; 원본 체크포인트와
  기존 모델·손실·평가 구현, 논문 및 최종 테스트 결과는 수정하지 않았다.
- 학습: TUM/Bonn/VKITTI2 각 16클립, 4프레임씩, 총 48클립·192프레임.
  기존 holdout 제외 및 reserved-test 차단 적용. 증강 없는 소규모 화면 검사다.
- 개발: `manifests/acc_real_L8.json`의 4개 장면에서 각 4클립,
  총 16클립·128프레임. 학습/개발 파일 교집합 없음 확인.
- 256×256, 원래 K30 스트리밍 설정, 클립 시작마다 상태 초기화.
  학습 클립 길이 4와 평가 길이 8은 장기 상태 검증을 대신하지 않는다.
- AbsRel(raw): GT 보정 없는 직접 metric 깊이. AbsRel(median): 클립당
  단일 median 정렬. 경계 지표에는 모든 후보의 정렬 전 깊이를 동일하게 전달한다.
- 집계: 프레임을 포함한 클립 지표 → 장면 평균 → source 평균 → source 균등 평균.
  TUM 1장면과 Bonn 3장면이므로 장면 전체의 단순 평균과는 다르다.
- 경계 지표는 기존 `sharpness_scores` 정의를 유지했다. undefined 지표는
  조용히 제외하지 않고 실행 실패로 처리한다. flat-TV 비율은 센서 GT의 양자화로
  크게 불안정할 수 있어 절대 flat-TV를 gate에 사용한다.
- 원본 추론 RGB/예측 FP32, 보정기 캐시 RGB/디코더 특징 FP16→FP32.
  두 모듈의 초기 출력은 캐시 기반 원본과 정확히 동일함을 확인했다.
  실제 스트리밍 FP32 특징과 캐시 간 수치 동등성은 아직 별도 검증하지 않았다.
- seed 20260908, batch 8, AdamW lr=0.001. 각 구성 동일 초기화·샘플 순서.
  다중 seed 또는 비트 단위 재현성을 검증했다는 뜻은 아니다.
- 보정 식: `depth_new = depth_base * exp(delta)`.
  `|delta| <= 0.15`, 즉 이론상 원본 대비 약 0.861–1.162배 이내.
  0 초기화로 원본 예측 보존. 이 범위 제약 자체가 정확도 보장은 아니다.
- 경계 국소 모듈은 보정에서 5×5 국소 평균을 빼고, coarse log-depth
  기울기로 게이트한다. RGB 텍스처만으로 완전히 평탄한 깊이를 바꾸지 않는다.
  반대로 coarse에 없는 물체 경계를 복원하지 못할 수 있다.

## 4. 동일 조건 개발 결과

작은 적응적 개발 실험이다. 논문 최종 성능 또는 통계적으로 확정된 개선이 아니다.

| 후보 | AbsRel raw ↓ | AbsRel median ↓ | 경계 F1 ↑ | Overshoot ↓ | Flat-TV ↓ |
|---|---:|---:|---:|---:|---:|
| 기존 v11 | 0.113399 | 0.084217 | 0.338671 | 0.207601 | 0.027413 |
| 과거 D1-boundary3 | 0.111903 | 0.083417 | 0.448716 | 0.253431 | 0.031014 |
| 과거 v11-teacher | 0.112845 | 0.087597 | 0.349134 | 0.206957 | 0.030912 |
| 신규 기본 보정 / metric | 0.113789 | 0.084220 | 0.340289 | 0.214838 | 0.027686 |
| 신규 기본 보정 / metric+gradient | 0.114001 | 0.084208 | 0.339190 | 0.215656 | 0.027617 |
| 신규 국소 경계 / λ=0.5 | 0.113335 | 0.083794 | 0.358103 | 0.213044 | 0.027214 |
| 신규 국소 경계 / λ=2.0 | 0.113402 | 0.083786 | 0.376973 | 0.214760 | 0.027379 |

국소 λ=0.5는 경계 F1 약 5.74% 상대 개선, raw AbsRel 약 0.056% 감소,
flat-TV 약 0.73% 감소를 보였으나 overshoot는 약 2.62% 증가했다.
정확도 변화는 매우 작아 개선이 확정되었다고 표현할 수 없다.
λ=2.0은 경계 F1 약 11.31% 상대 개선이나 raw 정확도 및 overshoot 조건을 실패했다.
경계 영역 AbsRel도 두 국소 후보에서 소폭 악화했다. 경계 F1 증가가 곧
경계 픽셀의 깊이 정확도 향상을 의미하지 않는다.

**채택한 신규 모델: 없음.** Gate는 raw/median 정확도 비악화, F1 +0.005 이상,
overshoot·flat-TV 비악화, gradient ratio ≤1, source별 raw 오차 증가 ≤1%를
모두 요구한다. 유의성 검정이나 배포 보장은 아닌 초기 안전 장치다.

## 5. 다음 우선 작업

1. λ=0.5의 형태를 시작점으로, 학습 GT의 유효 경계에서 기존 예측보다
   깊이 오차·국소 범위 위반이 증가하는 보정에 벌점을 주는 matched ablation.
   기존 전체 모델의 boundary/overshoot 손실 동시 강화와 구분한다.
2. 학습 장면·표본을 확대하고 센서 결측 경계, 기하 경계, RGB 텍스처를 분리한다.
   이번 유효 마스크 erosion은 센서 구멍 인접 제거일 뿐, 교사 신뢰도 검증은 아니다.
3. 사전 고정된 다른 개발 클립에서 같은 gate를 통과해야 L32/L256와 다중 seed로 이동한다.
   이미 관찰한 개발 16클립만을 반복 최적화해 성공을 선언하지 않는다.
4. 경계 안전성과 장기 안정성이 확보된 이후에만 실제 장치 측정 및 원고 교체.

## 6. 산출물 및 재현

- `sokkanaem/bounded_refiner.py`, `sokkanaem/edge_local_refiner.py`
- `scripts/quality_refinement.py`, `scripts/quality_refinement_followup.py`
- `tests/test_bounded_refiner.py`, `tests/test_quality_refinement_screen.py`
- `work_dirs/quality_refinement_20260908/`: protocol/selection, 학습·개발 캐시,
  기준 지표, 2개 후보 가중치, per-clip/source 지표, 학습 로그 JSON.
- `work_dirs/quality_refinement_local_20260908/`: 역사 후보 동일 조건 비교,
  국소 보정 2개 가중치, protocol 및 모든 gate 판정.
- 프로토콜에 체크포인트·config·manifest·코드·선택 목록 해시를 기록했다.
  입력 영상 파일 자체의 전체 내용 해시를 기록한 완전한 데이터 아카이브는 아니다.
- 실행 후 비유한 GT의 masked-loss 안전 처리를 보강했다. 실험에 사용한 이전
  `bounded_refiner.py`는 첫 실행 폴더의 `bounded_refiner.executed.py`에 보존했다.
  변경은 NaN/Inf GT 처리에 한정되며, 이번 캐시 GT는 모두 유한함을 확인했다.
- 관련 회귀 테스트 39개 통과: 새 모듈·gate·집계, 기존 sharpness/alignment 및
  paper protocol. 전체 저장소 테스트 또는 엣지 장치 테스트를 의미하지 않는다.

```bash
python scripts/quality_refinement.py --out work_dirs/quality_refinement_reproduction --steps 400
python scripts/quality_refinement_followup.py --parent work_dirs/quality_refinement_reproduction --out work_dirs/quality_refinement_local_reproduction --steps 800
python -m pytest -q tests/test_bounded_refiner.py tests/test_quality_refinement_screen.py
```

출력 경로가 이미 있으면 중단하여 이전 실행을 덮어쓰지 않는다.
