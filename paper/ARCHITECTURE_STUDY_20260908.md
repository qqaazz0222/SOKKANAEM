# 구조 고도화 순차 실험 — 2026-09-08

## 판정 요약

출력부 대조(1단계)와 고해상도 세부 특징 경로의 독립 대조(2단계)를 실행했다.
**채택 gate를 통과한 후보는 없다.** 실패한 변경의 결합(3단계), 장기 영상 및
기기 측정(4·5단계)은 실행하지 않았다. 원본 v11 가중치·원고·최종 테스트는 유지했다.

이전 작은 잔차 보정 실험과 다른 개발 프레임을 사용했으므로 그 표의 절대 수치와
이번 표를 직접 혼합하지 않는다. 같은 기존 개발 장면에서 다른 프레임을 뽑은 것이며,
미지 장면 일반화나 새로운 최종 테스트는 아니다.

## 1. 사전 조건과 데이터

- 기준: `work_dirs/v11-longclip-spread-s0/latest.pt`, EMA 우선 로딩.
- 입력 256×256, 기존 K30 동작, 클립 시작마다 상태 초기화.
- 학습: TUM/Bonn/VKITTI2 각 64클립 × 4프레임 = 192클립·768프레임.
  기존 holdout 제외와 reserved-final-test 차단 적용. clip stride 4, 증강 없음.
- 개발: 기존 `acc_real_L8.json`의 4개 장면에서 각 4클립 = 16클립·128프레임.
  이전 시각화/보정 평가의 개발 프레임과 파일 교집합이 없도록 선택했다.
  학습/개발 파일 교집합도 차단했다. 선택 목록은 학습 전에 저장했다.
- 고정된 backbone/decoder의 1/2 해상도 16채널 특징을 FP16로 캐시한 후 FP32로 사용.
  원본 깊이는 FP32. RGB 세부 경로는 캐시 RGB FP16→FP32를 사용한다.
- 기준 bin head를 캐시 특징에 적용한 초기 raw AbsRel 차이는 원본 대비
  약 `1.77e-8`. 캐시 오차가 없다고 가정하지 않고 초기 결과를 별도 기록했다.
- 각 학습 실행 1,200 step, batch 8, AdamW lr 0.0001, seed 20260909.
  날짜가 아닌 고정 seed 정수다. 모든 실행은 같은 독립 generator로 minibatch 순서를 맞췄다.
- 실제 학습은 bin/mixture/detail의 **3회, 총 3,600 step**이다.
  mixture mean/mode는 하나의 체크포인트를 읽는 두 방식이지 두 번의 학습이 아니다.
- 기존 raw/clip-median AbsRel, 경계 F1, overshoot, flat-TV 등을 그대로 사용한다.
  집계는 clip→scene→source→source 균등 평균이다. Gate에 경계 영역 AbsRel
  비악화를 추가하여 경계 F1만 증가하는 후보가 통과하지 않도록 했다.

## 2. 1단계 — 출력부 대조

### 구현 범위

1. **Bin control:** 기존 64-bin head와 bin 간격만 추가 학습. 앞단 특징 고정.
2. **Anchored mixture:** 같은 특징에서 두 log-depth 후보, 혼합 확률, 두 척도를 예측.
   원본 bin 깊이를 동결된 중심 기준으로 유지하고 학습 가능한 두 후보를 생성한다.
   중심 이동은 ±0.5 log-depth, 후보 간 반간격은 최대 1.0 log-depth다.
3. **Mean vs mode:** 동일 mixture 가중치에서 log-depth 확률 평균과 두 후보 위치의
   전체 혼합밀도를 비교한 선택 출력을 각각 평가한다. 혼합 가중치만 비교하지 않는다.

이는 **동결 원본을 기준으로 한 출력부 전환의 첫 실험**이다. 기존 bin head를 완전히
제거한 새 모델이나 전체 encoder-decoder의 end-to-end 재학습은 아니다.
Mean 초기화는 원본을 보존하지만 두 후보는 처음부터 약 ±0.02 log-depth로 떨어져
있으므로 mode 초기 출력은 정확한 identity가 아니다.

공통 감독은 유효 내부 픽셀의 log-depth L1 + 0.5 signed-gradient L1이다.
Mixture는 mode 출력에 공통 손실을 적용하고 0.05 Laplace mixture NLL을 추가한다.
따라서 bin과 mixture 비교에는 손실 차이도 있다. 반면 **mean/mode 비교는 같은 가중치**라
출력 선택 방식의 영향을 직접 확인한다. 연속 확률밀도의 NLL은 음수일 수 있으며
이것 자체가 손실 오류는 아니다.

### 선택 붕괴 진단

개발 픽셀에서 두 번째 후보 선택 비율은 약 **0.00353%**로 거의 모두 첫 후보를 선택했다.
평균 후보 간 log-depth 간격은 0.02042, mean/mode의 평균 상대 차이는 약 0.548%였다.
이 진단은 픽셀/클립 평균으로, 아래 source 균등 지표와 집계 방식이 다르다.

즉, 이 학습 조건에서는 두 후보가 전경·배경을 잘 분리하는 기능을 확보하지 못했다.
Mixture 구조 전체가 무효라는 증거가 아니며, 초기화·후보 분리 감독·학습 규모의
영향을 분리하지 않은 상태다. 일반적인 두 후보 출력의 근거는
[EfficientDepth §3.1](https://arxiv.org/html/2509.22527v1#S3.SS1)이지만,
이번 anchored log-depth 구현은 그 논문의 재현이 아니다.

## 3. 2단계 — 독립적인 세부 특징 경로 대조

1단계 출력부가 실패했으므로 이를 결합하지 않았다. 고해상도 특징의 효과는 별도 가설이라
계획을 조정해 **기존 bin 출력부를 고정한 독립 실험**으로 진행했다.

- RGB→1/2 해상도 16채널, 1/4 해상도 16채널 특징을 만들고 융합한다.
- 정규화한 원본 decoder 특징과 세부 특징을 결합하여 원본 특징에 보정량을 더한다.
- 마지막 특징 보정 convolution을 0 초기화하여 시작 시 원본 특징을 보존한다.
- 최종 깊이의 제한된 잔차를 예측하는 이전 보정기와 달리, **깊이 예측 전 특징**을 변경한다.
  원본 깊이의 경계가 없다는 이유로 게이트를 닫는 제약은 없다.
- 원본 SSM·decoder·bin head는 모두 동결. 원본 head의 모든 state tensor가
  학습 전후 정확히 동일함을 검사했다.

이 실험은 RGB 세부 경로 추가의 초기 대조다. 전체 계층형 encoder 교체,
원본 decoder와의 공동 학습, 최종 1× 해상도 학습형 upsampling은 아직 하지 않았다.

첫 실행은 모듈 이름 `half`가 PyTorch의 `Module.half()`와 충돌하여 **학습 전에 실패**했다.
이름을 수정하고 테스트를 통과한 뒤 `_v2` 경로에서 1,200 step을 완료했다.
첫 경로는 실패 기록이며 모델 성능 실험으로 세지 않는다.

## 4. 동일 개발 조건 결과

| 후보 | Raw AbsRel ↓ | 경계 AbsRel ↓ | 경계 F1 ↑ | Overshoot ↓ | Flat-TV ↓ |
|---|---:|---:|---:|---:|---:|
| 기존 v11 | 0.110890 | 0.152884 | 0.342448 | 0.196982 | 0.026897 |
| Bin head 추가 학습 | 0.114481 | 0.159159 | 0.344253 | 0.207857 | 0.026948 |
| Mixture / mean | 0.114975 | 0.159759 | 0.343974 | 0.202163 | 0.028070 |
| Mixture / mode | 0.115903 | 0.161432 | 0.343720 | 0.204234 | 0.028110 |
| RGB 세부 특징 + 고정 bin head | 0.114174 | 0.158306 | 0.342096 | 0.201788 | 0.027444 |

모든 후보에서 raw/경계 AbsRel과 overshoot가 악화했다. 단순한 gradient ratio 증가를
성공으로 해석하지 않았으며, 이번에는 F1의 사전 최소 개선폭(+0.005)도 달성하지 못했다.
기존 v11을 대체할 근거가 없다.

## 5. 완료 / 미완료 및 다음 판단

| 순서 | 상태 | 판단 |
|---|---|---|
| 1. 동일 특징에서 출력부 비교 | 완료 | 이번 anchored mixture와 head-only 학습은 미채택 |
| 2. 원본 출력부 고정, 세부 특징 추가 | 완료 | 이번 추가 경로 구성은 미채택 |
| 3. 두 변경 결합 | 보류 | 각각 검증되지 않은 변경을 결합해 개선을 주장하지 않음 |
| 4. L32/L256·다중 seed·캐시/가림 검증 | 미실행 | 단기 공간 품질 gate를 통과한 후보가 먼저 필요 |
| 5. Pi 4 / Nano B01 5W·10W | 미실행 | 추가 연산의 속도·전력 실측 없이 경량성을 주장하지 않음 |

다음에는 동결된 저해상도 특징을 계속 보정하는 방식과 구분하여,
**사전학습 공간 특징 또는 고해상도 encoder-decoder의 공동 학습**을 대조해야 한다.
이는 아직 구현·평가하지 않았다. 후보 선택 붕괴는 학습 데이터만을 사용한
단순 두 평면/가림 경계 진단으로 먼저 확인하고, 새 구조 비교의 학습량과 감독을
사전에 맞추는 것이 우선이다. 이번 768프레임·단일 seed 결과는 구조의 성능 상한,
충분한 장기 학습의 결과, 또는 일반적인 SSM의 한계를 뜻하지 않는다.

## 6. 산출물과 검증

- 코드: `sokkanaem/mixture_readout.py`, `sokkanaem/detail_feature_path.py`
- 실행: `scripts/architecture_readout_study.py`, `scripts/architecture_detail_study.py`
- 완료 출력: `work_dirs/architecture_readout_20260908/`,
  `work_dirs/architecture_detail_20260908_v2/`
- 각 출력의 `protocol.json`, `selection.json`(1단계), `results.json` 및 가중치.
  코드·기준 가중치·manifest·선택 목록 해시를 기록했다. 2단계는 부모 캐시 해시도 기록한다.
- 관련 테스트 **43개 통과**. 전체 저장소/장치/시간적 안정성 테스트가 아니다.
- 원본 체크포인트 불변 확인. 기존 paper final을 다시 평가하거나 학습에 쓰지 않았다.

```bash
python -m pytest -q tests/test_mixture_readout.py
python scripts/architecture_readout_study.py --out work_dirs/architecture_readout_repro
python scripts/architecture_detail_study.py --parent work_dirs/architecture_readout_repro --out work_dirs/architecture_detail_repro
```

두 실행기는 기존 출력 폴더를 덮어쓰지 않는다. 실측 가능한 개선 후보가 없으므로
원고의 모델·성능 표는 변경하지 않았다.
