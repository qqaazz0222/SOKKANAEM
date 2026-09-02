# SOKKANAEM 정확도 우선 고도화 계획

> 목표: 파라미터 수와 처리 효율의 일부를 정확도에 재투자하여, 최종 깊이 맵이 최소한
> Depth Anything V2 Small(DA2-small) 수준의 형상 정확도와 경계 선명도를 갖도록 한다.
>
> 작성일: 2026-08-29  
> 기준 체크포인트: `work_dirs/v11-longclip-spread-s0/latest.pt`

## 1. 결론

현재 4.19M 모델에 손실 항을 추가하거나 가중치만 조정하는 단계는 종료한다. 다음 주력 모델은
10~15M 파라미터 범위의 `SOKKANAEM-M`으로 확장하고, 단일 프레임 형상 품질을 먼저 확보한 뒤
시간 상태와 희소 연산을 재도입한다.

핵심 변경은 다음 세 가지다.

1. `dim/depth/d_state`를 키워 프레임당 공간 표현 용량을 확장한다.
2. 상대 형상과 metric scale을 분리한 이중 헤드 및 full-resolution learned upsampling을 도입한다.
3. 동적·근거리 clip을 명시적으로 더 자주 학습하고, 단일 프레임 품질 합격 후에만 희소·시간
   학습을 수행한다.

15M 이하 native 모델이 최종 품질 게이트를 넘지 못하면 손실 실험을 반복하지 않는다. 이 경우
DA2-small의 pretrained encoder/head를 직접 초기화하는 품질 보장 트랙으로 전환하고, 이후
SOKKANAEM-M으로 압축한다.

## 2. 현재 상태와 병목

### 2.1 현재 기준선

| 항목 | 현재 SOKKANAEM | 비교 기준 |
|---|---:|---:|
| 파라미터 | 4.19M | DA2-small 24.79M |
| L8 scale-shift balanced AbsRel | 약 0.115 | DPT-Large 0.0876 |
| L8 scale-shift balanced δ1 | 약 0.866 | DPT-Large 0.9274 |
| 선명도 gradient ratio | 0.432 | DA2-small 0.756 |
| 256px dense 연산량 | 1.64 GMAC | - |

DA2-small의 TUM 평균 AbsRel은 89개 clip 중 15개의 scale-shift 정합 0-crossing 때문에 오염되어
있다. 따라서 DA2-small의 단일 평균값을 목표로 삼지 않고, 안정적인 DPT-Large 공통 게이트와
DA2-small의 source별 유효 지표를 함께 사용한다.

### 2.2 확인된 병목

- TUM 정적 픽셀에서는 현재 4.2M 모델과 343M DPT-Large가 사실상 동률이다.
- 오차 격차는 동적 픽셀에서 2.66배, 2m 이내 근거리에서 2.38배로 커진다.
- 깊이 경계띠 격차는 1.15배이므로 patch 크기나 입력 해상도가 첫 번째 병목은 아니다.
- 재귀 상태는 L8/L32/L256 정확도에 측정 가능한 이득을 주지 않는다. TemporalBlock 자체는
  프레임당 용량으로 필요하지만, 프레임 간 기억은 정확도 생성기로 작동하지 않는다.
- 현재 DPT decoder는 1/2 해상도에서 깊이를 만든 뒤 마지막 2배를 bilinear interpolation한다.
  깊이 bin의 기대값 역시 경계에서 평균화를 유발할 수 있다.
- `fuse_norm`, log-space gradient/normal, DA2 teacher-gradient, 근거리 loss 재가중은 단독으로
  DA2 수준의 선명도에 도달하지 못했다.
- A7 근거리 가중은 Bonn 근거리·동적 영역 일부를 개선했지만 TUM 동적 오차는 거의 그대로였다.

따라서 다음 단계는 게이팅 튜닝이 아니라 **공간 표현 용량, 고해상도 복원, 동적 장면 감독**에
집중한다.

## 3. 최종 합격 기준

평가는 봉인된 `manifests/acc_real_L{8,32,256}.json`을 사용하며, holdout이나 clip 순서를
변경하지 않는다.

### 3.1 단일 프레임 및 짧은 clip 정확도

L8, scale-shift 기준:

- balanced AbsRel ≤ **0.090**
- balanced δ1 ≥ **0.930**
- clip P95 AbsRel ≤ **0.153**
- 정합 실패 0, catastrophic failure 0
- TUM clip median AbsRel ≤ **0.095**
- Bonn pooled AbsRel ≤ **0.058**
- Bonn δ1 ≥ **0.973**

앞의 네 항목은 정합 실패가 없는 DPT-Large 기준을, 뒤의 두 Bonn 항목은 DA2-small의 실제 유효
성능을 반영한다. 한 source의 평균으로 다른 source의 실패를 가리는 모델은 승격하지 않는다.

### 3.2 선명도

- balanced boundary gradient ratio ≥ **0.756**
- depth-boundary F1 ≥ DA2-small
- edge-band AbsRel ≤ DA2-small
- flat-region total variation은 현재 보고 체크포인트보다 악화하지 않음
- overshoot/ringing 비율은 DA2-small 이하

Gradient ratio 하나는 고주파 노이즈로 살 수 있으므로 boundary 위치 정확도와 평탄 영역 노이즈를
반드시 함께 판정한다.

### 3.3 긴 스트림

프레임별 scale-shift gauge 기준:

- L8→L256 AbsRel 증가 ≤ **5%**
- L8→L256 δ1 하락 ≤ **1.0pt**
- sparse와 dense의 L8 정확도 차이 ≤ **2%**
- sparse 전환 후 선명도 하락 ≤ **3%**

### 3.4 효율 예산

- 권장 파라미터 상한: **15M**
- native 모델 hard cap: **25M**
- 256px dense 연산량 목표: **4.5 GMAC 이하**
- 실제 4090 및 edge latency는 DA2-small 대비 1.5배보다 느려지지 않아야 함
- 최종 평가는 dense latency와 실촬 active ratio에서의 streaming latency를 모두 보고
- 파라미터나 latency 게이트를 넘더라도 정확도·선명도 게이트를 통과하지 못하면 승격하지 않음

## 3.5 게이트 개정 (2026-08-31)

다섯 라운드·15개 arm을 이 게이트로 판정하면서 계약 자체의 결함이 드러났다. 수치는 REPORT
§4.44~§4.46. 아래는 **측정으로 뒷받침되는 개정안**이며, 봉인된 manifest·holdout·클립 순서는
건드리지 않는다.

1. **P1의 gauge를 모델 종류에 맞춘다.** §3.1은 전 항목을 2-DOF scale+shift로 재는데, 이는
   **metric 모델에게 틀린 자다**: 정합이 metric 보정을 정확히 상쇄하므로, DA2 형상을 쓰는 Q0는
   scaleshift에서 TUM 0.3101·정합실패 10인 반면 median(1-DOF)에서는 **0.1097·실패 0**이다.
   개정: 상대 모델 비교용으로 scaleshift를 계속 보고하되, **승격 판정은 배포 gauge(metric
   모델 = median 1-DOF)에서 하고, 그 gauge의 정합 실패 0을 필수 항목으로 둔다.**
2. **선명도 임계는 채점 해상도를 명시해야 한다.** DA2의 grad_ratio는 256px 채점에서 0.7559,
   384px 채점에서 0.6454다. "0.756"은 숫자가 아니라 (기준 모델, 채점 해상도) 쌍이다. 개정:
   **"DA2-small을 같은 채점 해상도에서 잰 값 이상"** 으로 쓰고, 해상도를 표에 적는다.
3. **grad_ratio 1.0 초과는 통과가 아니다.** Q1 arm이 1.154로 게이트를 "통과"했는데, 같은
   실행에서 overshoot 0.347·edge AbsRel 0.187로 실패했다 — GT보다 가파른 것은 선명함이 아니라
   오버샤프닝이다. 개정: **0.756 ≤ grad_ratio ≤ 1.0**, 그리고 overshoot·flat TV·edge AbsRel을
   동시 판정(이미 그렇게 구현돼 있고, 이 케이스가 그 설계를 검증했다).
4. **flat TV 기준을 "보고 체크포인트 이하"에서 "DA2 이하"로 올린다.** 자기 자신을 기준으로
   삼으면 노이즈가 있는 현 상태가 영구 기준이 된다. 실측: 우리 0.0358, DA2 0.0319,
   DPT-Large 0.0262.
5. **P1의 Bonn 항목은 유지한다.** DA2의 Bonn(0.0578 / δ1 0.9733)은 실재하고, Q0가 0.0592 /
   0.9687로 거의 재현했다 — 도달 가능한 기준임이 확인됐다. 반대로 **balanced AbsRel 0.090은
   현재 어떤 arm도 근처에 못 갔다**(최고 0.1113). 이 값을 유지할지 완화할지는 §7 트랙의 결과를
   보고 결정하되, 유지한다면 native 트랙으로는 도달 불가라는 것이 이번 라운드의 결론이다.

## 4. 목표 아키텍처

### 4.1 Backbone 후보

첫 용량 프로브는 기존 구조를 유지하고 크기만 바꾼다.

| 모델 | dim | depth | d_state | 예상 파라미터 | 256px dense MAC |
|---|---:|---:|---:|---:|---:|
| 현재 | 192 | 4 | 16 | 4.19M | 1.64G |
| M0 | 256 | 6 | 24 | 10.83M | 3.80G |
| M1 | 288 | 6 | 24 | 13.55M | 4.59G |

M0를 기본 후보로 한다. M1은 M0가 표현 상한에 걸렸다는 증거가 있을 때만 전체 학습한다.

### 4.2 상대 형상과 metric scale 분리

현재 하나의 depth 출력이 근거리 경계와 150m 원거리 metric range를 동시에 해결한다. 이를 다음
두 경로로 분리한다.

1. **Shape head**
   - per-frame normalized relative disparity를 예측
   - scale-shift invariant shape, ordering, boundary를 담당
   - full-resolution 또는 최소 1/2-resolution 특징에서 출력

2. **Metric calibration head**
   - global token에서 scale/shift 또는 log-scale/offset을 예측
   - relative shape를 metric depth로 변환
   - 시간 방향으로 완만하게 변화하도록 별도 안정화 가능

예시 형식은 다음과 같다.

```text
disp_metric = softplus(scale) * disp_shape + shift
depth_metric = 1 / clamp(disp_metric, eps)
```

모델이 metric range 불확실성을 공간 경계 평활화로 해결하지 못하도록 shape loss와 metric loss를
각 헤드에 분리해 적용한다.

### 4.3 실제 멀티스케일 디코더

현 구조의 얕은 RGB stem과 마지막 bilinear upsampling을 다음과 같이 교체한다.

- 1/4, 1/8, 1/16 feature를 명시적으로 생성하고 각각 decoder에 전달
- decoder 기본 폭을 64→96으로 확대하되 총 파라미터 15M 예산 안에서 조정
- 단순 `a + b` 대신 normalized gated fusion 사용
- 마지막 2배 bilinear interpolation을 learned convex upsampling 또는 pixel-shuffle residual로 교체
- coarse bin/log-depth와 full-resolution continuous residual을 합성
- full-resolution residual은 zero-init하여 초기 출력이 coarse head보다 나빠지지 않게 함

`fuse_norm` 단독 arm은 기각됐으므로 동일 변경만 반복하지 않는다. 이번 변경의 검증 대상은
정규화 자체가 아니라 **유효한 고해상도 feature와 학습형 복원 경로**다.

### 4.4 시간 모듈의 역할

- 공간 backbone과 decoder가 프레임별 정확도를 만든다.
- 시간 블록은 zero-init residual adapter로 시작한다.
- 시간 상태는 temporal stability 및 sparse update에만 사용한다.
- dense/state-reset 모델을 정확도 oracle로 계속 보존한다.
- 시간 모듈이 dense oracle보다 정확도를 낮추면 해당 stage를 통과시키지 않는다.

## 5. 평가 도구 보강

본 학습 전에 다음 지표를 `eval_acc.py` 또는 별도 sharpness evaluator에 추가한다.

- multi-threshold depth-boundary precision/recall/F1
- boundary localization error
- edge-band AbsRel/RMSE
- gradient magnitude ratio와 gradient 방향 오차
- flat-region total variation
- edge overshoot/ringing 비율
- dynamic/static 및 near/mid/far 교차 영역 지표
- clip별 P50/P90/P95와 정합 실패 목록

필수 테스트:

- 완벽한 GT 입력은 모든 boundary 지표에서 최적이어야 함
- blur를 적용하면 gradient ratio와 boundary F1이 함께 하락해야 함
- 고주파 노이즈를 추가하면 gradient ratio는 오를 수 있지만 flat TV/precision에서 실패해야 함
- scale/shift 변환은 relative-shape 지표를 바꾸지 않아야 함

## 6. 학습 계획

### Stage A — dense single-frame 형상 학습

목적: 시간 상태와 sparse approximation을 배제하고 모델의 순수 공간 표현 상한을 측정한다.

- T=1 또는 매 프레임 state reset
- `tau=0`, spatial/temporal cache off
- 256px, 40k~60k steps
- M0와 M1은 처음부터 전부 돌리지 않고 hard-subset saturation probe로 선별
- pretrained 모듈이 있으면 backbone LR을 새 head LR의 0.1배로 설정

주 손실:

```text
L = L_shape_ssi
  + λ_metric L_metric_silog
  + λ_grad L_multiscale_grad
  + λ_boundary L_boundary_location
  + λ_rank L_pairwise_order
```

초기에는 다섯 항 이상을 동시에 자동 가중하지 않는다. 기존 auto loss weighting의 불안정성이
확인되어 있으므로 고정 가중치로 2~3개 arm만 스크리닝한다.

Stage A 통과 조건:

- 현재 dense 기준 대비 balanced AbsRel 10% 이상 개선
- dynamic 및 near AbsRel 각각 10% 이상 개선
- gradient ratio 0.60 이상
- P95가 평균보다 더 크게 개선

### Stage B — 디코더 및 이중 헤드

Stage A 승자 하나에서 다음 순서로 단일 변수 비교를 한다.

| Arm | 변경 |
|---|---|
| D0 | 기존 DPT decoder |
| D1 | full-resolution learned upsampling |
| D2 | D1 + shape/metric 이중 헤드 |
| D3 | D2 + coarse-bin/continuous-residual 출력 |

판정 우선순위는 boundary F1 → dynamic/near AbsRel → balanced AbsRel → MAC 순이다.

- D1이 gradient ratio를 0.05 이상 올리지 못하면 learned upsampling 설계를 재검토한다.
- D2가 metric 정확도를 유지하면서 shape 지표를 개선하지 못하면 이중 헤드를 기각한다.
- D3의 bin head가 entropy와 경계 지표를 개선하지 못하면 scalar shape head로 단순화한다.

### Stage C — 동적·근거리 데이터 재구성

현재 데이터셋별 균등 sampler에 clip 난이도 strata를 추가한다.

사전 계산할 clip 속성:

- 2m 이내 유효 픽셀 비율
- optical-flow residual 기반 동적 픽셀 비율
- GT frame-to-frame depth 변화량
- 유효 depth 밀도
- 현재 보고 체크포인트의 clip AbsRel/P95

배치 구성 목표:

- 40~50%: 동적 또는 near-heavy clip
- 20~30%: 일반 실촬 clip
- 나머지: 원거리·합성 다양성 유지

동적 mask는 `warp_residual_loss`가 이미 계산하는 flow를 재사용하여 추가 RAFT 호출을 피한다.
근거리 band는 동적 mask의 대체물이 아니라 보조 가중치로만 사용한다.

### Stage D — 고해상도 미세조정

Stage C까지 통과한 한 모델만 진행한다.

- 256→384px, 10k~20k steps
- 낮은 LR과 frozen/unfrozen 비교 2-arm 이내
- 384px 개선이 boundary F1과 small-object 정확도에 나타나는지 확인
- balanced AbsRel 개선이 2% 미만이고 선명도 개선도 5% 미만이면 384 배포는 기각
- 512px는 384px에서 정확도는 통과하고 선명도만 미달할 때만 사용

### Stage E — 비디오와 희소 연산 재도입

아래 순서를 건너뛰지 않는다.

1. T=8 dense, state reset
2. T=8 dense, temporal residual adapter
3. T=24 dense long-clip
4. random mask skip 0→50% curriculum
5. 실제 detector mask fine-tuning
6. L8/L32/L256 sparse 평가

각 단계는 바로 앞 dense oracle 대비 accuracy와 sharpness 열화를 측정한다. 희소성 때문에 품질이
합격선 아래로 내려가면 tau를 먼저 낮추고, 그다음 cache 정책을 조정한다. backbone 품질을 희생해
active ratio를 맞추지 않는다.

## 7. DA2 품질 보장 트랙

native `SOKKANAEM-M`이 Stage D까지 수행하고도 최종 accuracy 또는 sharpness gate를 넘지 못하면
다음 트랙으로 전환한다.

### Q0 — DA2 직접 초기화

- DA2-small의 DINOv2 encoder와 depth head를 직접 초기화
- 초기 shape 출력이 DA2-small과 동일하도록 유지
- metric calibration head와 zero-init temporal residual만 추가
- 처음에는 모든 프레임에서 dense DA2 경로를 실행하여 품질 하한을 봉인

### Q1 — metric/video 적응

- relative shape head에는 낮은 LR 또는 초기 freeze 적용
- metric GT는 calibration head와 residual을 주로 학습
- DA2 원본 출력에 대한 retention loss로 catastrophic forgetting 방지
- checkpoint 승격 시 DA2 sharpness와 source별 정확도보다 낮아지지 않았는지 확인

### Q2 — native student 압축

- teacher의 여러 encoder stage를 student의 여러 block에 정렬
- 단일 final-token cosine loss 대신 feature affinity, pairwise ordering, boundary logit을 함께 증류
- teacher value는 metric GT와 충돌하지 않도록 confidence가 높은 relative shape에만 사용
- student가 최종 게이트를 통과하면 Q 모델은 학습용 teacher로만 남김

예상 파라미터는 25~30M으로 native 목표보다 크지만, 요구한 DA2 수준 품질을 가장 확실하게
보장하는 안전장치다.

## 8. 실험 순서와 중단 기준

| 순서 | 실험 | 예상 산출물 | Go 기준 |
|---:|---|---|---|
| 0 | boundary evaluator 추가 | DA2/current 기준표 | 지표 단위 테스트 통과 |
| 1 | M0 hard-subset saturation | 용량 상한 | dynamic/near 10% 개선 |
| 2 | M1 제한 프로브 | M0 대비 용량 효과 | M0 대비 2% 이상 개선 |
| 3 | D1 learned upsampling | 선명도 변화 | grad ratio +0.05 이상 |
| 4 | D2 이중 헤드 | shape/metric 분리 효과 | shape 개선, metric 무열화 |
| 5 | D3 residual bin head | bin 평균화 해소 | boundary F1 추가 개선 |
| 6 | 동적 strata full 256px | 본 학습 후보 | accuracy gate 근접/통과 |
| 7 | 384px fine-tune | 최종 image model | accuracy+sharpness 통과 |
| 8 | 3 seeds | 재현성 | CI가 합격선 안쪽 |
| 9 | sparse/video 재도입 | 배포 후보 | long-stream gate 통과 |
| 10 | 필요 시 Q0~Q2 | 품질 보장 모델 | DA2 이상 품질 봉인 |

> **2026-08-29 측정 반영 — 순서 변경.** 3번(D1)을 1·2번(M0/M1)보다 먼저 수행한다. GT를
> patch-16 토큰 그리드로 통과시킨 oracle의 boundary gradient ratio가 **0.2612**로 현재 모델의
> 0.4323보다 낮다(REPORT §4.44). 즉 §3.2의 0.756은 토큰 경로 용량으로는 도달 불가이고, 선명도의
> 레버는 `dim/depth/d_state`가 아니라 전체 해상도 복원 경로다. M0 프로브는 폐기하지 않되 그
> 목적을 **dynamic/near 정확도**로 한정한다. 같은 측정에서 우리 출력이 blur뿐 아니라
> ringing도 한다는 것이 드러났으므로(overshoot 0.3196, DA2 0.2255, DPT-Large 0.2237),
> 선명도 arm은 grad_ratio 상승분을 overshoot·flat TV와 함께 판정한다.

다음 경우 해당 방향을 중단한다.

- 두 가지 가중치에서 개선이 없으면 같은 loss의 추가 sweep을 하지 않는다.
- 8k probe의 미세한 차이만으로 60k 구조를 승격하지 않는다.
- 평균은 개선하지만 P95, dynamic, near 중 두 항목 이상이 악화하면 기각한다.
- gradient ratio만 개선하고 boundary F1 또는 flat TV가 악화하면 노이즈/oversharpening으로 기각한다.
- sparse 성능이 나쁘면 image backbone을 재학습하지 않고 sparse adaptation stage에서 해결한다.

## 9. 예상 일정과 비용

현재 4090에서 4.2M 모델의 60k 학습은 약 14.5시간이다. 모델과 해상도 증가를 고려한 대략적인
예산은 다음과 같다.

| 작업 | 예상 시간 |
|---|---:|
| 평가기 보강 및 기준선 재생성 | 1~2일 |
| M0/M1 제한 프로브 | 2~3일 |
| decoder/head arm | 3~5일 |
| M0 256px 전체 학습 | run당 약 1.5~2일 |
| 384px fine-tune | run당 약 1~2일 |
| 최종 3-seed | 약 5~7 GPU-day |
| sparse/video 적응 및 평가 | 3~5일 |

총 예상 기간은 native 트랙 기준 약 3~5주다. Q 트랙 전환 시 pretrained 배관과 추가 압축 실험으로
1~2주가 더 필요할 수 있다.

## 10. 첫 구현 단위

다음 구현 라운드는 아래 범위로 제한한다.

1. boundary F1, flat TV, overshoot 지표와 테스트 추가
2. model config에서 `dim`, `depth`, `d_state`, decoder width를 명시적으로 기록·복원
3. M0/M1 파라미터·MAC·latency 측정 스크립트 확장
4. dense/state-reset hard-subset saturation probe 작성
5. 결과를 보고 D1 구현 여부 결정

첫 라운드에서는 full-resolution decoder, 이중 헤드, 새로운 sampler를 동시에 구현하지 않는다.
먼저 **현재 decoder에서 용량 증가가 실제 dynamic/near 병목을 움직이는지** 판정한 뒤 다음 변경으로
넘어간다.
