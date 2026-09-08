# 작업 9~11 실행 계약

2026-09-06. 결과를 계산하기 전에 고정하는 개발 검증 실행 범위다.
[기존 평가 계약](PROTOCOL.md)과 [번호별 작업](WORK_PLAN.md)을 변경하지 않는다.
새 학습·최종 test 추론·외부 제출은 하지 않는다.

## 9. 품질과 비용의 동일 운용점

- 고정 native v11-s0: K=5/10/30/60, dense carry/reset, 단순 output hold.
  Dense는 불필요한 검출기·출력 캐시를 사용하지 않으며 all-active 출력 동등성을 검사한다.
  Output hold는 기본 K30 검출기·dense fallback 비용을 실제 지불하고 stateless dense 출력을 재사용한다.
- 비교군: DA2 Small(기존 pinned HF snapshot), MiDaS v2.1 Small(공식 release 가중치).
  MiDaS와 EfficientNet-Lite 구현은 별도 로컬 checkout과 SHA-256으로 고정한다.
  두 비교군은 상대 disparity, single-frame이며 미래 프레임을 쓰지 않는다.
  FastDepth 공식 가중치 서버는 HTTP/HTTPS 모두 연결 timeout이어서 이번 실측에서 제외한다.
  따라서 이 실행을 모든 초소형/CNN 경쟁군을 포괄하는 비교라고 부르지 않는다.
- DA2 입력 목표: 196/256/350/518; 실제 patch-grid 크기를 기록한다.
  MiDaS 입력 목표: 192/256/384. 256 목표는 frozen common RGB를 사용한다.
  다른 목표는 5~8에서 정의한 원본 해상도 scoring ROI를 사용한다. 상위 해상도 입력의 추가 정보와 비용을 함께 기록한다.
- 모든 조건의 정확도: 기존 L256 13개, 같은 3,328프레임, clip none/median/scale–shift.
  시간과 품질을 결합하는 주 표는 각 개발 시퀀스의 첫 완전한 L256, 총 4개/1,024프레임으로 제한한다.
  전체 13개 점수와 4개 시간 측정 subset 점수를 혼용하지 않는다.
- AbsRel–latency Pareto 표에 δ1/TCE/실패도 함께 둔다. Native K30을 기준으로
  AbsRel ≤1.05×, δ1 ≥기준−0.01, 실패 수 ≤기준인 비교군 중 가장 빠른 실측점을 찾는다.
  별도로 latency ≤1.10×기준인 비교군 중 최소 AbsRel 점을 찾는다.
  이는 사전 지정 허용오차 내 운용점 탐색이지 통계적 동등성 검정이 아니다. 보간·외삽은 하지 않는다.

## 10. 측정 경계와 비용 분해

- RTX 4090, batch=1, FP32 eager, TF32 off, PyTorch/OpenCV CPU thread=2/1.
  모든 모델에 동일 정밀도/실행 방식을 적용한다. compile/FP16/TensorRT 최고 성능이나 에지 성능으로 일반화하지 않는다.
- 주 시간은 warm-filesystem-cache PNG 경로 읽기/디코딩 → crop/resize/정규화 → H2D →
  검출기/GMC/embedding/backbone/decoder → 256 출력 보간 → CPU 출력 복사까지다.
  모델 로드, GT·RAFT·정합·채점, 출력 파일 저장, 카메라/영상 코덱/네트워크 대기는 제외한다.
  실제 저장장치 cold I/O나 완전한 제품 파이프라인이라고 부르지 않는다.
- 클립마다 state를 초기화한다. 전체 256프레임 warmup 1회 후 5회 반복한다.
  반복마다 상태를 다시 초기화하고 첫 프레임/키프레임/fallback을 포함한다.
  프레임별 wall-clock P50/P95, 평균, 반복 간 범위, 시퀀스별 값, peak allocated GPU memory,
  stream별 중복 storage 제거 후 persistent-state bytes를 기록한다. 최솟값만 선택하지 않는다.
- 별도의 동기화 계측 pass에서 전처리/H2D/검출기/GMC/embedding/backbone/decoder/D2H 비용을 분해한다.
  이 계측은 동기화 오버헤드가 추가되므로 주 latency를 대체하지 않는다.
- GMC K30은 feature tau=0.1/0.05의 별도 진단 행이며, 기본 pixel tau=0.05/0.025와 섞지 않는다.
  GMC 호출·fallback과 정확도를 함께 측정한다. 상태/출력 캐시가 homography로 정렬된다는 보장은 하지 않는다.
- 품질 pass와 시간 pass의 예측 checksum/오차를 대조하고 모델별 GPU 메모리를 분리한다.
  측정 중 다른 GPU compute process가 있으면 중단한다. GPU 상태를 앞뒤로 기록한다.

## 11. 통계 단위와 seed 범위

- 시퀀스 내 유효 픽셀을 먼저 합산하고 시퀀스별 점수를 만든다.
  주 paired CI는 네 시퀀스 점수의 동일 가중 평균 차이에 대해 cluster bootstrap을 적용한다.
  네 시퀀스에서 복원추출하는 4^4=256개 ordered draw를 모두 열거하고 percentile 95% 구간을 낸다.
  같은 draw를 두 모델에 적용한다. 정확한 bootstrap 분포 열거이지 finite-sample coverage 보장은 아니다.
- 두-sided paired sign-flip 2^4=16개도 모두 열거한다. 가능한 최소 p=0.125로
  네 시퀀스만으로 5% 유의성을 주장할 수 없다. 인접 클립이나 반복 실행을 독립 시퀀스처럼 늘리지 않는다.
  기존 TUM/Bonn 1:1 pixel-pooled 평균과 새 equal-sequence estimand는 별도 열로 표시한다.
- 5~8의 L8 모델 비교와 L256 기여 분리, 새 경량 baseline/시간 결과에 paired 분석을 수행한다.
  다수 비교는 탐색적이며 미보정 CI를 확정적 검정으로 제시하지 않는다.
- 기존 v11 seed 0/1/2를 현재 동일 평가 코드로 L8/L256에서 재평가/검증한다.
  세 체크포인트는 동일 v10 부모에서 마지막 8k 단계만 seed가 다른 반복이다.
  checkpoint/meta/config/부모 해시와 역사적 commit 차이를 기록하고 평균·표본 SD·범위를 보고한다.
  처음부터 독립 학습한 3개 모델, 신규 seed 실험, 최종 test 불확실성이라고 부르지 않는다.

출처: [MiDaS 공식 소형 모델 안내](https://pytorch.org/hub/intelisl_midas_v2/),
[MiDaS v2.1 코드](https://github.com/isl-org/MiDaS/tree/94afff70fe0873c3af70355e5388e4d482bfc807),
[FastDepth 공식 배포 경로](https://github.com/dwofk/fast-depth).
