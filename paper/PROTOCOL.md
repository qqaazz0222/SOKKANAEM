# SOKKANAEM paper evaluation contract v1

2026-09-06. 실행 계약은 [evaluation_protocol.json](evaluation_protocol.json), 작업 결과는
[PREPARATION.md](PREPARATION.md) 참조. 이 계약은 향후 논문용 평가에 적용한다.
기존 결과를 새 계약으로 재평가한 것으로 간주하지 않는다.

## 1. 모델과 자료의 고정

기준 모델은 `work_dirs/v11-longclip-spread-s0/latest.pt`의 EMA 우선 로딩 결과다.
실제 파라미터 수는 4,185,872개이며, 입력과 채점은 256×256이다. 체크포인트·설정·실행 코드의
SHA-256, 생성 시점 git commit, 기본값까지 해석한 설정과 학습 계보를
[freeze/baseline.json](freeze/baseline.json)에 기록한다. Git commit만으로 수정 중인 코드가
고정되었다고 간주하지 않는다. 기준 가중치는 이번 작업에서 학습하거나 변경하지 않았다.

기록된 계보는 v8 60k → v9 60k → v10 25k → v11 8k다. 합계 153k는 각 체크포인트의
단계별 step 값 합계이며, 독립 검증한 optimizer 업데이트 총량은 아니다. v8 이전 부모는 기록되지
않았다. 후속 단계의 teacher loss가 0이어도 전체 계보를 teacher-free라고 부르지 않는다.

기존 `acc_real_L*.json`의 holdout은 반복적으로 모델 선택에 사용된 development validation이다.
기존 실촬 루트에서 24개 학습 풀 시퀀스와 4개 개발 검증 시퀀스를 구분했다. 이 수는 전체 합성
학습 자료 수가 아니다. 상세 경로와 구분 근거는 [freeze/data_split.json](freeze/data_split.json)에 있다.

새 final-test 후보는 TUM Freiburg3의 sitting_xyz, sitting_rpy, walking_xyz, walking_rpy 네 시퀀스다.
시퀀스 이름은 다운로드·모델 예측 전에 정했고, 현재 확인 가능한 프로젝트 설정에 이 이름의 사용
기록은 없다. 동일 촬영 환경의 다른 시퀀스이므로 scene-disjoint가 아니며, 카메라 움직임이 있어
fixed-camera primary test도 아니다. 외부 사전학습 모델의 데이터 중복 여부는 미확인이다.
고정 카메라·새 장면 일반화 주장은 별도의 미사용 자료가 필요하다.

원본 출처와 5000 단위/m 깊이 인코딩은 [TUM 다운로드 안내](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download)와
[파일 형식 안내](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats)를 따른다.
원본 압축 파일 및 각 RGB/depth 파일을 해시하고 모든 짝을 디코딩했다. 이 무결성 검사는 모델 평가가 아니다.

| 시퀀스 | RGB–depth 짝 | L8 | L32 | L256 |
|---|---:|---:|---:|---:|
| sitting_xyz | 1,219 | 152 | 38 | 4 |
| sitting_rpy | 795 | 99 | 24 | 3 |
| walking_xyz | 827 | 103 | 25 | 3 |
| walking_rpy | 867 | 108 | 27 | 3 |
| 합계 | 3,708 | 462 | 114 | 13 |

길이별로 각 시퀀스의 처음부터 겹치지 않게 타일링하고 끝의 불완전한 클립만 버린다.
서로 다른 길이의 평가 목록은 프레임 집합이 완전히 같지 않다. 따라서 L8/L256 점수 차이만으로
재귀 상태의 드리프트를 추정하지 않는다. L256의 동일 프레임에서 sparse/dense/reset과
frame/clip 정합을 비교하는 후속 분해 실험이 필요하다.

최종 자료를 보고 모델·임계값·손실을 선택하지 않는다. `build_mixed`는 train/val 모드 모두에서
예약 시퀀스를 거부하고, 평가기는 봉인 manifest와 `--final-test` 명시를 요구한다. 이는 실수 방지
장치이지 모든 외부 학습 스크립트를 통제하는 보안 경계는 아니다. 최종 실행 전 모델 선택과 비교군의
가중치 revision까지 확정해야 한다. 이번 1~4번 작업에서는 final-test 추론을 하지 않는다.

## 2. 정확도 패널: 서로 다른 질문을 합치지 않기

| 패널 | CLI | 평가 GT를 이용한 보정 | 해석 |
|---|---|---|---|
| Metric | `--align none` | 없음 | 미터 단위 출력의 실제 거리 오차 |
| Scale-aligned | `--align median` | 클립당 깊이 median 비율 1개 | 절대 스케일 오차를 제거한 정확도 |
| Relative shape | `--align scaleshift` | 클립당 시차 scale+shift 최소제곱 | 상대 깊이 형상 비교 |
| Shape diagnostic | `--align scaleshift --per-frame` | 프레임마다 시차 scale+shift | 시간에 따른 스케일 변화까지 제거한 보조 진단 |

`--align scale`은 시차 공간 scale-only 보조 진단으로 유지한다. 기존 작업 큐와의 호환성을 위해
CLI 기본값은 기존 세 GT-fit gauge다. 논문용 실행은 반드시 `--align`을 명시한다.

Metric 패널은 `space=depth`인 미터 단위 출력을 평가한다. 상대 disparity 모델을 단순히
`--space depth`로 재명명해 metric 비교에 넣으면 안 된다. Q 계열의 보정 head 역시 median 보정
점수만으로 실제 거리 보정의 성공을 주장할 수 없다. Q의 파라미터 수·내부 518 입력 비용을
4.19M native 모델의 속도와 합쳐 제시하지 않는다.

Scale–shift fit은 시차 제곱오차를 최소화한다. 자유도 증가가 이 목적함수의 최적값을 악화시키지는
않지만, 역수 변환 후 depth AbsRel이 항상 좋아지는 것은 아니다. 정합된 시차가 0 이하가 될 수도 있다.

## 3. 데이터·실패·통계 처리

- RGB/depth를 가장 가까운 timestamp로 ≤20 ms 안에서 짝짓는다. 기존 어댑터는 depth 재사용을
  허용하므로 전역 one-to-one matching이라고 쓰지 않는다.
- 짧은 변을 목표 크기로 resize하고 center crop한다. RGB는 bilinear, 깊이는 nearest다.
- 센서별 sentinel을 제거한 뒤 유한하고 0보다 큰 GT 픽셀만 채점한다. 새 upper-depth cutoff는 없다.
- Metric 예측은 GT 기반 보정이나 깊이 범위 clipping을 하지 않는다. 0 이하 출력은 실패로 센다.
  이는 채점기의 추가 clipping이 없다는 뜻이다. 기준 모델 자체의 binned head 범위(0.3–150 m)는
  모델 설정에 포함되며, 다른 모델의 내재적 출력 범위와 함께 공개한다.
- 기존 disparity 변환은 `EPS=0.001` floor를 유지한다. 정합 후 비양수 시차 비율과 실패 클립 수를
  별도로 기록하고, 실패 클립의 큰 오차도 전체 평균에서 제거하지 않는다. AbsRel>1은 catastrophic이다.
- NaN/Inf 예측은 명시적 실행 오류다. 손상된 RGB/depth는 같은 클립 재시도 후 실패하며 다른 클립으로
  바꾸지 않는다. 유효 GT가 전혀 없는 클립은 한 번만 집계하고 모델 실행 전에 제외한다.
- 실제 채점된 클립 ID를 기록한다. manifest의 오래된 `valid_px` 메타데이터로 채점 집합을 추정하지 않는다.
- 소스별 pixel-pooled AbsRel/RMSE/δ1과 equal-source 평균, clip mean/median/trim10/P90/P95,
  실패 수를 함께 보고한다. balanced P95는 소스별 P95 평균이지 전체 클립의 95분위가 아니다.
- 기존 bootstrap CI는 소스별 clip bootstrap이다. 인접 클립의 상관을 무시하므로 독립 시퀀스 일반화의
  확정적 CI로 해석하지 않는다. 최종 test는 4개 시퀀스뿐이다. 시퀀스별 결과와 sequence-cluster CI는
  후속 통계 분석에서 별도로 보고할 필요가 있다.

## 4. 시간·선명도·효율 계약

클립 시작에 detector/state/cache를 초기화하고 클립 안에서는 유지한다. 기준은 K=30,
tau_on=0.05, tau_off=0.025, GMC off다. dense/reset 등의 대조군 변경은 별도 행으로 표시한다.

시간 지표는 no-fit 또는 clip-fit에서 측정한다. Per-frame fitting은 실제 scale inconsistency도
제거하므로 primary temporal 평가와 함께 요청하면 오류로 차단한다. t-delta는 상수 예측으로 낮출
수 있어 정확도와 OPW/TCE·상수 대조군을 함께 봐야 한다. OPW/TCE를 계산하지 않은 실행은
`temporal_computed=false`이며 호환용 0 값을 실측 결과로 인용하지 않는다.

선명도 평가는 명시한 scaleshift gauge에서 수행한다. 특히 edge AbsRel은 정합 의존적이다.
비양수 시차 clipping이 있으면 정규화된 형상 지표도 무조건 정합 불변이라고 주장하지 않는다.

활성률은 첫 프레임을 포함한 값과 제외한 값을 따로 보고한다. 두 값 모두 유효 GT가 있는 실행 클립의
평균이다. MAC 감소와 latency 감소는 다른 결과다. 향후 시간 측정은 같은 checkpoint, 하드웨어,
precision, batch=1, warm-up, compile 조건, 입력 크기에서 sparse/dense/baseline을 비교해야 한다.
전처리 포함 여부와 peak memory, 측정 반복 수도 별도로 기록해야 한다.

Common-input 목표 256 비교와 각 모델 official-input 비교를 분리한다. HF processor가 패치 크기에
맞춰 252/266 등으로 조정하면 실제 입력 크기를 기록한다. 모든 결과의 채점 격자는 256이다.
평가 총 소요 시간에는 데이터 로딩·지표 계산 등이 섞이므로 inference latency로 인용하지 않는다.

## 5. 재현·검증 명령

프로젝트 루트에서 실행한다. 아래 명령은 모델 학습·최종 테스트 예측을 하지 않는다.

```bash
python scripts/prepare_paper_test.py          # 다운로드된 원본과 봉인 목록의 동일성 재검사
python scripts/paper_freeze.py --inspect     # CPU로 기준 가중치와 계보 확인
python scripts/paper_freeze.py --verify      # 코드·가중치·자료 SHA-256 재검사
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 python -m pytest tests/ -q
```

기존 개발 자료에서 계약을 사용하는 예시(이번 준비 작업에서는 실행하지 않음):

```bash
python scripts/eval_acc.py --manifest manifests/acc_real_L8.json \
  --model ours:work_dirs/v11-longclip-spread-s0/latest.pt \
  --align none --align median --align scaleshift \
  --out-dir work_dirs/paper_v1_dev --tag v11-L8
```

`paper_freeze.py --write`는 최초 기록 생성에만 쓴다. 이미 생성된 기록을 덮어쓰지 않으며,
변경된 모델·계약은 새 버전으로 관리한다. [evidence_ledger.json](freeze/evidence_ledger.json)의
기존 결과 해시는 현재 파일 보존의 증거이지, 과거 실행 시점의 가중치·코드 계보를 소급 증명하지 않는다.
