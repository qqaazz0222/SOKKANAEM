# 논문화 준비 1~4번 작업 결과

2026-09-06. 대상: SOKKANAEM의 Mathematics 투고 준비. 승인 범위는 기준 고정, 데이터 분리,
평가 계약 정립, 수학·구현 일치성 수정이다. 신규 학습, 최종 테스트 추론, 논문 제출은 수행하지 않았다.

## 작업 결과표

| 번호 | 실제 작업 | 산출물 | 판단과 한계 |
|---|---|---|---|
| 1 | v11 가중치·설정·실행 코드 해시와 실제 학습 계보 고정, 기존 결과 목록화 | [baseline](freeze/baseline.json), [evidence ledger](freeze/evidence_ledger.json) | 4,185,872 parameters. v8→v9→v10→v11 계보. 기존 측정 파일의 과거 실행 provenance는 미완전 |
| 2 | 반복 사용 holdout을 개발 검증으로 재분류, 미사용 TUM 4개 시퀀스 확보·검증·목록 봉인 | [data split](freeze/data_split.json), [final test](freeze/final_test.json), [원본 해시](freeze/final_test_files.json) | 3,708 RGB/depth 짝; L8 462, L32 114, L256 13 clips. Same-scene moving-camera transfer에 한정 |
| 3 | no-GT-fit metric / median / scale–shift 패널 분리, 실패·크기·활성률·provenance 기록, 자료 보호 테스트 | [평가 계약](PROTOCOL.md), [실행 계약 JSON](evaluation_protocol.json), [평가기](../scripts/eval_acc.py) | GT 보정 후 정확도를 실제 거리 정확도로 오인하지 않도록 구분. 기존 수치 재평가·성능 향상은 미수행 |
| 4 | 정확 ZOH와 구현의 1차 입력 항 구분, Δ=0 특이식 제거, binary copy 선행연구 정정, 출력 캐시와 상태 보존 분리 | [영문](draft.md), [국문](draft_ko.md), [기여 범위](../NOVELTY.md) | 상태 항등성은 유지. 최초성·dense 출력 동등성·예측 드리프트 상한은 주장하지 않음 |

## 성능·정확도에 대한 현재 판단

장점은 작은 native 모델, 명시적 상태 보존, 연산 절감량과 정확도 손실을 분해할 수 있는 구조다.
그러나 낮은 활성률만으로 실제 속도 우위를 증명할 수 없고, 기존 결과에서는 장클립 정확도와
동적 장면의 형상 정확도가 약점으로 남아 있다. 이번 작업은 이를 개선했다고 주장하는 단계가 아니라
동일 기준에서 검증할 수 있도록 근거와 도구를 정리한 단계다.

논문에서 특히 다음을 혼합하면 안 된다.

- Native 4.19M의 효율과 Q 계열 사전학습 backbone의 품질.
- 서로 다른 checkpoint, 입력 크기, GT 보정 조건에서 얻은 정확도와 속도.
- 작은 t-delta와 실제 기하학적 시간 정확도.
- Δ=0에서의 state copy와 캐시를 포함한 전체 출력의 정확성.
- 여러 번 선택에 쓴 holdout 점수와 미사용 final-test 점수.

## 수학적 정정의 근거

상수 입력 구간의 정확 ZOH 입력 행렬은

\[
\bar B(\Delta)=\int_0^\Delta e^{sA}\,ds\,B
=\Delta\varphi_1(\Delta A)B,\quad
\varphi_1(Z)=\sum_{k=0}^{\infty}\frac{Z^k}{(k+1)!}.
\]

따라서 Δ=0에서도 정의되며 \(\bar B(0)=0\)이다. 실제 코드는 지수 transition과
\(\Delta Bx\) 입력 근사를 쓴다. 이 역시 zero-step identity를 가지지만 active input의 정확
ZOH 적분이라고 부를 수는 없다. 이 정정은 모델 계산을 바꾸지 않고 설명을 구현과 맞춘 것이다.

Binary mask에서는 \(F_{m\Delta}=mF_\Delta+(1-m)I\) 형태의 state update/copy와 동등하다.
[Skip RNN 원문](https://arxiv.org/pdf/1708.06834)의 식 (3)도 binary update/copy를 사용한다.
따라서 차별점은 상태 복사의 최초성이 아니라 변화 마스크와 selective SSM, dense depth readout의
결합 및 그 정확도–효율 특성으로 제한했다. Soft mask에서는 같은 동등성이 일반적으로 성립하지 않는다.

보존된 state를 읽는 C, 입력 x, output gate가 바뀌면 출력은 달라질 수 있다. Temporal readout 및
spatial output 재사용은 추가 근사다. GT 평균 풀링–bilinear 복원 역시 학습 가능한 다채널 decoder의
최적 해를 구한 것이 아니므로 엄밀한 accuracy ceiling이 아닌 고정 재표본화 진단으로 수정했다.

Eventful Transformers의 저자도 [공식 ICCV 기록](https://openaccess.thecvf.com/content/ICCV2023/html/Dutson_Eventful_Transformers_Leveraging_Temporal_Redundancy_in_Vision_Transformers_ICCV_2023_paper.html)에
맞춰 Matthew Dutson, Yin Li, Mohit Gupta로 정정했다.

## 검증 기록

- 최종 전체 CPU 회귀 테스트: 127 passed, 6 skipped (CUDA/Triton 전용); 3.97초.
- 합성 임시 자료로 metric/median 점수 차이, 비유한 예측 오류, 손상 클립 대체 금지,
  예약 자료의 train/val 유입 차단, binary update/copy 및 state/readout 구분을 검증했다.
- 새 자료 3,708개 RGB/depth 짝을 다시 디코딩·해시한 뒤 L8/L32/L256 manifest가 봉인본과
  완전히 일치함을 확인했다. 모델 추론은 포함하지 않았다.
- `paper_freeze.py --write`로 기준·분할·기존 증거의 세 기록을 생성했다. 기존 결과 155개를
  목록화했으며, `--verify`로 가중치·코드·자료를 포함한 7,608개 고유 파일/해시 쌍을 검증했다.
- 변경한 Python 스크립트 구문 검사와 이번 수정 범위의 `git diff --check`를 통과했다.
  저장소 전체 검사에서 기존 `work_dirs/acc/table.txt`, `work_dirs/queue21.log`의 trailing whitespace가
  발견됐지만, 사용자 실험 로그를 보존하기 위해 수정하지 않았다.
- Δ-gating SVG의 XML 유효성과 생성 함수 출력의 완전 일치를 확인했다.
- 원본 압축 파일을 포함한 새 자료는 `data/paper_test`에 약 4.6 GiB 보관했다. 자료는 기존 gitignore의
  `data/` 규칙으로 제외되며 재현용 출처·해시·manifest는 별도로 보존한다.

## 다음 실험에서 남은 일

현재 상태를 바로 투고 가능 또는 정확도 문제가 해결된 상태로 해석하면 안 된다. 다음 단계는
개발 검증 자료에서 동일 가중치의 sparse/dense/reset 비교와 no-GT-fit 정확도를 산출하고,
동일 입력·precision의 latency를 재측정하는 것이다. 이후 비교군 revision과 모든 모델 선택을 끝낸
뒤에만 봉인 final test를 연다. 새로운 방·고정 카메라 일반화 주장을 유지하려면 별도의 미사용
촬영 자료도 필요하다. 다중 seed 및 sequence 단위 불확실성 분석, 본문 전체 결과표의 계약 통일도 남아 있다.

기존 영문·국문 원고에는 오래된 실험표와 서로 다른 단계의 수치가 남아 있다. 이번 수정은 방법론과
주장 범위를 바로잡는 1~4번 작업이며, 모든 결과표를 최신 실험으로 재생성한 최종 원고는 아니다.
