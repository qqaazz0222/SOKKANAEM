# K4 흐름 정렬·실제 SSM 상태 전달 학습과 기억 제거 대조

후속: [K2 출력 보존·불확실 영역 한정 보정](Q0_REGION_REPAIR_STUDY_20260909.md).
영역 밖 보존은 확인했으나 새 보정량은 거의0으로 품질 향상 미입증. 아래 K4 결과는 유지한다.

2026-09-09. [K2 정렬 특징 학습](Q0_ALIGNED_TRAINING_STUDY_20260909.md)의 후속.

**전체 추론1회 뒤3회 업데이트하는 K4 경로에서 SSM 상태 전달·학습·출력 효과를 확인했다.
그러나 상태를 이어도 품질은 개선되지 않았고, 새 K4 후보는 모두 기존 동시 기준에 실패했다.**
기억이 작동한다는 사실과 기억이 유용하다는 주장을 분리한다. 기존 출력-flow K2 및 논문 모델은 유지한다.

## 1. 실제 시간 상태를 사용하는 실험

K4의 주기는 `Q0 전체 갱신 → update1 → update2 → update3 → Q0 전체 갱신`이다.
전체 갱신 때 Q0 특징을 정확히 계산하고 h를0으로 초기화한다. Update1이 만든 h는
흐름 정렬 후 update2로, 다시 update3으로 전달된다. Update1–3의 특징과 h를 detach하지 않고
세 업데이트 전체로 역전파했다. 이 범위는 **짧은 실제 재귀**이지 장기 기억의 검증은 아니다.

| 학습 구성 | 별도 SSM h | 특징 cache | 초기 조건 / 파라미터 |
|---|---|---|---|
| Carry SSM | 업데이트 사이 유지 | 계속 유지 | Reset SSM과 전체 가중치 동일 /89,040 |
| Reset SSM | 매 업데이트 직전0 | 계속 유지 | Carry SSM과 전체 가중치 동일 /89,040 |
| MLP | h를 사용하지 않음 | 계속 유지 | 공통 경로 동일 초기값 /89,037 |

- 원래 training192클립·768프레임(각4프레임), source별64클립을 사용했다.
  학습 가능한 생략 위치는 총576개이며, 각 학습 step에서 선택한 한 클립의3개 생략 위치를 순서대로 처리한다.
- 세 모델 각각600 optimizer step, 총1,800step /5,400 update 위치 노출이다.
  Seed739, batch1, AdamW lr.0003/wd.0001, 동일600개 클립 index 순서를 저장했다.
- 이전 모델을 이어 학습하지 않고 동일 공통 초기화로 새 학습했다. Residual readout은0이다.
- Q0 encoder/decoder/calibration 동결. CLS/pooled는 그대로 유지한다.
- DIS nearest로 특징/h를 함께 이동한다. 기존 RGB anchor 차이 mask와 threshold .001/.0005, dilation을 유지했다.
- 첫 Q0 특징 이후에는 현재 teacher 특징을 입력으로 넣지 않는 free-running 학습이다.
  Teacher는 손실 계산에만 사용한다. Training flow는 미리 계산하며 개발 추론의 flow 비용은 실제 시간에 포함한다.
- Teacher feature는 FP16 cache→FP32 복원, RGB/depth/pooled는 FP32다.
- 동일 손실 `1feature+5Q0logdepth+10Q0loggradient+20trainingGTflat`을3개 업데이트 평균으로 사용했다.
  **이번에는 기억 효과를 분리하려고 손실을 바꾸지 않았다.** 이전에 드러난 손실의 경계/flat trade-off는 남아 있다.
- Training/development 경로 교집합0 및 reserved final sequence 제외 확인.
  개발 L32/L256은 기존 자료를 재사용했다. 독립 장면이나 최종 검증이 아니다.

## 2. 상태가 실제로 작동하는가

- Carry 학습의 기록된100–600step 모두 transition A의 데이터 gradient가0보다 컸다.
  기록 중 최대5.7366×10⁻⁴다. Step1은0 초기 readout 때문에0이었다.
- Reset 학습의 기록된 모든 step은 A 데이터 gradient0이었다.
- 단위 검사에서 같은 입력·가중치의 update1은 carry/reset이 같고, update2부터 달라짐을 확인했다.
- 학습 완료 가중치의 개발 예측에서도 전체 갱신과 update1은 carry/reset이 정확히 같았다.
  두 checkpoint×L32/L256에서 총1,280회 비교했다.
- Carry checkpoint의 update2/3 출력은 상태 제거 시 달라졌다. L256 source/scene-balanced
  평균 절대 log-depth 차이는.001383이며 최대 단일 픽셀 깊이 차이는1.212m였다.
  평균 차이와 최대 차이는 다른 통계다. 변화 자체는 정확도 향상을 뜻하지 않는다.

이제 K2와 달리 transition이 실제 학습되고 출력에도 영향을 준다.
다만 다음 결과에서는 그 영향이 원하는 경계·정확도 개선으로 이어지지 않았다.

## 3. L256 개발 결과

행의 `학습/추론`은 h 유지 여부를 뜻한다. Reset 학습 가중치에 추론 carry를 켠 행은
학습 때 보지 않은 h 입력을 주는 사후 대조이며, 최적화된 carry 모델로 취급하지 않는다.

| 구성 | Raw AbsRel↓ | 경계 F1↑ | Flat-TV 상대 변화 | 평균 ms¹ | 동시 gate |
|---|---:|---:|---:|---:|---|
| Q0 dense | .159183 | .614665 | 기준 | 7.530 | 기준 |
| 기존 출력-flow K2 | .159120 | .620225 | +0.852% | 4.369 | 통과 |
| 출력-flow K4 | .160207 | .622594 | +3.757% | 2.676 | Edge/Flat 실패 |
| K4 특징 이동, update 없음 | .164306 | .585400 | +5.957% | 4.029 | 정확도·경계 등 실패 |
| Carry 학습 / Carry 추론 | .163943 | .538047 | −3.305% | 4.384 | 정확도·경계 등 실패 |
| 같은 Carry 가중치 / Reset 추론 | .163900 | .540668 | −2.941% | 4.385 | 정확도·경계 등 실패 |
| Reset 학습 / Reset 추론 | .164729 | .541535 | −2.566% | 4.390 | 정확도·경계 등 실패 |
| 같은 Reset 가중치 / Carry 추론 | .164837 | .539022 | −2.921% | 4.381 | 정확도·경계 등 실패 |
| MLP | .164111 | .542644 | −2.555% | 4.274 | 정확도·경계 등 실패 |

¹ 단일 평가의 GPU-resident 동기화 호출 시간. RTX4090에서 CPU flow·내부 전송·검출·cache이동·SSM·decoder·telemetry를 포함한다.
최초 파일IO/H2D·영상 decode는 제외하고 다른 GPU 프로세스도 존재한다.
작은 시간 차이를 모델 고유 비용 차이로 확정하지 않으며 Pi4/Nano 실측으로 해석하지 않는다.

Carry checkpoint에서 h만 끄면 raw가 소폭 낮아지고 경계F1은+.002621 좋아진다.
별도 reset 학습보다 carry 학습의 raw가 좋다는 것만으로 기억의 유익성을 주장할 수 없다.
같은 가중치 개입 결과와 함께 보면 품질·비용상 일관된 이득은 없다.

L32에서도 carry/carry F1 .593387, 같은 가중치 reset .596206,
reset/reset .596123, MLP .596959로 모두 실패했다(Q0 .660219).
출력-flow K4도 L32 edge AbsRel+1.087%로+1% 제한을 초과했다.
L256 출력-flow K4는 edge+1.355%, flat+3.757%다. 약2.81×라는 지연상 이득만으로 채택하지 않는다.

## 4. 시간 변화의 추가 진단

저장된 예측에서 다음 사후 진단을 계산했다.

`mean |(log p_t − log p_(t−1)) − (log Q0_t − log Q0_(t−1))|`

모든 픽셀을 사용하고 source/scene 단위로 균형 집계했다. 낮을수록 Q0의 시간별 변화를 더 잘 보존한다.
이것은 **GT temporal accuracy, flow-aligned OPW, 기존 논문의 TCE가 아니다.**
새 채택 기준을 추가하거나 기존 gate를 바꾸는 데 쓰지 않았다.

| 구성 | L32 | L256 |
|---|---:|---:|
| 출력-flow K2 | .022095 | .031038 |
| 출력-flow K4 | .024712 | .037710 |
| K4 특징, update 없음 | .028601 | .042670 |
| Carry 학습 / Carry 추론 | .030908 | .044250 |
| 같은 Carry 가중치 / Reset 추론 | .030647 | .043992 |
| Reset 학습 / Reset 추론 | .030331 | .043681 |
| MLP | .030204 | .043640 |

이 진단에서도 carry가 Q0의 시간 변화를 더 잘 보존했다는 근거는 없었다.
부드러운 출력이 실제 영상의 깊이 변화를 잘 따라간다는 뜻은 아니므로 flat 감소만으로 안정성을 주장하지 않는다.

## 5. 검증·결정·다음 우선순위

- 관련 테스트 **90 passed**.
- 모든 정책의 전체 갱신2,880회에서 dense Q0와 출력 차이0.
- 같은 가중치의 전체 갱신/update1 비교1,280회 차이0, update2/3에서 실제 기억 개입 효과 확인.
  이들은 정책 사이에 반복된 비교 횟수이며 고유 프레임 수가 아니다.
- 주 실행/감사 source·Q0·training/development/cache hash 계약 검사 통과.
- 원래 논문 모델, 이전 통과 후보, 원고 및 final 평가 결과는 수정하지 않았다.

**새 K4 후보는 모두 미채택한다.** 기존 출력-flow K2를 품질·속도 기준 대조군으로 유지한다.
이번 결과는 SSM 일반이 불필요하다는 증명이 아니라, 현재 특징 누적·학습 손실·주기 조건에서 고유 우위가 없다는 결과다.

특징 update 없이도 K4의 정확도·경계가 악화되므로 오류는 학습된 h만의 문제가 아니다.
이동 특징의 반복 사용과 decoder 재실행 경로도 함께 검토해야 한다. 동시에, 이전 손실의 평활화 trade-off를
그대로 둔 실험이므로 기억 구조만 바꾸면 경계 손상이 해결된다고 기대할 수 없었다.

다음 우선순위는 다음과 같다.

1. 통과한 K2 출력-flow를 기준으로 **깊이 형태 보존 경로**와 **새로 드러난 영역의 갱신 경로**를 분리해 비교한다.
   전체 특징을 계속 수정하는 방식은 이번 결과상 우선순위를 낮춘다.
2. 보정 학습 전에 training 자료에서 depth/경계/flat을 따로 감사하고,
   평활화로 총손실만 줄어드는 문제를 다룬다. 기존 dev 품질 제한은 완화하지 않는다.
3. SSM은 비순환 대조군 대비 품질 또는 갱신 비용 감소가 입증되는 역할에만 남긴다.
4. 후보를 고정한 후 독립 장면/시간 일관성/반복 지연/실기기 단계로 확대한다.

## 산출물 / 재현

- `sokkanaem/qrecurrent_flow.py`, `tests/test_qrecurrent_flow.py`.
- `/home/hyunsu/miniforge3/envs/sokkanaem/bin/python scripts/qrecurrent_study.py --out <새 경로>`.
- 감사: 같은 Python으로 `scripts/audit_qrecurrent.py --out <새 경로>` (이번 실행 경로가 입력으로 고정됨).
- `work_dirs/qrecurrent_20260909/`: protocol/cache/order, 세 checkpoint/history,8개정책 예측/results/parity/integrity.
- `work_dirs/qrecurrent_audit_20260909/audit.json`: gradient·h·같은 가중치 개입·시간 변화 진단.

새 정성 그림은 만들지 않았다. 과거 출력-flow 그림을 이번 K4 학습 결과로 대신 사용하지 않는다.
