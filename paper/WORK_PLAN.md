# 논문화 작업 순서

사용자와 합의한 원래 표의 번호를 보존한다. 1~4번 결과는 [PREPARATION.md](PREPARATION.md),
평가 계약은 [PROTOCOL.md](PROTOCOL.md) 참조.

| 번호 | 작업 | 완료 기준 |
|---|---|---|
| 1 | 기준 확정 | 체크포인트·계보·설정·코드·해상도 추적 |
| 2 | 데이터 분리 | 학습·튜닝·미사용 최종 시퀀스 구분 |
| 3 | 평가 계약 | no-fit metric / GT-scale-aligned / relative-shape 분리 |
| 4 | 수학적 정정 | ZOH·zero-step·선행연구·상태/출력 보장 범위 일치 |
| 5 | 정확도 재평가 | 동일 프레임의 native와 주요 baseline, common/official input 별도 비교 |
| 6 | 취약 영역 분석 | 동적/정적·거리·경계 영역, P50/P95·실패 클립 목록 |
| 7 | 장기 열화 분리 | 동일 L256에서 dense carry/reset/sparse/refresh, frame/clip 정합 곡선 |
| 8 | 기여 분리 | 동일 checkpoint/mask의 Δ-gating/drop/output reuse와 캐시, 정확도/TCE/연산/시간 |
| 9 | 효율 비교 | 경량 경쟁 모델과 동일 품질·시간 운용점 비교 |
| 10 | 실제 처리시간 | 전처리·검출기·GMC·decoder·keyframe 포함 동일 조건 측정 |
| 11 | 통계 검증 | paired sequence CI 및 학습 seed 반복 범위 명시 |
| 12 | 수학적 보강 | 가정하의 skip 오차·refresh·고정 비용 분석 |
| 13 | 조건부 모델 개선 | 진단이 지목한 최대 1~2개 개선안 검증 |
| 14 | 조건부 엣지 검증 | 실제 입력·실행 경로의 기기 latency/energy |
| 15 | 최종 원고·재현 패키지 | 표·그림·초록·결론·명령·가중치 통일 |

## 5~8 실행 범위 (2026-09-06, 결과 확인 전 고정)

신규 학습 없이 개발 검증 `acc_real_L8.json`, `acc_real_L256.json`을 사용한다. 최종 TUM test는
봉인 상태로 남긴다. 기존 코드·freeze 기록은 변경하지 않고 추가 분석 코드를 별도로 버전 기록한다.

5~6의 비교군은 DPT-Large(대형 상대 형상), DA V2 Small(소형 사전학습 상대 형상),
ZoeDepth N-K(metric)이다. 로컬 캐시의 정확한 snapshot revision과 가중치 해시를 고정한다.
세 모델은 모두 프레임 독립적이며 미래 프레임을 사용하지 않는다. 이번 표는 전체 비디오 baseline
벤치마크가 아니다. DA3/VDA와 모든 경량 경쟁군 확장은 별도 작업이며 미실행 행을 실측처럼 넣지 않는다.

Common-input은 기존 256 RGB crop을 사용한다. Official-input은 원본 RGB에서 동일 채점 ROI를
복원해 공식 processor에 전달한다. 256으로 축소한 RGB를 다시 키우는 실험과 구분한다. 모든 GT와
채점 출력은 256이다. 모델별 실제 processor 입력 크기와 checkpoint hash를 기록한다.

7의 기본 조건은 dense carry, dense reset every frame, sparse K30, sparse K5/K10/K60이다.
8의 마스크는 sparse K30이 실제 사용한 post-fallback mask를 모든 조건에 동일하게 재생한다.
공간/시간 출력 캐시의 2×2 비교와 temporal cache off에서 delta/drop 비교를 수행한다.
단순 출력 재사용 대조군은 memoryless dense 예측의 active patch만 갱신하며, 완전 비활성 프레임
외에는 predictor를 전체 실행한다. 패치 출력 선택을 계산량의 선형 절감으로 잘못 계산하지 않는다.

8의 연산량은 실제 호출된 Linear/Conv2d MAC을 hook으로 집계한다. 융합 SSM scan·elementwise·
정규화·resize·scatter는 제외되므로 total FLOPs가 아닌 측정 가능한 부분 연산량으로 표기한다.
시간은 동일 원본 프레임·고정 마스크의 GPU-resident FP32 eager 경로를 비교한다. 검출기·입출력·
정합·지표 계산은 제외하며 9~10의 end-to-end 배포 비교를 대신하지 않는다.
