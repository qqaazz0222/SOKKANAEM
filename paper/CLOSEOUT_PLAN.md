# 논문화 후속 작업 실행 계획

2026-09-07. 사용자 요청: 남은 항목 진행. 외부 투고·공개 업로드·저자 대신의 윤리/권리 확약은 제외한다.
이 문서는 새 개발 진단과 최종 평가의 결과를 보기 전에 작성했다.

## 기여와 조건부 작업 결정

연구의 기여는 change-gated SSM의 상태 보존/출력 근사 구분, 조건부 skip 오차와 refresh 분석,
공정한 실제 경로 측정과 그 한계다. 최고 정확도, 일반적인 희소화 가속, 최초 update/copy,
전역 수축성, 새로운 장면/고정 카메라 일반화는 주장하지 않는다.
이 방향은 기존 5~12 결과에 근거한 모델 선택 이후의 분석이며 사전등록 연구라고 부르지 않는다.

- 13번: 모델 개선은 조건부로 미실행한다. 성능 우위 개발이 아니라 고정 모델의 분석 논문으로
  범위를 정한다. 기존 K5 진단은 비교 행으로 유지하되 K30 기준을 교체하지 않는다.
  새 학습·loss/threshold/decoder 탐색은 수행하지 않는다.
- 14번: 사용 가능한 엣지 기기 정보가 아직 없다. 새 원고에 엣지 speed/energy 수치 및 우위를
  싣지 않는다. 사용자에게 기기 정보를 요청했으며 미제공 상태를 실측 완료로 처리하지 않는다.
- 독립적인 사람의 수학 검토, 저자/권리/윤리 선언은 자동화로 대체하지 않는다.

## 이론–개발 영상 연결

기존 L256 개발 manifest에서 각 시퀀스의 첫 완전한 클립, 총 4×256프레임을 사용한다.
기본 K30만 평가하며 결과에 따른 튜닝은 없다. 각 프레임에서 다음을 기록한다.

1. 실제 sparse/cache step과 동일 입력 상태의 all-active counterfactual step 간 확장 상태 결손.
2. 독립적으로 계속 dense 실행한 참조와의 누적 상태 차이 및 출력 MAE. GT 정확도 주장이 아니다.
3. 최대/평균 cache age, keyframe/fallback, 활성률, 비활성 패치 픽셀 MSE.
4. 첫 temporal block의 실제 입력에서 추출한 파라미터로 FP64 shared-input recurrence를 재생해
   정리 2의 결손 누적 상한과 실제 shadow 오차를 대조한다. FP32 배포의 전역 인증이 아니다.
5. keyframe에서 국소 결손은 사라져도 dense-history 상태 차이가 남는지 기술적으로 점검한다.

관측된 상수나 최대값을 전역 Lipschitz 상수로 승격하지 않는다. 독립 시퀀스 수는 4이며
프레임 단위 상관이나 이벤트 평균으로 통계적/인과적 유의성을 주장하지 않는다.

## 미사용 최종 평가 조건

모든 모델 선택은 이미 개발 자료를 사용했다. 이 계획·추론 소스·가중치·비교군 구현 및 기존
manifest 해시를 `work_dirs/paper_closeout/final_contract.json`에 기록한 후에만 실행한다.
과거 `paper/freeze/final_test.json`은 당시의 봉인 기록으로 보존하고, 개봉/완료 사건은 별도 기록한다.

- 기준: frozen v11 seed0 EMA, K30, pixel tau=.05/.025, 양쪽 cache, dense fallback=.4.
- 동일 가중치 대조: K5, dense carry, dense reset, output hold.
- 외부 비교: pinned MiDaS v2.1 Small 256, DA2 Small common256(실제252), DPT-Large common256.
  DA2/MiDaS revision은 9~11, DPT revision은 5~8과 동일하다. 미래 프레임 입력은 없다.
- 자료: 봉인된 TUM L256 13클립을 주 평가, L8 462클립을 보조 평가로 전수 실행한다.
  L32는 이번 계획에 포함하지 않는다. 동일 촬영 환경의 새 시퀀스이지 scene-disjoint test가 아니다.
- FP32 eager, TF32 off, scoring 256. 각 클립의 상태를 초기화한다.
- native는 no-fit/median/scale–shift, 상대 모델은 median/scale–shift. L256은 frame-fit도
  보조 형상 진단으로 기록하되 temporal metric을 붙이지 않는다. 실패를 평균에서 빼지 않는다.
- source pixel-pooled와 네 시퀀스 균형 평균을 구분한다. paired sequence bootstrap/sign-flip는
  탐색적이며 n=4, 최소 양측 p=.125 제약을 그대로 공개한다.
- 최종 자료에서 속도는 측정하지 않는다. 개발 subset의 latency와 최종 정확도를 묶어
  동일 품질 speedup을 주장하지 않는다. 테스트 결과를 보고 기준/비교군을 교체하지 않는다.
- smoke는 개발 영상에만 실행한다. 재개는 완료 clip ID만 건너뛰고 결과가 나쁜 clip을 재선택하지 않는다.

## 원고·재현 패키지

기존 초안은 역사 기록으로 보존하고 새 `paper/submission/`에 일관된 영문 원고, 국문 요약,
이론 부록, 표, 참고문헌, cover letter 초안, 저자 확인 목록과 재현 안내를 만든다.
기존 초안의 검증되지 않은 수치/인용/에지 결과를 그대로 복사하지 않는다.
배포 자료와 데이터 경로의 라이선스/권리 확인은 저자 확인 사항이며 임의 공개하지 않는다.
가능하면 로컬 PDF를 빌드한다. 저자 정보·전체 원고 검토·공동저자 승인·정책 재확인이 남으면
`submission_ready=false`로 표시하고 전체 완료라고 보고하지 않는다.
