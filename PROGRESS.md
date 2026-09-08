# PROGRESS

2026-09-08: [깊이 품질 고도화 1·2단계](paper/QUALITY_IMPROVEMENT_20260908.md).
동결 v11 + 2,634파라미터 제한형/경계 국소 보정기 구현, 총 4개 후보 2,400 step 학습,
개발 16클립에서 기존 D1·교사 후보와 동일 조건 비교. 국소 λ=0.5의 경계 F1은
0.3387→0.3581이지만 overshoot 악화로 채택 보류. 관련 테스트 39개 통과.
원본 체크포인트와 논문 final은 유지. 다음 우선순위는 경계 보정 안전성이다.

2026-09-08: [모델 중심 개정](paper/MODEL_FOCUS.md). 제목/초록/서론/방법/결론 재편,
동일 가중치·마스크 ablation 7행 복원. 새 학습·추론·테스트 선택은 없으며 기존 final은 변경하지 않음.

작업 상태 추적용. 실험 수치는 [REPORT.md](REPORT.md), 아이디어 배경은 [IDEA.md](IDEA.md) 참조.
로드맵 4단계는 IDEA.md §7 기준.

## 논문화 후속 작업 — 통합 검토 원고 (2026-09-07)

R3 후속: [최신 draft.md](paper/draft.md)를 LaTeX와 동기화하고 구조/정확도/비용/scale
벡터 그림 4종 및 기존 상태 진단 PNG를 통합했다. 저장된 최종 점수만으로 all-8 scale
통계를 사후 집계했다. [40항목 대응표](paper/self-revision/r3/revision-status.md)와
정정/VDA 감사/재현 부록을 추가했으며, 아래 183개 테스트 기록은 이 보강 전의 기록이다.
R3 검증은 CPU **190 passed / 6 skipped**, CUDA **6 passed**, 기존 freeze **7,608쌍 통과**다.

기기 정보 후속 정정: 사용자가 Nano Developer Kit B01 및 5W/10W 측정을 확인했다.
`measures/nano-b01-5w.log`, `nano-b01-10w.log`를 찾아 [감사 기록](paper/submission/EDGE_LEGACY_AUDIT.md)에
정리했다. 엣지 실측이 전혀 없다는 이전 설명을 철회한다. 과거 합성 활성률·캐시 실측은 존재하나,
고정 가중치/코드 연결과 새 실제 영상 경로의 정확도·end-to-end 검증은 여전히 남아 있다.

[현재 상태](paper/CLOSEOUT.md), [영문 PDF](paper/submission/manuscript.pdf),
[최종 평가](paper/FINAL_EVALUATION.md), [재현 안내](paper/submission/README.md).
이전 단계의 “최종 테스트 미실행/PDF 미검증/원고 통합 미실행”은 당시 기록이며 현재 상태가 아니다.

- 4×256프레임 개발 진단 완료: 후속 keyframe 32곳의 국소 결손 0, dense-history 차이는 잔존.
- 조건·해시 봉인 후 8개 모델로 최종 L256 13클립 및 L8 462클립 전수 평가 완료, 새 학습/튜닝 없음.
- L256 clip scale–shift AbsRel: K30 0.1869, dense carry 0.1837, MiDaS Small 0.1849.
- MDPI 공식 클래스 기반 영문 검토 원고·이론 부록·표·그림·저자 확인란·로컬 재현 묶음 작성.
- CPU 183 passed/6 CUDA skipped, CUDA 6 passed; 기존 freeze 7,608개 file/hash 쌍 보존 확인.
- Pi 4/미확정 Jetson용 비공개 측정 묶음 준비, 데스크톱 CPU smoke만 확인. 기기 접속 정보 대기.

**투고 준비 전체 완료 아님.** 저자는 개인/연구비/COI 정보를 나중에 입력하기로 했다.
인간의 수학·신규성 검토, 공동저자 승인, 현행 정책·재배포 권리 확인이 남아 있다.
새로운 성능 개선·엣지 실측·공개 업로드·외부 투고는 수행하지 않았다.

## 논문화 준비 12번 완료 (2026-09-06)

고정 v11 구현에 맞춘 [수학적 분석·증명](paper/THEORY_12.md)과
[영문 LaTeX 이론 절](paper/theory12.tex)을 작성했다. 신규 학습·모델 변경·실제 영상 재평가·
최종 test 추론은 하지 않았다. 전체 원고 통합과 13~15번은 이번 작업 범위가 아니다.

| 항목 | 완료 내용 | 해석 제한 |
|---|---|---|
| Skip 오차 | binary update/copy 항등식, 구현/ZOH 차이, 공통 입력 SSM의 결손 전파식과 기하급수 상한 | 단일 SSM 수축을 전체 모델 수축으로 확대하지 않음 |
| Cache·출력 | 첫 block의 조건부 age·sqrt(tau) 상한, 전체 모델 perturbation 식, no-fit AbsRel 연결 | 전역 Lipschitz 상수·작은 실용 오차를 인증한 것이 아님 |
| Refresh | 같은 입력 상태 기준 국소 결손 제거, K주기의 조건부 누적 오차 상한 | keyframe은 state reset/dense-history 복원이 아님 |
| 고정 비용 | Amdahl형 가속 손익분기, keyframe/fallback 합집합, 조건부 오차–시간 K 범위 | 1.44×는 입력 비용 고정 후 나머지 profile 비용을 0으로 한 이상화이며 실측 가속이 아님 |
| 반례 | 입력 변화 0에서 dense-state 오차 발생, 공간 문맥 생략, 내부 수축만으로 전체 수축 불충분 | tau 하나로 정확도나 출력 동등성을 보장할 수 없음 |

추가 검증 27개 통과. CPU 전체 테스트 174 passed/6 CUDA skipped, CUDA scan 6 passed.
합성 100개×128step×9좌표에서 오차 항등식 잔차 최대 1.45e−15 미만, 부등식 위반 0.
이는 해석적 증명의 보조 수치 검증이며 실제 데이터의 전역 오차 인증이 아니다.
고정 v11 state 구조 12 MiB와 기존 profile 산술을 재확인했다. 기존 freeze 7,608개 file/hash
쌍은 보존됐고 분석 입력·산출물 해시는 `work_dirs/paper_theory_12/artifacts.json`에 기록했다.
환경에 TeX 엔진이 없어 LaTeX의 링크/환경/참조 구조만 검사했으며 PDF 조판은 미검증이다.

## 논문화 준비 9~11번 완료 (2026-09-06)

고정 v11 및 기존 seed 1/2의 개발 검증을 완료했다. 신규 학습·최종 테스트 추론·외부 제출은
하지 않았다. [전체 보고서·표·그림](paper/STUDY_9_11.md),
[사전 실행 계획](paper/STUDY_9_11_PLAN.md) 참조. 아래 과거 기록과 충돌하는 효율성·상태 비용
해석은 이번 결과를 우선하며, 서로 다른 측정 경계의 시간을 혼합하지 않는다.

| 번호 | 완료한 작업 | 주요 결과 |
|---|---|---|
| 9 | DA2 Small/MiDaS Small 해상도 및 native K 변경 등 15개 운용점, L256 13개 전수 품질 평가 | 시간 측정 subset 4개에서 native/MiDaS256 AbsRel 0.1630/0.1637, 시간 비 1.26×. 전체 13개에서는 0.2091/0.1749로 정확도 열세가 남음 |
| 10 | 동일 4개×256프레임×5회 파일-to-depth 시간, 별도 구성요소 계측, 메모리·state 기록 | native 7.496 ms, dense carry 7.521 ms: 희소화 시간 절감 0.33%에 그침. 별도 계측의 입력 읽기·디코딩·전처리 비중 69.4% |
| 11 | 네 시퀀스 단위 paired bootstrap/sign-flip 및 기존 마지막 8k 단계 seed 0/1/2 평가 | 시퀀스 수가 4라 최소 양측 p=0.125. L256 scale–shift AbsRel 0.2040±0.0077(표본 SD); 독립적인 전체 학습 3회가 아님 |

표의 정확도는 clip scale–shift 보정 후 값이다. 품질 허용선 내 1.26× 시간 비는 사전 지정한
동일 1,024프레임 subset의 운용점 비교일 뿐 통계적 동등성이나 전체 장기 영상의 품질 보존을
뜻하지 않는다. 주 표는 TUM/Bonn 균형 평균, paired 통계는 네 시퀀스 균형 평균으로 구분했다.
작은 모델의 이점과 솎아냄의 이점은 별개이며, native의 최대 stream state 13.501 MiB도
dense carry의 12.000 MiB보다 작지 않았다. 9~11 완료는 투고 준비 완료가 아니다.

검증: CPU 전체 테스트 147 passed/6 CUDA skipped, CUDA scan 별도 6 passed.
기존 freeze 7,608개 file/hash 쌍 보존 확인. 네 실험 단계 각각 기존 7,838개 입력 기록과
추가 174개 기록 재검증 완료. 기존 native 7조건×13클립의 점수 동등성 및 모든 시간/비용
계측 출력의 품질 pass 대비 SHA-256 일치를 확인했다. 생성기·입력·보고서와 12개 보조 산출물의
해시는 `work_dirs/paper_study_9_11/report_artifacts.json`에 기록했다.

## 논문화 준비 5~8번 완료 (2026-09-06)

고정 v11의 개발 검증 재평가를 완료했다. 신규 학습·최종 테스트 예측·외부 제출은 하지 않았다.
전체 결과, 재현 명령, 소스별 CSV와 동일 프레임 곡선은
[paper/STUDY_5_8.md](paper/STUDY_5_8.md), 번호별 범위는
[paper/WORK_PLAN.md](paper/WORK_PLAN.md) 참조. 아래 과거 기록과 충돌하는 정확도·기여 해석은
새 평가 계약 아래의 이번 결과를 우선한다.

| 번호 | 완료한 작업 | 주요 결과 |
|---|---|---|
| 5 | L8 488개 전수, 유효 GT 487개에서 native + DPT/DA2/Zoe common·official 7행 | common-input scale–shift AbsRel native 0.1153, DPT 0.0965: +19.6% 열세 |
| 6 | 모든 비교군의 영역·P50/P95·실패·worst-5 목록 | native는 동적/근거리에서 특히 취약; P95 0.2692 vs DPT 0.1669 |
| 7 | 동일 L256 13개·3,328프레임의 carry/reset/희소/refresh 비교 | K30 0.2091 vs dense carry 0.1822; carry/reset 정확도는 유사하나 TCE는 0.0294 vs 0.0331 |
| 8 | 동일 mask의 Δ/drop/출력 재사용/캐시 분리, 부분 MAC·제한된 추론 시간 측정 | Δ는 drop보다 유리하나 단순 output_hold보다 기본 경로가 우월하다는 근거는 없음 |

위 표의 비교 정확도는 clip scale–shift 후 값이다. No-fit/median 결과도 보고서에 별도로 보존했다.
5~8 완료는 성능 목표 달성이나 투고 준비 완료가 아니다. 시간 측정은 시퀀스당 앞 64프레임의
FP32 eager 고정-mask 경로로 검출기·I/O를 제외한다. MAC은 Linear/Conv2d만 포함한다.
당시 남았던 9~10번 end-to-end/경량 경쟁군, 11번 시퀀스 단위 통계·학습 seed 검증은
위 9~11번 항목에서 후속 완료했다.

검증: CPU 전체 테스트 134 passed/6 CUDA skipped, CUDA scan 별도 6 passed.
기존 freeze 7,608개 file/hash 쌍 보존 확인. 세 실험 단계 각각 7,838개 입력 기록 재검증 완료.
보고서 생성기는 전체 manifest/동일 모델 집합/영역 픽셀 분할/벤치마크 창/결과 해시를 검사하며,
`work_dirs/paper_study_5_8/report_artifacts.json`에 생성기·입력·산출물 해시를 남긴다.

## 논문화 준비 1~4번 (2026-09-06)

기준 v11 가중치/코드/설정 해시와 v8부터의 계보를 기록하고, 기존 holdout은 개발 검증으로 재분류했다.
새 TUM 네 시퀀스의 원본과 L8/L32/L256 목록을 봉인했다(최종 추론 없음). no-GT-fit metric 평가,
실패 처리, 자료 보호와 provenance 기록을 추가하고 ZOH·Skip RNN·상태/출력 구분을 정정했다.
산출물과 검증 결과는 [paper/PREPARATION.md](paper/PREPARATION.md) 참조.
아래 과거 기록의 median=metric, 무조건 정합 불변, GT pooling=엄밀 상한 같은 해석은
[새 평가 계약](paper/PROTOCOL.md)과 [NOVELTY.md](NOVELTY.md)의 개정 내용으로 대체한다.

## 로드맵 상태

| 단계 | 상태 | 비고 |
|---|---|---|
| 1. PoC (4주) | ✅ 완료 (Go) | vkitti2, 스킵 55%에서 AbsRel 열화 +0.15% — 기준(5% 이내) 크게 상회 |
| 2. 본 학습 (8주) | ✅ 확정 체크포인트 `work_dirs/v9-60k` | v3(δ1 0.40) → v5(dim 384) 보류 → v7(실촬 혼합) → v8(60k, 정밀화 4종) → 3-arm으로 teacher 제거·bin CE 확정 → teacher-free 60k → **v9-60k(edge 2.0 + warp 2.0, 실촬 AbsRel 0.1595 / δ1 0.8262 / TCE 0.0323)** |
| 3. 시스템 (4주) | 🟡 커널 완료, 에지 실측 보류(기기 미확보) | 융합 Triton 스캔으로 dense compiled **1.29 ms(776 FPS)**, 희소 2.04 ms — 합격선 2.34 ms 통과. **단 4090에서는 dense가 항상 더 빠르다**(§4.24): 희소의 이득은 연산량 37.0%와 스트림당 state로 한정, 환산 여부는 Jetson 실측이 가른다 |
| 4. 논문화 | 🟡 진행 중 | 초록·§5.2/5.3/5.5/5.6·결론·한계를 확정 체크포인트와 §4.24~4.26 결과로 갱신. 남은 것: 참고문헌, §6 ablation 최신화 |

## PLAN 1라운드 — 계측기·게이트·D1, 그리고 순서가 바뀌었다 (2026-08-29)

[PLAN.md](PLAN.md) §10의 1~3항 완료. 수치는 [REPORT.md](REPORT.md) §4.44.

- **신설**: `sokkanaem/sharpness.py`(경계 P/R/F1, flat TV, overshoot, grad_ratio, edge AbsRel —
  전부 프레임별 정규화 disparity 위, 정합 규칙 무관), `tests/test_sharpness.py`(6건),
  `tests/test_full_res.py`(3건), `tests/test_stage_a_losses.py`(6건). pytest 109건 통과.
- **게이트가 처음으로 실행 가능해졌다**: `scripts/acc_gate.py`에 P1(PLAN §3.1 8항)·P2(§3.2 5항)
  추가. 보고 체크포인트는 P1 1/8, P2 2/5 통과 — 승격선은 어느 쪽으로도 열려 있지 않다.
  `scripts/eval_acc.py --sharp`로 선명도를 **봉인 manifest**에서도 재게 해서 P1과 P2가 서로 다른
  클립 집합에서 측정되던 문제를 없앴다.
- **측정이 계획 순서를 바꿨다**: GT를 patch-16 토큰 그리드로 통과시킨 oracle의 grad_ratio가
  0.2612로 **우리(0.4323)보다 낮다**. 선명도 게이트 0.756은 `dim`/`depth`를 키워서 도달할 수
  없다 — 레버는 용량이 아니라 복원 경로다. 그래서 PLAN §8의 3번(D1)을 1·2번(M0/M1)보다 먼저
  구현했다. 또한 우리 출력은 blur만이 아니라 **ringing도** 한다(overshoot 0.3196, 네 arm 중 최악).
- **신설 구현**: D1 = `DPTDecoder(full_res=True)` — 1/2 해상도 pixel-shuffle residual + 전체
  해상도 RGB detail, **둘 다 zero-init**이라 초기 출력이 D0와 동일(+0.024 GMAC, +0.8k 파라미터).
  Stage A 손실 2종 = `boundary_location_loss`(평탄 영역 gradient L1 + 경계 단측 hinge,
  overshoot을 사지 않음), `rank_loss`(정규화 disparity 위 pairwise ordering, gauge-free).
  용량 노브는 CLI로 노출(`--dim/--depth/--d-state/--dec-width/--full-res`)하고 config.toml
  왕복을 테스트로 고정.
- **M0/M1 실측**: 10.82M/3.803G/3.56ms, 13.55M/4.587G/3.85ms (256px 4090 eager) — PLAN §4.1
  예상과 일치, 셋 다 15M 예산 안.

**arm 결과 (2026-08-29~30, 전부 보고 체크포인트에서 4k fine-tune)** — 수치는 REPORT §4.44.

- **용량은 병목이 아니다**: M0(10.8M)가 학습 손실은 더 낮은데 dense 채점은 전 항목 열세.
  4.19M 유지, M0 전체 학습 보류.
- **선명도 레버는 `boundary_location_loss` 하나**: λ=1에서 grad_ratio 0.431→0.457,
  λ=3에서 **0.578**, boundary F1 0.342→0.441. 가중치에는 붙고 step에는 거의 안 붙는다(12k에서 +0.008).
- **D1(full-res 학습형 업샘플링)은 손실이 요구할 때만 값을 한다**: D1 단독은 무변화, `up_res`
  가중치가 zero-init 그대로였다. boundary와 함께면 boundary 단독을 전 열에서 이긴다.
- **정확도 최고는 실촬 경계띠 trim**: AbsRel 0.1253 → **0.1215**(−3.0%), near −2.8%.
- **기각**: DA V2식 형상 감독 분리(합성 전용)는 실내 경계 감독을 없애 선명도 하락(0.431→0.416).
  DPT식 `fuse_norm`은 AbsRel +14%·precision 하락으로 재차 기각. rank·dynamic·msgrad는 노이즈 수준.
- 게이트는 여전히 전 항목 FAIL. 최고 선명도 arm이 grad_ratio 0.578, 목표 0.756.

진행 중: boundary λ=3/λ=6 + overshoot + trim 결합 arm 2종. 이후 384px, 그래도 미달이면 PLAN §7 Q 트랙.

## 정확도 우선 재설계 — Phase A 완료 (2026-08-25)

[PLAN_ACC.md](PLAN_ACC.md)의 Phase A(A1~A6). **학습은 하지 않았고 평가 계약만 봉인했는데,
계획의 전제 셋이 바뀌었다.** 수치·표는 [REPORT.md](REPORT.md) §4.43.

- **신설**: `sokkanaem/alignment.py`(모든 모델 공통 정합 + 0-crossing failure 집계),
  `scripts/make_acc_manifest.py`, `scripts/eval_acc.py`(ours·HF baseline 단일 경로),
  `scripts/acc_gate.py`(G1/G2 판정), `scripts/acc_summary.py`, `tests/test_alignment.py`(8건).
  `manifests/acc_real_L{8,32,256}.json`에 TUM+Bonn holdout 전 클립을 프레임 단위로 봉인 —
  cap도 loader 순서 의존도 없다. dump 28개는 `work_dirs/acc/`.
- **G2가 잘못 정의돼 있었다.** clip 전체 gauge에서는 **상태가 없는** DPT-Large가 L8→L256
  +116%로 우리(+81%)보다 나쁘다 — 드리프트할 상태가 없으니 전부 정합 창이다. 프레임별 gauge를
  구현해 분리하니 DPT-Large는 평탄(0.0847→0.0850, δ1 0.00pt)하고 **우리만 +20%·δ1 −5.06pt가
  남는다.** 실재하는 형상 드리프트가 그동안 "정합 창 탓"에 가려져 있었다. G2 합격선을 프레임별
  gauge로 옮겼다.
- **재귀 상태는 정확도를 만들지 않는다.** dense(tau=0)와 매 프레임 state reset이 L8/L32/L256
  전부에서 소수 셋째 자리까지 같다(0.1125/0.1130, 0.1180/0.1191, 0.1822/0.1823). 반대로
  TemporalBlock 자체를 우회하면 0.1153→0.2518로 무너진다 — 블록은 **프레임당 용량**으로 필수,
  **기억**으로는 무용. (시간 안정성 지표는 이 실행에서 껐으므로 §4.19 주장은 유효.)
- **격차는 경계가 아니라 움직임·근거리다.** 동적 픽셀 2.66×, 근거리(<2m) 2.38× 열세인데
  깊이 경계띠는 1.15×로 전체 배율과 같다. **TUM 정적 픽셀에서는 4.19M이 343M DPT-Large와
  동률**(0.1097 대 0.1130). 해상도·패치 크기를 첫 레버로 쓰지 않는 판단이 측정으로 뒷받침됐다.
- **비교군 재측정**: DPT-Large 384px가 G1 전 항목 통과(0.0876 / 중앙 0.0792 / δ1 0.9274 /
  실패 0) — 기준선은 실재하고 도달 가능하다. DA V2 Small의 TUM 평균 0.3464는 89클립 중
  **15클립의 정합 0-crossing** 산물이며(중앙값 0.0950), 쉬운 합격선으로 쓰지 않는다.
- **다음**: A6가 새로 만든 후보 A7(동적·근거리 가중 재학습)·A8(상태 없는 대조 학습)을 B 단계
  anchor 작업보다 먼저 잰다 — 둘 다 외부 가중치 없이 현재 구조로 가능하다.

## 리뷰 1차 대응 (2026-08-20)

`paper/self-revision/r1.md`의 리뷰 코멘트를 검증하고 반영하는 라운드. **이 라운드에서 평가
프로토콜 버그를 찾았고, 그 이전의 모든 수치가 바뀐다.**

- **클립 상한이 holdout이 아니라 첫 시퀀스를 표본하고 있었다.** 클립이 시퀀스 순서로 concat되므로
  `--max-clips 100`은 Bonn 399클립 중 `crowd2`만, PointOdyssey 1984클립 중 첫 시퀀스만 평가했다.
  같은 체크포인트가 상한 60에서 활성 55.8%·AbsRel 0.2239, 상한 100에서 44.7%·0.1914로 읽혔다.
  `sokkanaem.data.even_subset`으로 균등 추출(commit 5b3bd37), eval·baseline 3종·probe 전부 수정.
- **표 출처 검증 도구 추가.** `scripts/make_tables.py`(로그 → markdown 행 생성),
  `scripts/table_check.py`(원고의 각 표 행이 단일 run에서 나오는지 검증). `eval.txt` 헤더에
  clip_len/stride/align/keyframe/dense_above/tag 기록. 계기: 표 11이 v9의 AbsRel과 v10의 δ1을
  섞어 쓰고 있었다.
- **Video Depth Anything·Depth Anything 3 체크아웃 복구**(`~/checkouts`, 가중치는 HF 캐시).
  둘 다 현 프로토콜로 재측정해 비교군에 편입 — 영상 전용 baseline 부재(r1-3) 해소.

수정 후 주요 수치 변화:

| 항목 | 이전 | 현재 |
|---|---|---|
| 실촬 AbsRel / 활성률 (8프레임) | 0.1595 / 32.2% | **0.1302 / 22.0%** |
| 희소성 교환 | 13배에 13% | **22배에 6.4%** |
| Bonn 범위비 | 0.47 | **0.75** |
| Bonn 구조적 상한 | 0.0651 (3배 위) | **0.0367 (3.5배 위)** |
| dense 폴백 정확도 이득 | 0.1685 → 0.1633 | **없음** (끄면 활성 16.1%에 동일 정확도) |
| 프레임 인덱스 Bonn 악화 | +61% | **+43%**, OPW·TCE는 평평 |
| 8→32 벌점 | +11.2% | +14.2% / 8→256은 **+86.9%**(장클립 +52.8%) |
| spread 항 정확도 이득 | -5.5% | **노이즈 안**(범위비 0.77→0.90은 유효) |

256프레임 스트리밍 프로토콜을 주 표로 승격했고, 그 프로토콜에서는 상태 없는 baseline이 더 크게
악화한다(DPT-Large +116%, DA V2 Base +145%). 정합 규칙 교차표로 "공통 단일 규칙이 더 공정하지
않다"를 실측했다(상대 깊이 모델에 1-DOF를 강제하면 오차가 한 자릿수 배 증가).

**진행 중**: 최종 설정 후보 `v11-longclip-spread`(v10 + spread 0.5, clip 24, 8k) seed 3개 학습.
완료 시 보고 체크포인트로 승격하고 한국어판까지 동기화한다.

## 현재 상태 (2026-07-24 기준)

- **PC 재설치로 conda 환경 소실 → 재구축 완료**: `conda env create -f environment.yml` + `uv pip install -e ".[dev,video]"`. pytest 14개 전부 통과.
  - **주의: `torch>=2.3` 무제한 상한 탓에 2.13.0+cu130이 잡힘 — `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`로 conv2d 자체가 안 됨.** `torch==2.8.0`(cu128, cudnn 9.10)으로 다운그레이드해 해결, 재발 방지로 `pyproject.toml`에 `torch<2.9` 상한 박음.
  - `work_dirs/*` 일부가 root 소유라 hyunsu 계정으로 못 쓰던 문제 있었음 — 사용자가 별도 터미널에서 `chown` 완료.
- **체크포인트 resume 기능 검증 후 사용**: `scripts/train.py`/`sokkanaem/model.py`의 `{model, optim, step}` 딕셔너리 저장 + `--resume` 복원 로직(이전 세션에서 미커밋 상태로 남아있던 것)을 합성 데이터로 스모크 테스트해 정상 동작 확인.
- **데이터셋 경로 버그 수정 (2026-07-24, 이전 세션)**: `configs/main.toml`, `configs/vkitti2.toml`가 존재하지 않는 로컬 경로를 가리키던 것을 `/archive/Dataset_SOKKANAEM` 하위로 통일.

### 사건: 본 학습 v1 (100k 완주) 붕괴 발견 → v2 데이터 정합성 버그 다수 발견 → v3 재학습

1. **v1 완주 (100,000/100,000 step)**: EMA/Kendall auto-loss-weight/normal-loss/해상도 커리큘럼/LR warmup+cosine/grad clip 전부 적용해 처음부터 학습, 정상 완료된 것처럼 보였음(로그상 loss 변동, 크래시 없음).
2. **holdout eval 돌리다 발견**: `scripts/eval.py` 결과 모든 tau에서 AbsRel 0.80, δ1 0.115, **t-delta 정확히 0.0000**. 직접 forward 찍어보니 **모든 픽셀·모든 프레임이 정확히 같은 상수(4.85)** — 완전 붕괴. raw model/EMA 둘 다 동일 증상(EMA 버그 아님).
   - **원인**: Kendall auto-loss-weight가 `temporal_loss`(연속 프레임 depth 동일해야 한다는 loss) 가중치를 clamp 상한(`exp(8)≈2981×`)까지 밀어붙임 — si_log(~0.5×)·grad(~0.6×) 대비 압도적. `temporal_loss`는 **상수 출력이 정확한 전역 최솟값(0)** 인 트리비얼 해가 있어서, 가중치가 그 정도로 쏠리면 실제 깊이 추정을 포기하고 상수로 붕괴하는 게 수학적으로 최적. IDEA.md §4.5의 모든 과거 ablation(고정가중치 1/0.5/0.1)은 이 문제 없었음 — auto_loss_weight 하나가 원인.
   - **v1 결과 전량 폐기**, `work_dirs/main-COLLAPSED-autolossweight-20260724`로 보관.
3. **holdout eval 재시도 중 데이터셋 버그 별도 발견 (v1 학습 자체와는 무관, 데이터 로더 단)**:
   - tartanair2 10개 env 중 **6개**(OldTownFall + Downtown/ModularNeighborhood/Office/SeasonalForestSpring/Supermarket)가 압축 해제 시 `env/env/Data_easy/...`로 이중 중첩되어 있어서 어댑터 glob이 전혀 못 찾음 — 학습·eval 양쪽에서 완전히 빠져있었음(오염 아니고 그냥 미사용). 전부 `mv`로 경로 정정, 이중 중첩이 없어질 때까지 재귀적으로 확인.
   - `ModularNeighborhood`의 P008은 image 프레임에 827개 구멍(3421 vs depth 4248) — `_pair_sorted`가 이름이 아니라 정렬 순서로 짝짓기 때문에 프레임이 밀려서 잘못 페어링될 위험. P009/P010은 image 자체가 없음(depth만). 셋 다 `Data_easy/_excluded/`로 이동해 어댑터 glob에서 제외.
   - vkitti2는 문제 없음. pointodyssey는 숨김 임시파일(`.depth_00411.png.BWMPW4`, 다운로드 중단 잔재) 하나 발견해 삭제 — Python glob이 dotfile을 안 잡아서 실제 영향은 없었음.
   - **결과**: tartanair2 정상 시퀀스 44→**74**개, 전체 학습 clip 수 160,973→**195,475**개로 증가. v1은 이 버그가 있는 상태로 학습했었음(즉 데이터의 일부 손실 상태로 진행됐던 것 — 붕괴 원인은 아니지만 별개로 고쳐야 했던 결함).
4. **수정 후 v3 재학습 착수**: `configs/main.toml`에서 `auto_loss_weight` 제거, 검증된 고정가중치(si_log + 0.5·grad + 0.1·temporal + 0.05·normal)로 복귀. **재발 방지로 `scripts/train.py`에 붕괴 자동감지 추가**(`sokkanaem/collapse.py`) — depth 예측 std가 `--collapse-eps`(기본 1e-4) 밑으로 `--collapse-patience`(기본 1000) step 연속 유지되면 즉시 중단·체크포인트 저장 후 종료, 8시간 날리기 전에 조기 발견. tmux 세션 `main`에서 처음부터 재학습 중, `mixed dataset: 195475 clips from 3 sources` 확인.
5. **v3 1차 시도, step ~1600에서 `OSError: image file is truncated`로 DataLoader worker 죽으면서 크래시**: 스크래핑 데이터셋(다운로드 중단 잔재) 특성상 손상 파일이 더 있을 수 있다고 보고, `ClipDataset.__getitem__`(`sokkanaem/data.py`) 자체를 방어적으로 고침 — `OSError`/`ValueError` 발생 시 경고 로그 남기고 다른 랜덤 클립으로 최대 10회 재시도 (개별 파일 하나 찾아 지우는 대신 로더 레벨에서 근본 수정, 향후 또 다른 손상 파일 나와도 학습 안 죽음). 체크포인트 저장 전(2000 step 이전) 크래시라 손실 미미 — 처음부터 재시작.
   - **후속 조사**: 실제 학습 중 걸린 손상 클립(`tartanair_v2/Hospital` P003/P004/P006)을 학습 밖에서 PIL로 직접 재검사(image 19950개, depth 19950개 전수) — **전부 정상 로드됨**. 즉 영구 손상이 아니라 `num_workers=4`가 `/archive`에 동시 접근하면서 생긴 **일시적 read 실패**로 추정. 로더를 "같은 파일 2회 재시도 → 그래도 실패하면 랜덤 클립 폴백"으로 개선(멀쩡한 데이터를 불필요하게 버리지 않도록). 이미 20%+ 진행된 현재 런은 안전하게 도는 중이라 재시작 안 하고 다음 런부터 적용.

## 완료된 마일스톤 (날짜순)

- **2026-07-06** — Δ-gated Mamba PoC 초기 구현 (`ab6014b`)
- **2026-07-07** — 이동 카메라 확장(GMC+feature gating) 설계 및 구현, vkitti2 PoC 결과 확보(Go 판정), wall-clock 베이스라인, 청크 segment-sum 스캔(3.6× FPS), static-patch 캐싱(phase 3, opt-in)
- **2026-07-08** — CUDA graph 캡처(2.4–2.9×), detector-driven 마스크 학습 플래그, eval-only ablation(keyframe 주기, MSE vs cosine), 본 학습 인프라(주기적 ckpt, holdout split, aspect crop), 256px OOM 수정(gradient checkpoint), 장기 스트림 드리프트/스트리밍 active% 조사, 마스크 분포 3-arm ablation(iid random @ 0.5 확정)
- **2026-07-08** — 본 학습 데이터셋 다운로드 스크립트 작성 (`/archive`로 VDA 믹스)
- **2026-07-09** — 스톨 워치독 추가, tartanair 다운로더를 `hf_hub` → `wget -c`로 교체 (다운로드 중 재시작 버그 실측·수정)
- **2026-07-10** — tartanair2 + pointodyssey 어댑터, `configs/main.toml` 작성 → 본 학습 착수
- **(런타임, 로그 기준) ~2026-07-14** — 본 학습 7500 step까지 진행 후 중단
- **2026-07-24** — 데이터셋 경로 버그 수정, README에 데이터 위치 명시, checkpoint resume 기능 작업 중
- **2026-07-24** — 본 학습 v1 100k step 완주 → 완전 붕괴(상수 출력) 확인·폐기, tartanair2 6개 env 데이터 미사용 버그 발견·수정(160,973→195,475 clips), 붕괴 자동감지 추가, v3 재학습 착수
- **2026-07-25** — v3 100k step 완주(붕괴 없음), 손상 프레임 방어 로더 추가(1차 재시도 중 크래시 겪고 수정), holdout eval 정상 확인(AbsRel 0.382/δ1 0.397, 스킵 100%→11%에도 정확도 유지+t-delta 0.214→0.154 개선), EMA vs raw 사실상 동일(cosine LR이 막판 0 근접이라 차이 안 남)
- **2026-07-25** — 외부 SOTA 3개(DA v2, DA3, Video Depth Anything metric) 같은 holdout으로 직접 실행·비교. `sokkanaem` env 보호 위해 별도 conda env 2개(`baselines`, `vda`) 신설. 결과: 단일 프레임 정확도는 파라미터 규모 순(전부 우리보다 큰 모델이 더 정확, 예상대로), **시간 안정성은 SOKKANAEM이 압도**(명시적 시간 모듈 있는 VDA조차 t-delta 17배 나쁨) — REPORT.md §4.8

### 2026-07-26 — 논문화 전 감사: 지표·회계 결함 4건 발견, v6/v7 계획 확정

논문 초고 착수 전에 주장-근거 정합성을 점검한 결과, 고쳐야 할 것이 4건 나왔다. 상세 수치는
REPORT.md §4.10–4.13.

1. **지표 프로토콜 불일치**: `eval.py`는 t-delta를 raw 출력에서, baseline 3종은 median-scaled
   출력에서 계산 중이었음(v3 median scale 0.77). 영향 0.95배로 결론은 유지되나 프로토콜 통일.
   재발 방지로 모든 per-clip 지표를 `sokkanaem/metrics.py:clip_scores` 한 곳으로 모아
   eval.py·baseline 3종이 같은 함수를 쓰게 정리.
2. **시간 지표 퇴화**: t-delta는 상수 출력이 전역 최적(§4.6 붕괴가 0.0000). OPW(flow-warp)도
   상수 필드엔 무력. → RAFT 기반 OPW + **TCE**(GT 자신의 워프 잔차 기준, 퇴화하지 않음) 추가,
   `--control`로 상수 예측 제어행 병기. TCE의 const 행 = 데이터셋 고유 워프 잔차 바닥으로,
   우리 모델은 그 바닥보다 위 → **절대적 시간 일관성 우위는 주장 불가, baseline 대비만 가능**.
3. **표본 편향**: 모든 홀드아웃 수치가 8,929 클립 중 100개(1.1%), 분산 미표기.
   1,000 클립 재평가에서 $\delta_1$ 0.397 → **0.546**, active% 16.6 → 31.6으로 크게 이동
   (클립별 AbsRel std 0.389). §4.7/§4.8 표는 폐기·대체. baseline 3종도 1,000 클립 재실행 필요.
4. **기여 3(연산량∝변화율) 기각**: FLOPs를 처음 계산해보니 Δ-gating만으로는 active 0%에서도
   풀연산의 **96.4%**. 디코더가 MAC의 67.8%(IDEA §3.3의 자체 예산 22배 위반), Δ-gating은
   static 토큰의 state 갱신만 없애고 readout(58.5%)은 남김. §3.3의 "active 56%에서 372→373 FPS"
   미해명 수치가 이걸로 설명됨(게이팅의 속도 기여 0). 4090에서는 launch-bound라 디코더 실측
   비중은 3%뿐.
   → **v6**(`configs/main_v6.toml`): `ShuffleDecoder`(디코더 비중 68%→7%) + **학습 단에서**
   spatial cache 활성화(지금까지 inference-only 근사였던 경로를 학습된 경로로 승격). 투영:
   active 16.6%에서 38.6%, 0%에서 26.3%. v5 완주 후 착수(사용자 결정).

또한 **실촬 고정카메라 첫 평가**(TUM `fr3/sitting_static`, zero-shot): active%는 $\tau$=0.05에서
**5.9%** 로 CCTV 주장 구간 실측 입증. 그러나 절대 성능은 **상수 예측기에 패배**
(AbsRel 0.371 vs 0.244, $\delta_1$ 0.388 vs 0.616, TCE 0.037 vs 0.011) — 합성 옥외 학습이
실촬 실내로 전이 안 됨. → **v7**: 실촬 실내 fine-tune(TUM static + Bonn Dynamic, 다운로드 중).
로더 결함도 수정: TUM/Bonn은 rgb·depth 타임스탬프·프레임 수가 달라 정렬순 페어링이 GT를 밀고
있었음 → `_pair_by_timestamp`(0.02s 창) + 회귀 테스트.

게이팅 위치 ablation(§4.4의 미실시 항목)도 구현: `gate_mode="drop"`(static 토큰 블록 우회)를
추가해 Δ-gating의 state readout 기여를 iso-active로 분리 측정(실행 중).

### 2026-07-27 — 학습 크래시, I/O 병목 발견, 정확도 최우선 전환

1. **학습 2건 네이티브 크래시** (원인 미확정): v5 `Illegal instruction` (step 45350),
   v7 `Segmentation fault` (step 7850), 약 1시간 간격·서로 독립(v7은 v5 사망 20분 후 시작).
   Python 예외 아님. dmesg 권한 없어 XID/MCE 확인 불가. 메모리 58GB 여유·디스크 정상.
   체크포인트(2000 step 주기)로 v7은 재개해 15k step 완주, v5는 step 44000에서 대기 중.
   **재발 시 원인 추적 필요.**
2. **v7 결과**: 실촬 미학습 홀드아웃에서 AbsRel 0.179 / $\delta_1$ 0.790 — 상수 제어행
   (0.288 / 0.556)을 명확히 이겨 §4.12의 전이 실패 해소. 합성 홀드아웃도 망각 없이 전면 개선
   (AbsRel 0.4292→0.4166, $\delta_1$ 0.5285→0.5327, t-delta 0.2455→0.2131, TCE 0.0879→0.0808).
   실촬+합성 혼합이 양쪽 도메인 모두에 이득.
3. **사용자 지시로 최우선 전환**: 절대 정확도를 최소 DA v2 수준으로. 진단 결과는 REPORT §4.17
   (요약: 용량이 아니라 일반화. 학습 클립에서 이미 $\delta_1$ 0.707). 이식한 관행: 클립 일관
   증강(기존에 증강이 전혀 없었음), DPT식 다중스케일 융합 디코더(파라미터 0.31M→0.13M),
   disparity 회귀 + MiDaS 다중스케일 gradient 손실. 프로브 3종 실행 중.
4. **I/O 병목 발견·해결**: `/archive`가 스피닝 HDD(ST8000DM004)라 208k 클립 랜덤 접근이
   seek-bound — GPU 사용률 0%, 해상도와 무관하게 ~48 frame/s. NVMe(990 PRO, `/`에 1.7T 여유)로
   333GB 전량 복사 후 **0.67s/step → 0.135s/step (5배)**. workers 4→16 + persistent_workers +
   prefetch_factor 4도 함께 적용. 모든 config·스크립트를 `/home/hyunsu/dataset_ssd`로 전환
   (`/archive`는 원본 보관). **이전까지의 모든 학습 시간 추정치는 5배 과대**였음.

### 2026-07-27 — v8 정밀화: Mamba-depth 계열 기법 4종 이식 + 희소 경로 확장

Mamba 기반 깊이 추정에서 쓰이는 고정밀화 기법을 **SOKKANAEM의 논지(변화 없는 영역은 연산을
내지 않는다)를 깨지 않는 위치에만** 넣었다. 원칙: 추가 용량은 active%에 비례하는 경로(공간
스캔)나 dense여도 무시 가능한 비용(depthwise 3x3, 1/2해상도 bin 로짓)에만 배치. 디코더(dense)에는
더 붓지 않는다 — §4.11에서 기여 3을 기각시킨 범인이 바로 dense 디코더였다.

| 기법 | 구현 | 배치 근거 |
|---|---|---|
| SS2D 4방향 cross-scan (Vim/VMamba) | `ssm.py:BiSpatialSSM(directions=4)` + `column_major_order` | raster flatten은 수직 이웃을 gw=16 스텝 떨어뜨려 수평 경계를 뭉갠다. 열우선 순서로 한 쌍 추가. 비용 2배지만 이 경로가 spatial_cache 아래에서 active%에 비례하는 유일한 경로 |
| CNN-Mamba 하이브리드 / local refinement | `model.py:SpatialBlock(local_conv=True)` — depthwise 3x3 x2, 스캔과 병렬 | dense지만 2·dim·9 MAC/token = 스캔의 0.6%. 입력이 dense readout이라 **희소 상태에서도 근사가 아니라 정확**(마스크는 쓰기만 게이팅) |
| 멀티스케일 융합 | 기존 `DPTDecoder`(2026-07-27 프로브)로 충족 | 주파수 영역 정렬은 미적용 — 멀티모달(이벤트 카메라) 입력이 없어 분리할 대상이 없음 |
| Continuous depth binning | `DPTDecoder(bins=64)` — 학습된 log-depth bin 중심에 softmax 기대값 | 스칼라 회귀는 폐색 경계에서 평균을 내지만 분포는 bimodal이어도 하나의 값으로 수렴. bin은 전역·학습형이라 추론 비용은 마지막 3x3의 16·64 vs 16·1 |

요청 범위 밖에서 추가로 넣은 2건:

- **`temporal_cache`** (`TemporalBlock.step_cached`): Δ-gating은 static 토큰의 *state* 갱신만
  없앴고 readout(in_proj/out_proj/C = active 토큰의 58.5%)은 여전히 dense로 냈다. static 패치는
  정의상 픽셀 변화가 τ 미만이고 state는 bit-identical이므로 블록 출력을 재사용. spatial cache와
  같은 트레이드(τ로 유계, 키프레임마다 갱신, 학습 경로로 승격). **기여 3에 남은 가장 큰 구멍**.
- **`--teacher-weight`** (`distill.py:load_frozen_teacher`/`affine_invariant_loss`): frozen DA v2
  Small의 상대 disparity를 affine-invariant L1로 증류. 학습 전용, 추론 비용 0. §4.17 진단(용량이
  아니라 일반화)을 DA v2 자신이 쓴 방법으로 정면 공격하고, Kinect GT가 비어 있는 픽셀에도 타깃을 준다.

**측정된 회계 변화** (256px, `scripts/flops.py --decoder dpt --scan-directions 4 --local-conv --bins 64`):

| | v7 구조 (conv 디코더, 2방향) | v8 |
|---|---|---|
| full compute | 2.337 GMAC | **1.644 GMAC** |
| dense 몫 (embed+decoder) | 69.4% | **25.4%** |
| active%에 비례하는 몫 | 21.8% | **62.0%** |
| active 16.6% (캐시 전부) | 1.841 GMAC (78.8%) | **0.623 GMAC (37.9%)** |
| active 0% | 1.743 GMAC (74.6%) | **0.420 GMAC (25.5%)** |

즉 full compute가 30% 싸지면서 동시에 정밀화 모듈이 들어갔고, 절감 곡선이 ideal에 2배 가까워졌다.
남은 바닥 25.5%는 dense embed(2.3%) + DPT 디코더(23.1%) — 다음 효율 레버는 **디코더 자체의 희소화**
(변하지 않은 패치는 이전 depth 타일 재사용)이며, DPT의 RGB stem이 dense conv라 타일 단위 재작성이 필요.

검증: `tests/test_arch.py` 8건 신설 — 열우선 순열의 정확성(전체/부분집합/가역), 4방향이 실제로
2방향과 다름, 모든 기능 ON에서 동일 프레임 active 0%·depth 안정·희소 경로가 full compute와 일치,
temporal cache가 hidden state를 bit-identical로 유지, bin 중심 단조·범위 내, affine-invariant
손실이 teacher gauge에 불변. 전체 52건 통과. v7→v8 가중치 승계는 `--resume-partial`(비엄격 로드,
스케줄 step 0 재시작) — 실측 84개 신규 텐서/8개 미사용(v7 conv 디코더).

또한 이전 세션의 미커밋 증강 테스트 1건이 실행 순서에 따라 실패하던 것 수정 — depth는 nearest
리샘플로 평탄 구간이 생겨 strict monotonic assert가 성립할 수 없음. 불변식을 "rgb/depth 방향 일치"로 교체.

### 2026-07-28 — v8 학습 완주 + 3-arm 처치 분리 (수치는 REPORT §4.18)

- **v8 60k step 완주** (12h58m, 크래시·붕괴 없음). `work_dirs/main_v8/latest.pt`.
- **eval 결과가 도메인별로 갈림**: 실촬 미학습에서 v7 대비 개선(AbsRel 0.179→0.1642,
  $\delta_1$ 0.790→0.8111), 합성 holdout에서는 후퇴($\delta_1$ 0.5327→0.4573).
- **3-arm 프로브로 원인 분리** (각 8k step): **teacher 출력 증류가 순손실**이었다 —
  끄기만 해서 실촬 $\delta_1$ 0.8156→0.8405, 합성 0.4155→0.5629. 대조군(arm0)이 v8과
  동일해서 추가 학습 효과가 아님이 확인됨. config에서 제거.
- **bin 감독 손실 신설** (`losses.bin_ce_loss`, `--bin-weight`): 감독 없는 bins=64는
  **퇴화**해 있었다(엔트로피 0.77, 경계 0.776 vs 평탄 0.773으로 구별 없음, 99.9% 픽셀에서
  최빈 bin 질량 <50%). 감독 후 0.408/0.354로 경계>평탄 관계가 처음 성립하고 실촬 AbsRel
  −6.5%. 최종 조합 = v8 구조 + teacher off + bin CE = `work_dirs/arm2-binloss`
  (실촬 AbsRel **0.1386** / $\delta_1$ **0.8472** @ active 17.9%).
- **정성 시각화** `scripts/viz.py` → `outputs/v8/`, `outputs/arm2-binloss/` (RGB|예측|GT 10장).
  전경 물체가 안 살아난다. 원인은 그리드 상한이 아님 — 모델(0.2511/0.7446)이 **32px 블록
  상수 오라클(0.1459/0.8141)보다도 나쁨**. 표현 품질이 다음 병목.
- **남은 판단**: arm2는 실촬에서만 v7을 이긴다. 합성 RMSE 21% 열위(38.49 vs 31.77),
  t-delta·TCE도 열위 — 원거리 bin 해상도 부족 후보.

**다음 후보** (사용자 논의 중): ① frozen DA v2 특징 → 우리 디코더 프로브로 백본 병목 여부
판정 ② v4 특징 증류 실패 원인 규명(단일 블록·cosine only였음) ③ 그래도 필요하면 자기지도
사전학습(IN-1k 지도 사전학습은 dense task 정렬·비용·시간축 블록 미사용 때문에 후순위).

### 2026-07-29 — 감사 P0/P2 반영: 소스별 평가, 재현성 배관, wall-clock 도구 (수치는 REPORT §4.19)

- **`reports/20260729.md` 감사 수행** 후 코드 정정. first-N 표본 편향으로 "합성 500클립"이
  실제로는 VKITTI2 500클립이었던 문제를 `eval.py` 소스별 loader로 해결(`--max-clips`는
  소스당, `MEAN(src)`/`POOLED(px)` 통합행 분리, unscaled metric·scale drift 컬럼 추가).
- **active ratio가 소스별로 14.5%~70.3%** — "15.4%"는 VKITTI2 값이었다. 두 cache 기준
  해석적 MAC은 실촬 42.1%, 합성 53.1%(TartanAir 단독 77.9%)로 정정.
- **§4.18의 "합성 시간 안정성 후퇴" 판정 철회**: 데이터셋 균등 평균에서는 arm2가 v7 대비
  t-delta·OPW·TCE 모두 우위. RMSE 열위는 유지.
- **재현성**: `--seed` 추가, work dir `config.toml`을 실효 설정(+`[meta]` git commit/버전/
  실행 명령)으로 기록, 같은 메타를 체크포인트에도 저장. arm1/arm2 config의 잘못된
  `teacher_weight=0.5` 정정.
- **기본 추론이 희소 경로를 껐던 버그 수정**: `--spatial-cache`/`--temporal-cache` tri-state,
  `--size`를 체크포인트 학습 해상도에서 복원(128→256).
- **`scripts/bench.py` 신설**: active ratio별 latency/FPS, cache on/off, fp16, 멀티스트림,
  peak VRAM, 스트림당 state.
- **teacher-free 60k 재학습 완주** (07-29 06:55→21:26, 14h31m):
  `work_dirs/v8-teacherfree-60k`, seed 0, v7 EMA partial init. 실촬 AbsRel **0.1685** /
  δ1 **0.8083**(MEAN(src))로 arm2와 동급 — **arm2 수치가 회복 학습 아티팩트가 아님이
  재현 확인**. 합성 RMSE 열위는 세 체크포인트 공통으로 남음.
- **확정 wall-clock**(GPU 단독 점유, 256px, 4090): 실촬 평균 active 22.2%에서 4.94 ms /
  **203 FPS**, dense 11.67 ms 대비 **2.36배** — 해석적 예측 2.38배와 일치. §4.11의
  "MAC 절감이 FPS로 안 나온다" 판정은 v8 구조에서 뒤집혔다. 단 compiled dense가 4.67 ms라
  실촬 평균에서 희소 경로와 동률 — Triton 블록 희소 커널 필요 근거.
- **배포 경로 버그 2건 수정**: ① fp16 추론 전체 불가(detector의 fp32 mask가 스캔을 fp32로
  승격) ② 4방향 scan + `--compile`이 cudagraph pool 덮어쓰기로 사망. 체크포인트 `[meta]`의
  `TorchVersion`이 `weights_only=True` 로드를 막던 문제도 수정, 셋 다 회귀 테스트 추가.

### 2026-07-30 — 약점 해소 계획([PLAN.md](PLAN.md)) 수립 + Tier 0 완주 (REPORT §4.20)

외부 모델 대비 약점 12개(W1–W12)를 Tier 0~3 체크박스로 정리하고, Tier 0을 detached 큐
(`work_dirs/tier0.sh`, 03:47–14:20)로 완주했다.

- **고활동 dense 폴백 채택**(`dense_above=0.4`): 실촬 AbsRel 0.1685→**0.1633**, δ1
  0.8083→**0.8211**. 대가는 active 22.2%→32.2%와 t-delta 0.0881→0.0915이고, 이득은 전부
  Bonn에서 나온다. **속도 주장은 이제 active 32% 기준으로 인용해야 한다.**
- **seed 분산 분리**(8k × 6): bin CE의 실촬 이득은 seed 분산의 3~6배로 재현. 새로 드러난
  대가는 t-delta +8%·TCE +12%. **합성 δ1의 seed 표준편차 ±0.015**가 앞으로의 판정 기준.
- **baseline 재생성**: DA3-Base(0.12B)는 실촬 정확도·TCE에서 우리보다 낫고 t-delta만 열위.
  **동급 크기 DA v2 Small(24.8M)에는 AbsRel·RMSE·t-delta·TCE 모두 우리가 앞선다**(δ1만 열위).
- **scale drift**: 8→32프레임에서 drift ×1.7~1.9(랜덤워크 ×2보다 낮음 = 확산).
- **`scripts/bin_probe.py` 신설**로 T1-5 재설계: head의 양자화 바닥이 80 m 미만에서 이미
  AbsRel 0.0000이라 **bin 개수는 병목이 아니고**, `d_max=150`이 만드는 115 m 상한을 넘는
  vkitti2 픽셀 0.8%가 제곱오차의 54%를 낸다. 128 bin·log-disparity·adaptive bin 계획 폐기.
- **Tier 1 착수**(`work_dirs/tier1.sh`, 6 arm × 8k): `d_max` 600 계열 2개, 신규
  `warp_residual_loss`(TCE의 학습판) 2개, 신규 `edge_weighted_loss`(전경 경계 가중) 2개.

### 2026-08-06~07 — Tier 1 완주(v9-60k 확정) + T2-9 버킷 패딩 (REPORT §4.22, §4.23)

- **T2-9 버킷 패딩 채택**: `pad_to_bucket`이 active 토큰 수를 64의 배수로 올리고 패드를
  Δ-gating으로 꺼서 **결과 불변**(테스트로 고정), 그 위에서 `compile_sparse()`가 스캔만
  컴파일한다. active 22%에서 4.87→**2.99 ms**(334 FPS)로 compiled dense 대비 **1.57배**,
  실촬 평균 32%에서 1.19배. **교차점은 active 50%**이고 70%에서는 dense가 낫다 — 그 오른쪽
  끝은 T0-3 dense 폴백이 자동 처리한다. 버킷만 켜고 컴파일 안 하면 항상 느리다.
- **T1-8 최종 60k 확정 — `work_dirs/v9-60k`**(edge 2.0 + warp 2.0, 13h11m). 직전 확정
  체크포인트 대비 **연산 증가 없이**(active 32.2% 동일) 실촬 AbsRel 0.1633→**0.1595**,
  δ1 0.8211→**0.8262**, t-delta 0.0915→**0.0751**, TCE 0.0351→**0.0323**, 합성 RMSE
  15.18→**14.22**·OPW −32%. DA3 대비 AbsRel 격차 +31%→**+28%**, 원시 t-delta 우위
  1.12배→**1.36배**.
- **8k 프로브가 60k를 예측하지 못했다**: 8k에서 edge2+warp2는 대조군보다 실촬 AbsRel이
  나빴는데(0.1822 vs 0.1773) 60k에서는 앞선다. warp residual은 수렴 후기에 비용을 회수한다.
  → **8k arm 스크리닝은 손실 항의 부호는 정해도 크기는 못 정한다.**
- **귀인 한계 명시**: v9-60k와 대조군은 init 계보(main_v8 vs main_v7)와 누적 step이 달라
  60k 수준의 깨끗한 손실 ablation이 아니다. 최고 체크포인트로만 확정하고 인과 주장은 보류.
- **`v9-edge-60k`(edge 단독 대조 arm) 폐기**: step 23100/60000에서 원인 미상으로 죽은 채
  10일 방치됐고, 큐에 넣은 근거(8k 정확도 1위)가 위 반전으로 무너져 재개하지 않는다.
  디렉터리는 부분 학습 상태로 보존. 같은 큐의 32프레임 drift eval도 미실행으로 남긴다.

### 2026-08-18 — T2-10 융합 커널, T2-12, T3-13/T3-14 (REPORT §4.24~§4.26)

- **T2-10 융합 Triton 스캔 — 합격선 통과, 그리고 희소 우위의 소멸.** 병목은 gather가 아니라
  스캔이었다(희소 프레임의 71%). `scan_triton.py`가 청크별 쌍별 감쇠 텐서를 없애고 재귀를
  레지스터에서 돈다(추론 전용, 학습은 미분 가능 경로 유지). Δ-게이팅 bit-exact 유지,
  **평가 지표 소수점 4자리까지 불변**. dense eager 11.38→**1.98**, dense compiled 4.70→**1.29**
  (776 FPS), 희소+버킷+컴파일 2.99→**2.04 ms**. 합격선 2.34 ms 통과.
  **그러나 이제 모든 active에서 dense가 더 빠르다** — 희소가 아끼던 스캔이 공짜가 되면서
  고정 오버헤드만 남았다. wall-clock 시스템 기여 주장은 철회, 연산량·메모리로 한정.
- **T2-12** 운용 범위: "active>50%면 dense" → "**4090에서는 항상 dense**".
- **T3-13 cross-domain zero-shot (KITTI raw 5드라이브, 885프레임).** 정확도는 넘어간다
  (AbsRel 0.2894·δ1 0.4955, in-domain 합성 holdout보다 오히려 나으나 난이도가 달라 서열 아님).
  **넘어가지 않는 것은 희소성**: active 25.8% → **92.8%**. 검출기만 따로 재도 같은 그림
  (합성 7~10% vs 실촬 40~74%) — §3.1~3.2의 스킵률은 합성에서 측정된 낙관값이다.
  NYU(호스트 무응답)·ScanNet(서명 필요)은 미실시.
- **T3-14 GMC 실촬 검증 — 곡선으로 비교해야 보인다.** 기본 임계 on/off 비교는 동률처럼
  보이지만(연산만 더 씀), 같은 active에서 비교하면 GMC가 확실히 낫다: **43.8%에서
  AbsRel 0.3084·δ1 0.5342** vs 픽셀 51.1%의 0.3357·0.4749. **GMC는 14.1% active에서도
  픽셀 51.1%보다 전 지표 우위.** 단 기본 임계는 실촬에서 active 100%라 도메인별 재조정 필수.
- **신규 도구**: `scripts/gate_probe.py`(폴백 끄고 검출기만 측정), `tests/test_scan_kernel.py`.
  테스트 66개 통과.
- **논문 갱신**: §5.2/5.3(확정 체크포인트 + 현행 baseline 표 3b), §5.5(cross-domain·GMC 곡선),
  §5.6(융합 커널과 역전), 초록·결론·한계 목록.

### 2026-08-18 (같은 날, 후속) — 다음 라운드를 정하기 위한 세 측정 (REPORT §4.27)

세 개선 방향(정확도 / 게이팅 근거 / 고활성 효율)을 추측으로 정하지 않으려고 먼저 쟀고,
**셋 다 계획을 바꿨다.**

- **출력 구조 상한 측정 (`scripts/ceiling_probe.py` 신설).** GT를 모델과 같은 토큰 그리드
  병목에 통과시켜 "완벽한 head"의 점수를 낸다. 실촬 holdout patch 16 상한은
  **TUM 0.0844 / Bonn 0.0651**(δ1 0.914 / 0.944)인데 현재 모델은 0.1321 / 0.1869 —
  **상한의 2~2.9배 위**다. patch 16 상한만 뽑아도 DA3(0.1244)를 넘는다.
  → **패치 크기·해상도는 병목이 아니다. 용량·학습·동적 장면이 병목.**
  patch 8 / 384px 투자는 PLAN에서 `[-]` 기각.
- **이 결론은 한 번 뒤집혔다.** 첫 실행이 유효 픽셀 마스킹 없이 pool해서 실촬 depth의
  구멍(0)이 섞였고, disparity 공간에서 1/eps가 패치를 지배해 TUM AbsRel 11.4가 나왔다.
  그 상태의 depth 값(0.1388)은 "우리가 이미 상한을 넘었다"는 **정반대 결론**을 만들 뻔했다.
  마스킹 후 재측정한 값이 위 표다.
- **처리량 재측정 — DA v2 Small 대비 이미 앞선다.** fp16+compile에서 본 모델
  **0.378 ms / 2646 FPS (VRAM 20.6MB)** vs DA v2 Small **0.816 ms / 1226 FPS (49.6MB)** —
  **2.2배 빠르고 메모리 2.4배 적고 정확도도 우위**(0.1595 vs 0.2256). active 70%에서도
  0.371 ms로 동일(dense라 활성 비율 무관). **"고활성 구간 효율" 목표는 이미 달성.**
  §3.4의 "DA V2 Small 380 FPS vs 본 모델 366 FPS(동급)"는 v3 시절 수치로 양쪽 다 낡았다 —
  REPORT §3.4·IDEA §4에 경고 삽입.
- **연산 예산이 2.2배 생겼다** — Tier 4의 핵심은 그걸 어디에 쓰느냐이고, 답은 정확도다.
- **PLAN에 Tier 4 신설**: 방향 1(T4-15~17, Bonn 실패 해부 → dim 288 증설 → 동적 객체 손실),
  방향 2(T4-18~20, 픽셀 임계를 **상태 오차 예산**으로 대체 + 다중 스킵 드리프트 상계로
  keyframe 주기 유도), 방향 3(T4-21~23, 목표 달성 확인 + 희소 경로 존치 판정).

## 2026-08-20~23 — r1 대응, 엣지 실측, 그리고 문서 감사

세부는 REPORT §4.36~4.40과 `paper/self-revision/r1-revision-status.md`에 있다. 요약:

- **평가 표본 버그**로 2026-08-20 이전 수치가 전부 바뀌었다. `--max-clips`가 concat 순서
  앞에서 잘라 Bonn은 `crowd2` 하나만 평가하고 있었다. `even_subset`으로 수정(`5b3bd37`).
  base 체크포인트 실촬 AbsRel 0.1595 → **0.1302**, active 32.2% → **22.0%**.
- **확정 체크포인트 승격**: `v11-longclip-spread-s0`(base 60k → 장클립 25k → spread 8k,
  키프레임 주기 30). 8프레임 **0.1263 / active 22.0%**, 256프레임 0.1907.
- **256프레임 스트리밍을 주 프로토콜로 승격.** 클립 단위 벌점의 대부분이 드리프트가 아니라
  클립당 정합 창이었고, 상태 없는 baseline이 같은 클립에서 훨씬 크게 악화한다.
- **엣지 실측 성사**(Jetson Nano B01): 4090의 서열이 뒤집힌다. 5% active에서 시간 13.7배·
  에너지 15.3배. **Raspberry Pi 4B(2026-08-24)에서 14.2배** — 가속기가 없는 보드라
  "어느 가속기냐의 문제" 반론이 닫힌다. 고정 부기가 dense 프레임의 2.4%(Nano 2.1%)로 두 기기가
  일치. **TX2는 여전히 미측정이고, Pi는 전력 레일이 없어 에너지는 Nano 하나에 기댄다.**
- **rolling refresh 기각**, **dense 폴백은 운용점 2개로 정리**, **warp 가중치는 다이얼**로 확정.
- **문서·그림 감사(08-23)**: Table 7c의 "Reported" 열이 실제로는 base 체크포인트였다.
  보고 체크포인트에서 주기 5·10·15를 새로 재고 표와 그 표를 읽던 문단을 고쳤다. 같은 감사에서
  Figure 3·4·7의 데이터가 낡아 있었고(`make_figures.py`가 표의 사본을 들고 있다), 영문 draft에
  Figure 7이 아예 빠져 있었으며, `caption.md`에 Figure 8 캡션이 없었다. 전부 수정.
- **Figure 9(정성 비교) 생성** — r1 minor 12 해소. `scripts/viz_qualitative.py`.

## 다음 액션

1. **논문 제출본 정리** — r1 minor 1(`[CHECKPOINT-DEPENDENT]` 태그 8개 제거)과
   minor 15(저자·소속·Data Availability·Funding·Acknowledgments)는 **최종 교정본에서 처리**.
   minor 15는 사용자 정보 필요.
2. **Jetson TX2 실측** — 절차는 `EDGE_BENCH.md`에 있다. Nano와 Pi가 이미 두 점을 찍었고
   절편 비율이 2.1%·2.4%로 일치하므로, TX2는 그 상수성을 세 번째로 확인하는 자리다.
   전력 레일이 있으므로 **에너지 결과를 Nano 밖에서 재현할 유일한 후보**이기도 하다.
3. **Orin급 실측** — `sm_53`이 융합 커널을 못 돌려 Nano 수치는 배포 구성이 아니다.
   Orin이 그 공백과 실시간 처리량 공백을 동시에 닫는다.
4. **참고문헌 검증** — 미확인 6건, 특히 Depth Anything 3(0.12B baseline으로 전편에 쓰이는데
   저자·발표처·연도 미확인).
5. 실내 미지 도메인(NYU·ScanNet) zero-shot — 호스트 복구 또는 서명 절차 필요.
6. `tartanair_v2/Hospital` 간헐적 read 실패 — 재발하면 `num_workers` 조정.

## 이전 다음 액션 (2026-08-18 시점, 보관)

1. **T4-15 Bonn 실패 해부** (반나절, 선행 없음) — T4-16·T4-17의 방향을 정하는 선행 작업.
   `work_dirs/v9-60k/scores_real.json`의 클립별 오차 분포에서 최악 10%가 무엇인지 확인
2. **T4-16 dim 192 → 288 증설** (60k, 14.5h) — §4.27b가 준 예산을 정확도에 쓴다
3. **T4-18~20 상태 오차 예산 게이팅** — 픽셀 임계의 도메인 의존성을 수학적으로 제거.
   합격 기준: 한 예산 값이 합성·실촬·주행에서 같은 정확도 손실을 줄 것
4. **논문 마무리** — 참고문헌 작성, §6 ablation 최신화, `paper/draft_ko.md` 동기화
5. **T2-11 Jetson — 기기 미확보로 보류**(PLAN `[-]`). 그때까지 희소성 이득은 연산량·메모리로만 주장
6. 실내 미지 도메인(NYU·ScanNet) zero-shot — 호스트 복구 또는 서명 절차 필요
7. `tartanair_v2/Hospital` 간헐적 read 실패 — 재발하면 `num_workers` 조정
