# 논문화 후속 작업 결과

2026-09-08: [모델 중심으로 원고 개정](MODEL_FOCUS.md). 제목·초록·방법·기여 분리 ablation을
SOKKANAEM 중심으로 재편했다. 아래 조건부 분석 중심의 설명은 이전 개정의 기록이며,
기존 성능 수치와 미완료 검증 항목은 유지된다. 새 모델 개선 실험을 완료한 것은 아니다.

2026-09-07. **통합 검토 원고를 만들었지만, 투고 준비 전체가 완료된 상태는 아니다.**
저자 정보는 사용자가 나중에 입력하기로 했고 임의로 작성하지 않았다. 외부 투고·공개 업로드는 하지 않았다.

R3 보강: [최신 전체 draft.md](draft.md)에 현재 LaTeX의 본문·결과표·증명·재현 부록 및
그림 5개를 동기화했다. [40항목 self-revision 대응표](self-revision/r3/revision-status.md),
[주요 정정](submission/REVISION_NOTES.md), [VDA 감사](submission/VIDEO_BASELINE_AUDIT.md)를 추가했다.
새 scale 분석은 저장 점수의 **사후 기술 통계**이며 새 추론/학습/테스트 선택은 없었다.
연속 L1024·현 가중치 정성/range 진단·현 조건의 video baseline은 부분 해결 항목으로 남겼다.

기기 확인 후 정정: **Nano Developer Kit B01의 5W·10W 과거 실측 로그가 존재한다.**
앞서 엣지 실측이 전혀 없다고 한 설명은 잘못됐다. [기존 실측 감사](submission/EDGE_LEGACY_AUDIT.md)에
수치와 한계를 정리했다. 아래의 미완료는 새 실제 영상 경로와 고정 모델에 연결된 검증을 뜻한다.

| 항목 | 이번에 처리한 내용 | 현재 상태·해석 제한 |
|---|---|---|
| 기여·선행연구 | update/copy·선택적 SSM·영상 재사용 선행연구 대조, 정확도/가속 우위 주장을 축소 | 독립적인 신규성·저널 적합성 판단은 저자/전문가 검토 필요 |
| 이론–실제 영상 연결 | 개발 4×256프레임의 같은 상태 국소 결손, dense-history 차이, cache age, FP64 shadow 상한 | 32개 후속 keyframe에서 국소 결손 0이나 누적 차이는 남음; 전역 정확도 인증 아님 |
| 최종 평가 | 조건·파일 해시 봉인 후 8개 모델, L256 13클립/L8 462클립 전수 | 완료. 같은 TUM 촬영 환경의 네 시퀀스이며 독립 장면 일반화는 아님 |
| 13 조건부 개선 | 고정 모델의 분석 논문 범위를 유지하고 새 학습/튜닝은 미실행 | 성능 개선 완료라고 부르지 않음; 최종 결과를 본 뒤 튜닝하면 새 테스트 필요 |
| 14 조건부 엣지 | Nano B01 5W/10W 과거 합성 활성률·캐시 로그 확인; 새 실제 PNG 경로 runner·해시 묶음 준비 | 과거 센서 전력/에너지 값은 존재. 새 실제 영상 경로·정확도·고정 가중치 연결 검증은 미완료 |
| 15 원고·재현 자료 | 공식 MDPI 클래스 기반 영문 원고, 증명 부록, 표·벡터 그림, cover letter 초안, 검토 목록 | 로컬 검토용 PDF/패키지. 저자 확인과 현행 정책·권리 검토가 남음 |

## 핵심 결과

최종 L256 clip scale–shift AbsRel은 K30 **0.1869**, dense carry **0.1837**,
MiDaS Small **0.1849**다. L8은 K30 **0.1840**, MiDaS **0.1539**다.
보정 없는 metric 정확도가 아니라 보정 후 상대 형상 비교다. K30을 교체하거나 나쁜 클립을 빼지 않았다.
DA2의 L256 정합 실패 2클립도 평균에 포함했다. 시퀀스 단위 추론은 n=4인 탐색적 통계다.

개발 subset의 기존 파일-to-depth 시간 7.496/7.521 ms(K30/dense)는 희소화 자체의 큰 가속을
뒷받침하지 않는다. 최종 정확도와 개발 subset 시간을 섞어 동일 품질 가속률을 만들지 않았다.
논문의 주장은 **정확한 상태 복사와 근사 출력·실제 비용이 서로 다르다는 구현별 조건부 분석**이다.

## 산출물과 검증

- [영문 원고 PDF](submission/manuscript.pdf), [LaTeX](submission/manuscript.tex), [저자 정보 입력란](submission/author_details.tex)
- [최종 결과·표](FINAL_EVALUATION.md), [재현 안내](submission/README.md), [저자/검토 확인 목록](submission/REVIEW_CHECKLIST.md)
- [엣지 실행 안내](submission/EDGE_RUN.md), [로컬 엣지 묶음](../work_dirs/paper_edge_bundle.zip)
- [로컬 원고·소스·증거 묶음](../work_dirs/paper_submission_review.zip), [상태 JSON](../work_dirs/paper_closeout/status.json)

R3 후 CPU 테스트 **190 passed / 6 CUDA skipped**, CUDA scan **6 passed**. 기존 freeze의
**7,608개 고유 file/hash 쌍**과 후속 증거 해시 체인을 재검증했다. 새 표 생성기는 최종 manifest의
전수 ID·RGB/depth 쌍·모델 집합과 결과 해시를 점검한다. 검증 로그는 `work_dirs/paper_closeout/`에 있다.
PDF는 로컬 조판을 통과했으며 남은 경미한 패키지/자간 경고는 빌드 기록에 남겼다.

묶음은 **비공개 로컬 파일**이다. 원고 묶음에는 원본 데이터·외부 baseline 가중치·환경·자격 증명을
넣지 않았다. 엣지 묶음에는 사용자의 로컬 실험 편의를 위해 개발 RGB를 포함했으므로 제3자 제공이나
공개 전 데이터 사용·재배포 권리를 확인해야 한다. 고정 코드의 절대 경로 때문에 새 머신에서 압축만
풀면 모든 평가가 실행된다는 보장은 없으며, 깨끗한 머신의 전체 재현은 아직 검증하지 않았다.

## 사용자에게 남은 확인

저자·소속·교신 이메일·기여·연구비·이해충돌은 사용자가 추후 입력한다. 모든 원고/증명의 인간 검토,
공동저자 승인, 데이터·코드·가중치 권리, 현행 Mathematics 정책은 투고 전에 확인해야 한다.
새 실제 영상 엣지 실측을 추가하려면 Pi 4의 RAM/OS 및 SSH `사용자명@호스트`와 기존 키 인증
가능 여부가 필요하다. Jetson은 Nano Developer Kit B01로 확인됐다. 비밀번호를 보낼 필요는 없다.
