# R3 — 전체 self-revision 대응 감사 (2026-09-07)

대상은 [현재 투고용 검토 원고](../../submission/manuscript.tex)다. R1/R2의 `comment.md`는
원문 그대로 보존했다. 이전 `revision-status.md`의 완료 판정이나 “Weak Accept”는
외부 심사 결과가 아닌 당시의 내부 자기평가이며, 아래 판정으로 대체한다.

**총 40항목: R1 major 9 + minor 15, R2 major 2 + minor 14. 전부 해결된 것은 아니다.**

- **반영**: 현재 범위의 원고·표·설명에 반영. 학술적 타당성의 독립 승인과 다르다.
- **부분**: 필요한 일부 증거/설명은 있으나 원래 요청 전체를 충족하지 못함.
- **범위 제외**: 해당 주장을 철회/제외. 그 실험을 완료했다는 뜻이 아님.
- **저자 대기**: 저자만 확정할 정보·판단.

## R1 major (9)

| ID | 요청 | 판정 | 현재 반영 / 남은 점 |
|---|---|---|---|
| R1-M1 | streaming 주장에 맞는 장기 평가 | 부분 | 최종 L8/L256 전수 및 개발 4×256 진단. 연속 L1024는 v9/v10 과거 자료이며 v11 증거로 승계하지 않음. `manuscript.tex` Discussion |
| R1-M2 | 최종 체크포인트·K 확정 후 전 표 일관성 | 반영 | v11 seed-0 EMA, K30 봉인; K5/다른 경로는 고정 대조군. `final_contract.json`, `generated_results.tex`; 시험 후 교체 없음 |
| R1-M3 | video-specific baseline | 부분 | VDA/oVDA 관련 연구 복원, 과거 VDA의 비인과적 경로/출처 한계 감사. 최종 조건에 맞춘 비교는 미완료. `VIDEO_BASELINE_AUDIT.md` |
| R1-M4 | temporal consistency 주장 축소 | 반영 | raw delta/OPW/TCE/scale CV 구별, 보편적 안정성 우위 철회. 새 post-hoc scale 표와 재현 부록 |
| R1-M5 | MAC과 실측 속도 분리 | 반영 | PNG-to-depth 7.496/7.521 ms, 입력 처리 69.4%, 동일 품질 가속으로 해석하지 않음. 과거 Nano는 별도 제한적 감사 |
| R1-M6 | 신규성 구체화 | 부분 | Skip RNN/Mamba/DeltaCNN/Eventful 등 귀속, 조건부 오차·캐시·비용 결합으로 한정. 신규성/Mathematics 적합성의 독립 전문가 판단 필요 |
| R1-M7 | moving-camera/GMC 외삽 | 범위 제외 | 주 모델 GMC off; KITTI나 scene-disjoint 일반화 주장을 하지 않음. 과거 calibration split은 현 주장으로 복원하지 않음 |
| R1-M8 | 통계적 신뢰성·다중 seed | 부분 | 3개 최종 8k-stage seed와 n=4 시퀀스 bootstrap/sign-flip. 전체 학습 seed·더 많은 독립 장면·확증적 검정은 없음 |
| R1-M9 | alignment 공정성 | 반영 | none/median/disparity scale–shift, clip/frame fit 분리; 실패 포함·1000-unit clamp 명시. `generated_results.tex`, 전체 CSV |

## R1 minor (15)

| ID | 요청 | 판정 | 현재 반영 / 남은 점 |
|---|---|---|---|
| R1-m1 | 진행 중 실험 표식 제거 | 반영 | 현재 LaTeX에 queued/adoption pending 등의 과거 표식 없음. 저자 승인 전 working-draft 표식은 의도적으로 유지 |
| R1-m2 | 초록 압축 | 반영 | 상태/출력/비용 구별과 대표 결과·한계 중심; 과거 장문 초록은 제출 대상 아님 |
| R1-m3 | exact 범위 명확화 | 반영 | 제목부터 state preservation ≠ output equivalence; binary/real-arithmetic 및 FP 예외, 1차 입력 근사 구분 |
| R1-m4 | Figure 1 GMAC의 activity 조건 | 범위 제외 | 과거 1.644→0.608 도식은 현 원고에 미사용. 현 비용 표·도식은 조건과 측정 경계를 명시 |
| R1-m5 | token-drop 해석 | 반영 | 재현 부록에 제어 정의와 보편적 정확도 우위 부정. 개발 `study8_ablations.csv`와 최종 output-hold는 별개 제어 |
| R1-m6 | t-delta 단위·정합 순서 | 반영 | clip gauge 후 모든 픽셀의 깊이 차이, 깊이 단위; 무워핑/무GT-mask. `reproducibility_details.tex` |
| R1-m7 | RAFT 전처리·occlusion | 반영 | [-1,1], final refinement, current→previous, nearest warp, GT/inbounds mask, forward-backward 검사 없음. 재현 부록 |
| R1-m8 | GMC failure rate | 범위 제외 | GMC 비활성 주 모델만 주장. 현 최종 모델의 GMC 실패율을 측정했다고 하지 않음 |
| R1-m9 | fallback 0.4 민감도 | 부분 | 과거 draft Table 9와 `r1-nofallback.log` 보존; 현재 0.4와 heuristic 성격 명시. 새 frozen-protocol threshold sweep 및 보편적 최적성 검증 없음 |
| R1-m10 | W+NS 메모리 분리 | 반영 | FP32 W≈15.968 MiB, S=13.501/12.000 MiB, peak VRAM 아님; 재현 부록과 개발 메모리 CSV |
| R1-m11 | 파라미터≠효율 | 반영 | 파라미터·활성률·MAC·실제 latency를 분리. 보편적 소형/고속 우위 없음 |
| R1-m12 | 실제 정성 영상 | 부분 | 과거 그림은 존재하지만 현 frozen v11/frame-ID로 검증하여 통합하지 않음. 최신 모델의 keyframe 전후 사례가 남음 |
| R1-m13 | range compression 주요 결과화 | 부분 | 정확도 한계 및 scale 진단 반영. 과거 GT-range 0.47 수치를 현 최종 모델의 진단으로 승계하지 않음; 현재 matched range 진단은 없음 |
| R1-m14 | 학습 재현 세부 | 부분 | sidecar와 현재 보존 구현의 optimizer/LR/증강/sampler 구별하여 부록 복원. 전 학습 이력·RNG 및 clean-machine 재현 검증은 없음 |
| R1-m15 | 저자·Data/Funding/Acknowledgments | 저자 대기 | 사용자가 추후 추가 예정. 임의 인명·지원·권리·승인을 작성하지 않음 |

## R2 major (2)

| ID | 요청 | 판정 | 현재 반영 / 남은 점 |
|---|---|---|---|
| R2-M1 | drift는 alignment window 탓이라는 과잉 해석 | 반영 | metric 단위도 drift를 겪음, 20-score/8-curve 및 v9/v10 정정. 현 all-8 모델의 CV/log-SD/log-step을 저장 결과에서 복원, post-hoc 표기. additive 분해/endpoint 안정성 주장 철회 |
| R2-M2 | L256 ranking 불확실성 | 반영 | 네 시퀀스 짝지은 bootstrap/sign-flip과 효과크기 보고; CI가 null을 포함해도 동등성 아님. 과거 clip win fraction은 p-value로 사용하지 않음 |

## R2 minor (14)

| ID | 요청 | 판정 | 현재 반영 / 남은 점 |
|---|---|---|---|
| R2-m1 | falls 36%의 오차 방향 | 반영 | 현 초록은 그 역사적 비교 미사용; archived table은 악화(+오차)로 해석, REVISION_NOTES 정정 |
| R2-m2 | 초록 길이 | 반영 | 핵심 조건부 분석 및 제한된 대표 결과로 압축 |
| R2-m3 | 22× update와 MAC 혼동 | 반영 | 해당 update 배수를 현 가속 claim에 쓰지 않음. 비용 모델과 실측을 분리 |
| R2-m4 | cross-model regression은 causal 아님 | 범위 제외 | 회귀-normalized matched-quality 우위 주장을 현 원고에서 제외 |
| R2-m5 | DPT matched-accuracy 사례 우선 | 부분 | DPT를 동일 최종 표에 포함. .1891/.1907 과거 값이나 CI-null 포함을 정확도 동등성으로 승계하지 않음. 사전 정의 equivalence 검정 없음 |
| R2-m6 | 권장 configuration 정리 | 반영 | 하나의 reference K30 설정과 나머지 제어를 재현 부록에서 분리; 사후 성능별 추천 없음 |
| R2-m7 | fallback 표를 본문과 연결 | 부분 | 재현 부록에서 과거 draft Table 9의 존재와 frozen sensitivity 미완료를 명시. unchanged AbsRel 주장이나 현재 최종 결과로 승계하지 않음 |
| R2-m8 | GMC one-drive 일반화 축소 | 범위 제외 | 해당 일반화 주장을 현 원고에서 제외; split-specific 결과를 다른 환경으로 외삽하지 않음 |
| R2-m9 | edge reference scan 명시 | 반영 | EDGE_LEGACY_AUDIT 및 재현 부록에서 reference scan/고정 활성률/합성 입력/모듈 전력을 명시. 현재 실영상 에너지 우위 없음 |
| R2-m10 | optimizer/LR/EMA 등 | 부분 | 재현 부록 복원. sidecar에 없는 구현 설정은 역사적 실행을 입증하지 못한다는 한계 포함 |
| R2-m11 | 다중 stream memory | 반영 | W+NS와 버퍼·임시 메모리 제외 경계 명시 |
| R2-m12 | Data Availability/Acknowledgments | 저자 대기 | 실제 공개 URL·권리·AI 도구정보·저자 책임 확인 필요 |
| R2-m13 | σ는 마지막 단계 조건부 분산 | 반영 | 공통 parent의 마지막 8k 단계 3 seed임을 본문/결과에서 명시. full-pipeline σ나 2.5σ/4.4σ 유의성 주장 없음 |
| R2-m14 | title exact scope | 반영 | 현재 제목은 State Preservation Is Not Output Equivalence |

## 근거와 남은 우선순위

[정정 상세](../../submission/REVISION_NOTES.md), [VDA 감사](../../submission/VIDEO_BASELINE_AUDIT.md),
[scale 결과](../../submission/revision_results.tex), [재현 부록](../../submission/reproducibility_details.tex).
`work_dirs/paper_revision_r3/results.json`은 새 집계 입력/출력 해시를 기록한다.
원 final contract·선택·예측·scorer는 변경하지 않았으며 새 학습/추론은 하지 않았다.

1. 독립 인간 수학 검토 및 신규성/저널 적합성 판단: 자동 완료 불가.
2. 논문의 video-depth 실험 주장을 더 강화하려면 현재 조건의 video baseline,
   현재 가중치의 연속 장기/정성/range 진단. 기존 final은 이미 열었으므로 확장 분석은
   사후 분석으로 명시하고, 모델 개발 시 별도 untouched 평가를 마련해야 함.
3. 엣지 성능을 본문 주장으로 채택하려면 실제 영상·현재 가중치·코드·전력 로그 연결 검증.
4. 사용자 예정대로 저자 정보/선언을 채우고 공동저자 승인 및 공개 권리를 확인.

**결론: 문서 정정과 조건부 분석 원고 보강은 진행됐지만, 모든 self-revision 해결이나 투고 승인으로 간주할 수 없다.**
