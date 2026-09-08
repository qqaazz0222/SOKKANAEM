# 최종 평가와 이론–영상 진단

2026-09-07. 모델 선택 후 조건을 봉인하고 평가했다. 새 학습이나 최종 결과에 따른 튜닝은 없다.

| 모델 | L256 AbsRel | L8 AbsRel | L256 실패 클립 |
|---|---:|---:|---:|
| Sparse K30 | 0.1869 | 0.1840 | 0/13 |
| Sparse K5 | 0.1828 | 0.1837 | 0/13 |
| Dense carry | 0.1837 | 0.1946 | 0/13 |
| Dense reset | 0.1909 | 0.1669 | 0/13 |
| Output hold | 0.1955 | 0.1677 | 0/13 |
| DA2 Small 252 | 0.2166 | 0.2074 | 2/13 |
| MiDaS Small 256 | 0.1849 | 0.1539 | 0/13 |
| DPT-Large 256 | 0.1894 | 0.1810 | 0/13 |

Clip scale–shift 보정 후 TUM pixel-pooled 값이다. 낮을수록 좋으며 보정 없는 metric 정확도가 아니다. L256 13클립(3328프레임), L8 462클립(3696프레임)은 같은 영상에서 추출되어 서로 독립이 아니다. 새로운 시퀀스지만 같은 촬영 환경이며, 장면 분리 일반화를 검증하지 않았다.

개발 진단 1024프레임에서 첫 프레임 이후 keyframe 32곳의 같은 상태 국소 결손은 0이었다. 그러나 연속 dense 실행과의 누적 state 차이는 남았다. 첫 block FP64 shadow 상한은 모든 step에서 충족했으나 전체 FP32 네트워크 정확도의 인증은 아니다.

[전수 지표](submission/tables/final_quality.csv), [시퀀스별](submission/tables/final_sequences.csv), [paired 통계](submission/tables/final_paired.csv), [영역별](submission/tables/final_regions.csv), [진단 원자료](submission/tables/development_diagnostic_frames.csv). n=4로 최소 양측 sign-flip p=.125이며 탐색적 비교다.

이 테스트는 이제 미사용 상태가 아니다. 추가 개발 시 새로운 미사용 평가 자료가 필요하다. 기존 봉인 파일은 역사 기록으로 유지하고 `final_opened.json`과 `final_complete.json`으로 개봉/완료를 기록했다.
