# r2 리뷰 대응 상태

> 2026-09-07 정정: 아래는 당시 초안의 자체 평가다. [R3 대응표](../r3/revision-status.md)가
> 현재 제출본 기준이다. L1024의 모델/표본 혼합, metric scale 해석, VDA의 시간 문맥,
> clip bootstrap의 단위 및 동등성 해석이 다시 검토되어 “major 모두 해결”을 현재 상태로 사용하지 않는다.

[comment.md](comment.md)의 코멘트별 대응 기록. 작성 2026-08-24 16:30, 기준 커밋 `1d4d01a`.
판정은 **Minor Revision / Weak Accept**이므로 이 라운드는 새 학습이 아니라 **기존 예측의
재해석과 표현 교정**이다. 2026-08-28 갱신: major 1의 baseline \(s_t\) 재실행(2026-08-25 완료)
결과를 표 7g로 반영.

**요약**: major 2건 **모두 해결**(baseline 쪽 s_t 측정도 arms 학습이 GPU를 놓은 뒤 완주). minor 14건 중
8건 해결·1건 부분·4건 조치 불필요(이미 충족)·1건 사용자 정보 필요.

학습 작업(`work_dirs/v12a-fusenorm-s0` 등)이 GPU를 오래 점유해서, major 1의 baseline 재실행은
`work_dirs/r2-scale-after-arms.sh`로 arms suite PID를 기다렸다가 자동으로 붙었다(`r2-scale.log`
05:11~05:29, `r2-scale-rest.log` 12:52~13:10, 2026-08-25). 그 외 이번 라운드 대부분의 작업은
**모델을 다시 돌리지 않는 것**으로 한정했다 — `scripts/eval.py`와 `sokkanaem/metrics.py:report`가
이미 클립별 값을 JSON으로 남기기 때문이다.

---

## 1. Major comments

### 1. "드리프트 대부분은 정합 창"이라는 해석이 증거보다 강하다 — 해결

리뷰어 지적이 맞다. 프레임별 정합은 **정의상** 프레임별 전역 배율 오차를 제거하므로, 36% 대 1.1%는
"프로토콜 인공물"이 아니라 "전역 배율이 흔들린다"는 뜻이다. 그리고 metric 배포에서는 그 배율 안정성
자체가 모델의 시간적 품질이다.

**표현 교정(양쪽 원고)**: "almost all … is the protocol" → "most of the clip-level penalty disappears
under per-frame alignment … temporally varying global scale rather than local depth-shape drift".
초록·§5.8·한계 8번까지 같은 어법으로 맞췄다.

**요구된 분해 측정**: 리뷰어가 요청한 세 통계를 공용 채점기 한곳에 넣었다
(`sokkanaem/metrics.py:scale_stats`) — \(s_t\)의 변동계수, \(\log s_t\)의 표준편차,
\(\left|\log s_t - \log s_{t-1}\right|\)의 평균. `clip_scores`가 반환하므로 `scripts/eval.py`와
baseline 스크립트 4종이 **추가 코드 없이** 같은 값을 남긴다. `scripts/eval.py`에 중복 구현돼 있던
드리프트 계산은 삭제했다(같은 값을 두 곳에서 계산하던 것).

기존 dump로 이미 답할 수 있는 부분은 원고에 들어갔다(**표 7f**, 보고 체크포인트):

| 클립 길이 | 클립 수 | \(s_t\) 변동계수 | 95% CI |
|---|---:|---:|---|
| 8프레임 | 189 | 0.0165 | [0.0128, 0.0211] |
| 32프레임 | 120 | 0.0411 | [0.0324, 0.0515] |
| 256프레임 | 13 | 0.1127 | [0.0750, 0.1545] |

구간이 겹치지 않는다. 우리 전역 배율은 긴 클립에서 7배 덜 안정적이고, **이것은 우리 약점이지
프로토콜 탓이 아니다**라고 원고에 명시했다.

**baseline 재실행 완료.** `work_dirs/r2-scale.sh`(DA-v1-Small·DA-v2-Base·DPT-Large·ZoeDepth-NK, 자체
env)와 `work_dirs/r2-scale-rest.sh`(DA-v2-Small·DA3·VDA, 별도 conda env 3개)가 arms suite 종료 직후
`work_dirs/r2-scale-after-arms.sh`를 통해 자동 실행됐다(`r2-scale.log`, `r2-scale-rest.log`). 7종
baseline의 `work_dirs/baselines/*.json`이 새 `scale_drift`/`scale_logstd`/`scale_step` 필드를 포함해
갱신됐다.

`scripts/bootstrap_ci.py --ours work_dirs/v11-longclip-spread-s0/scores_L256K30.json --suffix l256
--metric scale_logstd --metric scale_step --metric scale_drift`로 짝지은 부트스트랩을 냈다. 결과가
**표 7g**로 양쪽 원고에 들어갔다: relative-depth 3종(DA v1/v2, DPT)은 우리보다 확실히 덜 안정적이지만
(비율 1.2~1.5, 넷 다 확정), metric·시간 사전 정보가 있는 3종(ZoeDepth·DA3·VDA)은 우리만큼 또는 더
안정적이다(ZoeDepth 비율 0.757, P 0.000으로 확정; DA3·VDA는 미확정, P 0.188/0.089). §5.8과 한계
8번의 "상태 없는 baseline이 가장 크게 악화한다"는 문장을 "그룹 전체가 아니라 셋에 대해서만 그렇다"로
좁혔다. 프레임 간 단계(step) 통계에서는 반전이 하나 있다 — ZoeDepth 대비 CV는 우리가 더 크지만 step은
우리가 더 작다(비율 1.314, P 1.000) — 표 7g에 함께 보고하고 본문에서 한 문단으로 해석했다.

### 2. 256프레임 main ranking의 uncertainty — 해결

새 스크립트 `scripts/bootstrap_ci.py`. 클립을 10,000회 재표집하고, 소스별로 층화해 표가 쓰는
데이터셋 균형 평균을 재현하며, **모델 간 짝지음**을 쓴다(모든 모델을 같은 재표집 클립에서 채점).
클립 간 변동이 모델 간 격차보다 훨씬 크기 때문에 짝짓지 않으면 구간이 무의미하게 넓다. 모델 재실행
없음. 결과는 **표 3d**로 원고에 들어갔다:

| 프로토콜 | Baseline | t-delta 비율 | 95% CI | P(우리가 더 낮다) |
|---|---|---:|---|---:|
| 256프레임 | DA3 Base | 1.239 | [1.183, 1.305] | 1.000 |
| 256프레임 | Video Depth Anything S | 1.234 | [1.184, 1.290] | 1.000 |
| 256프레임 | ZoeDepth N-K | 1.259 | [1.151, 1.367] | 1.000 |
| 8프레임 | DA3 Base | 1.101 | [0.951, 1.283] | 0.905 |
| 8프레임 | Video Depth Anything S | 1.106 | [0.947, 1.292] | 0.904 |
| 8프레임 | ZoeDepth N-K | 1.156 | [0.998, 1.347] | 0.973 |

**헤드라인 주장은 살았고, 부수 주장 하나는 죽었다.** 256프레임 1.23배는 13클립에서도 분해된다
(구간이 1.18 위, 재표집 10,000회 전부에서 우리가 낮다). 반면 **8프레임 1.10배는 189클립에서도 1과
분리되지 않는다.** 초록과 §5.3·한계 9번에서 8프레임 우위를 확립된 것으로 서술하지 않도록 고쳤다.

전체 지표(AbsRel·δ1·OPW·TCE, 두 프로토콜)는 `work_dirs/r2-bootstrap.log`. 부수 확인:
- DPT-Large 대비 AbsRel 비율 0.992, CI [0.884, 1.083]은 차이를 확립하지 못한 결과다.
  동등성 허용폭/검정이 없으므로 동등함을 확인한 것이 아니다(2026-09-07 해석 정정).
- 8프레임 TCE의 DA-v1-Small·VDA 비교 역시 우위를 확립하지 못했다. 재표집 방향 비율
  0.32/0.22를 동등성 검정이나 귀무가설 p값으로 해석하지 않는다.

---

## 2. Minor comments

| # | 코멘트 | 상태 | 비고 |
|---|---|---|---|
| 1 | "clip-level score falls 36%" 방향 혼동 | 해결 | 초록 "clip-level error worsens by 36%"(양쪽 원고) |
| 2 | 초록이 길다 | 해결 | 463→360단어(영문). Raspberry Pi 수치와 고정 비용 2.1/2.4%, 8프레임 폴백의 16.1%/δ1 반 포인트 부연을 삭제(본문 §6.3·표 9에 그대로 있음). 핵심 결과 3개(활성률-정확도 트레이드오프, 비교군 순위+부트스트랩, 프로토콜/클립 길이 발견)는 전부 유지, CI·미확정 표현도 그대로 유지 |
| 3 | "22-fold cut in update rate"를 compute와 동일시 말 것 | 조치 불필요 | Figure 3 캡션과 결론 모두 이미 "update rate". 변경 없음 |
| 4 | §5.3 회귀 정규화 표현 강함 | 해결 | "우리 정확도에서의 모델 간 회귀 추세 대비 — 인과 증거가 아니라 상관 기준선" 문장 안에 삽입 |
| 5 | DPT-Large 동률 쌍을 먼저 강조 | 조치 불필요 | 해당 문단이 이미 그 쌍을 앞세우고 회귀를 보조로 둔다. 부트스트랩이 동률을 확인해 근거만 강화 |
| 6 | 권장 운용 설정 정리 | 해결 | **표 8b** 신설(정확도/플리커/최소 연산 3행). 최소 연산 행은 256프레임 실측값 18.0% 활성·AbsRel 0.1924 |
| 7 | dense fallback을 표 9와 더 눈에 띄게 연결 | 해결 | 초록의 16.1% 문구에 "표 9" 지시, 표 8b 아래 문단에서 재연결 |
| 8 | "one drive is enough" 표현 강함 | 해결 | "이 split에서는 drive 하나로 충분했다" + 다른 drive를 보정 소스로 쓴 경우는 미시험이라고 명시 |
| 9 | 엣지 표 제목에 reference scan 강조 | 해결 | 표 13a·13b 캡션에 *참조 스캔*(배포 커널 아님) 강조 |
| 10 | 재현성 부록 | 조치 불필요 | 리뷰어가 해결로 판단 |
| 11 | 메모리 \(W + N S\) | 조치 불필요 | 리뷰어가 해결로 판단 |
| 12 | Data Availability·Acknowledgments placeholder | **미해결** | 저자·소속·학회 양식 미정 — 사용자 정보 필요(r1 minor 15와 동일 항목) |
| 13 | 2.5σ/4.4σ가 최종 단계 조건부 분산임을 캡션에 | 해결 | 표 14 캡션에 "장클립 체크포인트 하나를 세 번 fine-tune한 최종 단계의 조건부 산포" 추가 |
| 14 | 제목 | 조치 불필요 | 리뷰어가 해결로 판단 |

---

## 3. 코드 변경

| 파일 | 변경 |
|---|---|
| `sokkanaem/metrics.py` | `scale_stats()` 신설 — \(s_t\)의 CV·log-std·프레임 간 step. `clip_scores`가 반환하므로 eval과 baseline 4종이 같이 얻는다. `__main__`에 배율 불변성 자기검사 |
| `scripts/eval.py` | 중복 드리프트 계산 삭제(공용 채점기로 이관), 새 두 열을 헤더·per-clip dump·집계에 추가 |
| `scripts/bootstrap_ci.py` | 신설. 층화·짝지은 클립 단위 부트스트랩. 모델 재실행 없이 기존 dump만 읽는다 |
| `work_dirs/r2-scale.sh` | 신설. major 1의 baseline 재측정(자체 env 4종) |
| `work_dirs/r2-scale-after-arms.sh` | 신설. arms suite 종료를 기다렸다가 `r2-scale.sh` 자동 실행 |
| `work_dirs/r2-scale-rest.sh` | 신설. 별도 conda env가 필요한 나머지 3종(DA-v2-Small·DA3·VDA) |
| `work_dirs/baselines/*.json` | 7종 baseline 28개 파일이 `scale_drift`/`scale_logstd`/`scale_step` 포함해 갱신 |
| `work_dirs/r2-scale-bootstrap.log` | 신설. 표 7g 산출에 쓴 `bootstrap_ci.py` 실행 기록(표 검사 출처) |
| `paper/draft.md`, `paper/draft_ko.md` | 표 3d·7f·7g·8b 신설, major 1·2 어법 완화, minor 1·4·6·7·8·9·13 |

검사: `python sokkanaem/metrics.py`(자기검사 통과), `python scripts/table_check.py paper/draft.md`
(단일 run 출처 없는 행 0건 유지).

---

## 4. 남은 작업

| 항목 | 차단 요인 |
|---|---|
| Data Availability·Funding·Acknowledgments, 저자·소속, 학회 양식 | 사용자 정보 필요 |
| NVDS baseline | 리뷰어가 acceptance 차단 요인 아니라고 명시. 범위 판단 필요 |
| `[CHECKPOINT-DEPENDENT]` 태그 제거 | 제출본 분기에서 일괄 |

major 1의 baseline \(s_t\) 통계는 해결됐다(표 7g). 남은 항목은 전부 실험이 아니라 저자 판단·외부
정보가 필요한 것들이다.

측정 로그: `work_dirs/r2-bootstrap.log`(부트스트랩 전 지표·두 프로토콜).
