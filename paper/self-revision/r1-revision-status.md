# r1 리뷰 대응 상태

[r1.md](r1.md)의 코멘트별 대응 기록. 기준 시각 2026-08-21 01:20, 기준 커밋 `db77665`.

**요약**: major 9건 중 7건 해결·1건 진행 중(3 seed 학습)·1건 부분(엣지 기기 없음). minor 15건 중
12건 해결·2건 부분·1건 사용자 정보 대기.

이 라운드에서 **평가 프로토콜 버그를 발견해 2026-08-20 이전 모든 수치가 바뀌었다**(아래 §3). 리뷰
코멘트 대응과 그 재측정이 한 라운드에 겹쳐 있으므로, 수치를 인용할 때는 커밋 `5b3bd37` 이후인지
확인해야 한다.

---

## 1. Major comments

### 1. main 프로토콜이 streaming 주장과 불일치 — 해결

- 256프레임을 주 표로 승격(Table 3a), 8프레임은 문헌 비교용 보조 표(Table 3b).
- 클립 길이 계단 8/32/128/256/512 측정(Table 7a). 벌점: base +128%, 장클립 +78% (8→512).
- 512·1024프레임 연속 스트림 측정. 1024는 PointOdyssey 20클립 + Bonn `static_close_far` 1클립.
- frame-index 곡선을 AbsRel뿐 아니라 δ1·t-delta·OPW·TCE로 확장(Table 7b).
- state reset 규칙(클립 경계에서만 초기화), 클립 배치, 정합 단위를 §4.2에 명시.

**부수 발견**: 클립 단위 벌점의 대부분은 드리프트가 아니라 **클립당 정합 창**이다. 1024프레임에서
장클립 체크포인트의 클립 단위 오차는 8프레임 대비 +36%인데, 프레임별 독립 정합 오차는 첫 프레임에서
천 번째 프레임까지 +1.1%다(base는 +13.5%). 상태 없는 프레임 단위 baseline은 같은 클립에서 더 크게
악화한다(DPT-Large +116%, DA V2 Base +145%). Limitations의 "270프레임 너머는 더 나빠질 것"이라는
예상은 측정으로 대체되고 틀렸다.

관련 커밋: `ffc1c03`, `99b8023`, `a75841c`, `1e9b9eb`(512프레임 OOM 수정).

### 2. 개선 모델을 최종 모델로 안 씀 — 해결

보고 체크포인트를 **Final**(base 60k → 장클립 25k → spread 8k, 키프레임 주기 30)로 승격.

| 프로토콜 | base | 장클립 | **Final** |
|---|---|---|---|
| 8프레임 AbsRel | 0.1302 | 0.1302 | **0.1263** |
| 8프레임 t-delta | 0.0607 | 0.0702 | 0.0750 |
| 256프레임 AbsRel | 0.2434 | 0.1990 | **0.1907** |
| 256프레임 δ1 | 0.7134 | 0.7864 | **0.7922** |
| Bonn 범위비 | 0.75 | 0.78 | **0.85** |

승격 대가를 함께 보고한다 — spread 항이 예측 필드를 넓혀 8프레임 원시 프레임 차이가 24% 악화하고,
우리가 1위인 지표의 격차가 1.36배에서 1.10배로 좁아진다(256프레임은 1.23배). 플리커를 더 중시하는
배치용으로 장클립+주기 60 설정도 Table 3·8에 남겼다.

또한 **주기 선택이 체크포인트에 종속된다**는 것을 발견했다. 장클립에서는 주기 60이 공짜 안정성이지만
Final에서는 spread 항이 이미 그 예산을 써서 주기 30이 정확도·δ1 모두 우세하다.

관련 커밋: `5749468`, `55c28e3`, `db77665`.

### 3. video-specific baseline 부재 — 해결

- Video Depth Anything(metric, Small, 28.4M) 체크아웃 복구 후 현 프로토콜로 재측정. 비교군 7종.
- Depth Anything 3도 체크아웃 소실 상태였고 복구했다. 비인과(클립 전체 동시 처리)임을 표기.
- 결과: 8프레임 t-delta 0.0750(우리) 대 VDA 0.0829·DA3 0.0825, 256프레임 0.0692 대 0.0854·0.0857.
  **영상 전용 baseline을 넣어도 안정성 우위는 유지된다.**
- NVDS는 여전히 없다.

체크아웃 위치: `~/checkouts/Video-Depth-Anything`, `~/checkouts/Depth-Anything-3`. 가중치는 HF 캐시.

### 4. temporal 주장 축소 — 해결

title·abstract·intro·conclusion에서 raw t-delta / flicker / temporal consistency를 분리했다.
"일반적 시간 일관성 우위를 주장하지 않는다"를 명시하고, OPW·TCE에서 DA3가 앞선다는 사실을 유지한다.
frame-index 측정으로 **드리프트가 정확도에 있고 모션 기준 일관성에는 없다**는 더 정확한 서술도
추가했다(Bonn OPW 프레임 4의 0.0229 대 프레임 28의 0.0228).

### 5. efficiency 주장 분리 — 부분

- 4축 채점표 추가(§5.7): 파라미터·MAC 절감(확립), 동급 대비 지연(확립, 단 모델 규모와 커널의 공),
  스트림당 상태(확립), **희소성에 의한 속도 향상(미확립)**, 에너지(미측정), 엣지(미측정).
- "2.2배 빠르다"가 희소 경로의 공으로 오독되지 않도록 문장 재작성.
- **미해결**: 보드 전력 샘플링은 `scripts/bench.py --power`로 구현했으나 GPU가 학습에 점유돼 아직
  측정하지 않았다. Jetson급 기기는 없어 엣지 측정은 불가.

### 6. novelty 정밀화 — 해결

§2.4 신설. Skip RNN(Campos et al., 2018), spiking SSM(Tang et al., 2026), 동시기 event-gated
video generation(Maduabuchi & Wang, ECCV 2026) 대비 세 가지 차별점을 명시하고, 기여를 조합으로
재정의했다 — 외부·비학습 게이트, 억제가 아닌 항등, 조밀 출력. 서론에도 같은 취지를 반영.

세 인용은 arXiv 기록과 대조해 확인했다.

### 7. 캘리브레이션 프로토콜 불명 — 해결

기존 스윕은 보고하는 것과 같은 클립에서 임계값을 골랐다(oracle). drive 0002에서 보정하고 미관측
4 drive에서 보고하는 split을 측정했다(Table 6b).

- 보정에 **정답 불필요** — 목표가 활성률이고 검출기가 영상만으로 계산한다.
- **교환비는 전이되지만 동작점은 전이되지 않는다** — 보정 drive 활성 5.3%가 미관측에서 11.5%.
- 미관측 drive에서 전체 연산 → 11.5% 활성 대가는 상대 AbsRel 5.2%, t-delta 2.5배 개선.
- 픽셀 게이팅은 같은 split에서 최선 77.3% 활성 — GMC 우위는 split을 통과한다.
- GMC 항등 fallback 0/1,260 프레임.

관련 커밋: `654f54c`.

### 8. 통계 신뢰도(multi-seed) — 진행 중

최종 단계(clip 24, 8k, spread 0.5)를 seed 0/1/2로 반복 중. seed 0 완료, seed 1 학습 중, seed 2 대기.
완료 시 `scripts/make_tables.py --seeds`로 태그별 mean±std를 Table 8과 부록 A에 부착하고, 한계 12번을
실측 CI로 교체한다. spread 항의 3~4% 정확도 이득이 CI 밖인지도 그때 판정한다.

8k 스텝 6회 실행으로 얻은 기존 노이즈 하한(실촬 AbsRel ±0.005, δ1 ±0.004, 합성 δ1 ±0.015)은 그
아래 차이를 주장하지 않는 기준으로 계속 쓰고 있다.

### 9. alignment 공정성 — 해결

전 모델을 고유 규칙과 비고유 규칙 양쪽으로 측정했다(Table 12).

| 모델 | 고유 규칙 | 비고유 규칙 |
|---|---|---|
| DA V1 Small | 0.0650 (2-DOF) | 1.0620 (1-DOF) |
| DPT-Large | 0.0875 | 0.8962 |
| ZoeDepth N-K | 0.0992 (1-DOF) | 0.3782 (2-DOF) |
| DA 3 Base | 0.1130 (2-DOF, depth) | **0.1023** (1-DOF) |
| SOKKANAEM | 0.1263 (1-DOF) | 0.1155 (2-DOF) |

**공통 단일 규칙은 더 공정하지 않고 대부분에게 무의미하다.** DA3는 중앙값 규칙을 선호하는 유일한
모델이라 더 나쁜 2-DOF 수치로 인용한다 — baseline을 불리하게 인용하면 우리에게 유리해지므로.

관련 커밋: `4530208`.

---

## 2. Minor comments

| # | 코멘트 | 상태 | 비고 |
|---|---|---|---|
| 1 | draft note 제거 | 부분 | `[UNDER TEST]` 실사용 0건. `[CHECKPOINT-DEPENDENT]` 8개는 제출본 분기에서 제거 |
| 2 | 초록 압축 | 해결 | 471 → 295단어 |
| 3 | "exact" 범위 한정 | 해결 | 서론에 범위 조건 단락, 전 절 일관 |
| 4 | Figure 1 활성률 조건 | 해결 | 그림 안에 "at 15.4% activity" 표기 |
| 5 | token-drop 표현 | 해결 | "안정성을 재현하지 못한다"로 교체, 재측정으로 정확도 동률 확인(0.4317 대 0.4329) |
| 6 | t-delta 단위·정규화 | 해결 | 부록 A. 과거 정합 순서 불일치까지 명시 |
| 7 | RAFT 전처리·가림 | 해결 | 부록 A. 전후방 일관성 검사 없음을 명시 |
| 8 | GMC 실패율 | 해결 | 210프레임 0회, split 1,260프레임 0회 |
| 9 | dense fallback 민감도 | 해결 | 스윕 결과 기존 정확도 이득 소멸. 끄면 활성률 16.1%에 동일 정확도 |
| 10 | 메모리 분리·스케일링 | 해결 | 부록 A, \(W + N \times S\) |
| 11 | 6×~82× 표현 | 해결 | params·MAC·latency 축 분리 |
| 12 | 정성 예시 | 부분 | 키프레임 전후 패널(Figure 8, `figures/sawtooth.png`). 긴 스트림 filmstrip은 활성률 100%(dense 폴백)로 게이팅을 못 보여줘 폐기 |
| 13 | range compression 승격 | 해결 | §5.9 신설(구조적 상한 + 범위 압축) |
| 14 | 재현성 부록 | 해결 | optimizer·LR·schedule·batch·augmentation·sampling·EMA·3단계 학습·지표 정의 |
| 15 | placeholder 정리 | **미해결** | Data Availability·Funding·Acknowledgments 절이 아예 없다. 저자·소속·학회 양식 미정 — 사용자 정보 필요 |

---

## 3. 리뷰어가 지적하지 않았으나 이 라운드에서 발견한 것

### 클립 상한이 holdout이 아니라 첫 시퀀스를 표본 (`5b3bd37`)

클립이 시퀀스 순서로 concat되므로 `--max-clips 100`은 Bonn 399클립 중 `crowd2`만, PointOdyssey
1,984클립 중 첫 시퀀스만 평가했다. 같은 체크포인트가 상한 60에서 활성 55.8%·AbsRel 0.2239,
상한 100에서 44.7%·0.1914로 읽혔다 — **모델보다 상한이 수치를 더 움직였다.**

`sokkanaem.data.even_subset`으로 균등 추출하도록 `scripts/eval.py`, baseline 3종,
`frame_index_probe.py`, `ceiling_probe.py`, `range_probe.py`를 수정했다.

영향:

| 항목 | 수정 전 | 수정 후 |
|---|---|---|
| 실촬 AbsRel / 활성률 | 0.1595 / 32.2% | 0.1302 / 22.0% (base) |
| 희소성 교환 | 13배에 13% | 22배에 6.4% |
| Bonn 범위비 | 0.47 | 0.75 |
| Bonn 구조적 상한 | 0.0651 (3배 위) | 0.0367 (3.5배 위) |
| dense 폴백 정확도 이득 | 0.1685 → 0.1633 | 없음 |
| frame-index Bonn 악화 | +61% | +43% |
| spread 항 정확도 이득 | −5.5% | 노이즈 안 (범위비 회복은 유효) |

### Table 11이 두 체크포인트 혼합 (`4f359c0`)

AbsRel은 v9, δ1·TCE는 v10에서 가져와 있었다. 수정하면 fine-tuned+주기 60이 δ1 0.4포인트 하락이
아니라 **0.8포인트 상승**으로, 원고가 자기 결과를 깎아 쓰고 있었다. 원인은 `eval.txt` 헤더에
keyframe·clip_len·tag가 없어 행 추적이 불가능했던 것 — 헤더에 기록을 추가하고
`scripts/table_check.py`로 "각 표 행이 단일 run에서 나오는지" 검증한다(현재 미귀속 0건).

### DA3 정합 규칙 오표기 (`ffbae60`)

Table 3이 DA3를 1-DOF로 표기했으나 스크립트는 depth 공간 최소제곱 scale+shift(2-DOF)를 적용했다.
DA3는 OPW·TCE에서 우리를 앞서는 모델이므로 **우리에게 불리한 방향의 오표기**였다. `ALIGN` 환경변수로
양쪽을 측정하도록 수정.

### 온도 프로브가 임시 패치라 로그 없음 (`5749468`)

softmax 온도 실험은 재현 경로가 없었다. `decoder.bin_temp` + `eval.py --bin-temp`로 노브를 만들고
전체 holdout에서 재측정했다(결론 동일: 범위비 0.75 → 0.74, 정확도 소폭 악화).

---

## 4. 남은 작업

| 항목 | 차단 요인 |
|---|---|
| 3 seed CI 부착, 한계 12번 교체 | seed 1·2 학습 중(2026-08-21 오전 완료 예상) |
| 데스크톱 GPU 에너지/프레임 측정 | GPU가 학습에 점유됨. `bench.py --power` 준비됨 |
| Raspberry Pi 4 · Jetson Nano B01 측정 | 실행 문서 [EDGE_BENCH.md](../../EDGE_BENCH.md), 스크립트 `scripts/edge_bench.py` 준비됨. 데스크톱 CPU 4스레드 기준 수치 확보(활성 5%에서 15.1배, 30%에서 2.9배, dense는 활성률에 평평) |
| Orin급 엣지 측정 | 하드웨어 없음. Nano B01은 sm_53이라 융합 커널 측정 불가 |
| Data Availability·Funding·Acknowledgments, 저자·소속, 학회 양식 | 사용자 정보 필요 |
| NVDS baseline | 범위 판단 필요 |
| `[CHECKPOINT-DEPENDENT]` 태그 제거 | 제출본 분기에서 일괄 |

## 5. 검증 방법

```bash
# 원고의 모든 표 행이 단일 측정 run에서 나오는지 (목표: 0건)
python scripts/table_check.py paper/draft.md

# 로그에서 비교 표 행 재생성 (손으로 옮기지 않는다)
python scripts/make_tables.py work_dirs/r1-round.log --tag L256

# seed 집계
python scripts/make_tables.py --seeds work_dirs/v11-longclip-spread-s*/eval.txt
```

측정 로그: `work_dirs/r1-round.log`(비교군·프로토콜·정합), `work_dirs/r1-followup2.log`(합성 재측정·
seed 학습), `work_dirs/r1-calib-split.log`, `work_dirs/r1-longstream.log`, `work_dirs/r1-ls1024.log`,
`work_dirs/r1-kfsweep.log`, `work_dirs/r1-minor.log`, `work_dirs/r1-bintemp.log`.
