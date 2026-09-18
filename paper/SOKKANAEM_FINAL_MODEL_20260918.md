# 최종 모델 확정 — MambaVision-T 8탭 디코더 + band 평탄면 손실 (8000 step)

2026-09-18. `SOKKANAEM_MV_DECODER_PROBE_20260915.md` 부록 M~S의 결론을 논문화용으로
정리한 요약본. 개별 실험의 전체 과정·해석은 원 문서를 참조하고, 이 문서는 최종 주장과
그 근거·한계만 압축한다.

## 1. 최종 모델 정의

- **백본:** MambaVision-T, DA2(Depth Anything V2)에서 592px로 증류한 stem+stage0-2
  (stage3는 로드는 되지만 forward에 안 씀), 2단 공동 미세조정(백본+디코더) 완료.
- **디코더:** `DPTDecoder`, `set_n_blocks(8)` — stage2의 Mamba 믹서 4블록 + attention
  4블록, 총 8개 블록 전부의 토큰을 탭으로 사용(기존 배포판은 절반인 4탭만 씀).
- **손실(2단 공동 미세조정):** `boundary_weight=0`, `overshoot_weight=0.5`,
  `flat_weight=22`, `flat_mode=band`, `flat_win=5`, `flat_win_hi=17`. weight=22는
  8탭 배치 기준으로 재측정한 값(다른 백본의 가중치를 그대로 옮기면 틀린 방향으로
  튜닝된다 — 부록 O.2).
- **스텝 예산:** 1단(디코더만) 4000 step + 2단(공동, 위 손실) **8000 step**(appendix F의
  기본 4000+4000보다 2단을 2배 늘림 — O.3이 이 조합에서 필수임을 확인).
- **파라미터:** 11.38M(stem+stage0-2 11.19M + 디코더 0.17M + 보정 헤드 0.02M).
- **코드:** `scripts/mv_decoder_probe_bigdecoder.py`(원본 `mv_decoder_probe.py`는
  read-only import만, 한 줄도 수정 안 됨 — 격리 검증됨).
- **체크포인트:**
  `work_dirs/mv_decoder_probe_bigdecoder_20260917/mambavision_da2_hr_8taps_band8k_ft/`
  (seed0, 논문 대표 arm), 병렬 시드 `..._s{1,2}_band8k_ft/`.

### 재현 명령

```bash
# stage 1: decoder-only, 8 taps, no extra loss (appendix M)
python scripts/mv_decoder_probe_bigdecoder.py train --taps 8 --steps 4000
# stage 2: joint backbone+decoder, band loss, 8000 step (appendix O.3)
python scripts/mv_decoder_probe_bigdecoder.py finetune --taps 8 --tag band8k \
  --boundary-weight 0 --overshoot-weight 0.5 --flat-weight 22 --flat-mode band \
  --flat-win 5 --flat-win-hi 17 --steps 8000
python scripts/mv_decoder_probe_bigdecoder.py calibrate --taps 8 --tag band8k --steps 4000
python scripts/mv_decoder_probe_bigdecoder.py predict   --taps 8 --tag band8k --calibrated
python scripts/mv_decoder_probe_bigdecoder.py score     --taps 8 --tag band8k
```

## 2. 정확도 — 절대값과 G1 대비 (BEHAVE 6영상 동일 가중 평균)

| 지표 | Q0 dense | 네이티브(4.19M) | **G1**(현 배포 최선, MV 592+FT+평면잔차 창5) | **최종(seed0)** | **최종(3시드 mean±std)** |
|---|---:|---:|---:|---:|---:|
| raw AbsRel | 0.1737 | 0.3617 | 0.1812 | 0.1780 | 0.2081 ± 0.0516 |
| edge AbsRel | 0.1515 | 0.3007 | 0.1880 | 0.2010 | 0.2125 ± 0.0254 |
| overshoot | 0.3017 | 0.6390 | 0.5134 | 0.4621 | 0.5618 ± 0.0709 |
| boundary F1 | 0.5411 | 0.2957 | 0.5091 | **0.5213** | **0.5410 ± 0.0196** |
| flat TV | 0.0244 | 0.0644 | 0.0411 | **0.0371** | **0.0336 ± 0.0030** |
| Q0 관문 | 6/6 | 0/6 | 0/6 | 0/6 | 0/6 |

**논문에 쓸 수 있는 정확한 주장(3시드 검증, 부록 Q 기준):**
- G1(현재 파이프라인이 선택한 최선 구성) 대비 **boundary F1과 flat TV를 안정적으로
  개선한다**(표준편차가 평균의 6–9%로 작고 3시드 모두 같은 방향).
- raw/edge/overshoot는 **개선한다고 말할 수 없다** — seed0만 보면 이겼지만 3시드
  평균은 지고, 시드 간 표준편차가 평균의 13–25%로 커서 seed0의 승리가 통계적
  우연이었다(seed2는 raw가 0.28까지 튀어 손실 없는 4탭 기준선보다도 나쁘다).
- Q0(고정 DA2, 전체 갱신) 대비로는 raw·edge·F1·Q0관문을 제외한 전 지표에서 이 모델도
  네이티브 4.19M도 아직 못 이긴다 — Q0는 여전히 정확도 상한선이다. 이 트랙의 주장은
  "G1(같은 MambaVision 계열의 최선 파이프라인 구성) 대비 부분 개선"이지 "Q0를 이긴다"가
  아니다.

**해보고 안 된 것(정직하게 기록):**
- 모델 소프(가중치 평균, 부록 R.1): F1 승리가 깨짐(0.4684) — 시드 간 선형 모드 연결성
  없음.
- 출력 앙상블(부록 R.2): F1·flatTV 승리 유지, raw/edge/overshoot 격차를 15–25%에서
  1.5–7%로 좁히지만 뒤집지는 못함. 추론 비용 3배.
- 12탭+band, 8000 step(부록 S): 8탭과 정반대 트레이드오프(overshoot·flatTV 승, F1 패) —
  탭 수가 "F1이냐 overshoot냐"의 방향을 정하는 구조적 축이라 스텝을 늘려도 안 바뀜.
- edge는 창 폭 탐색(부록 O.6), boundary_location_loss 재도입(O.7), 탭 수 확대(M/N),
  12탭+8000step(S), 세 가지 앙상블(R) 전부 시도했지만 이 트랙 전체에서 **단 한 번도
  G1을 이긴 적 없다** — 최고 근접치는 12탭+band(8k)의 1.035×(부록 S).

## 3. 효율성 — 스트리밍 부분갱신 (부록 P)

`scripts/mv_stream_engine_8taps.py`(원본 `mv_stream_engine.py`는 무수정, 8블록 전체를
태우도록 새로 짠 엔진). 8탭+band(8k) 모델의 정확도 개선이 스트리밍에서도 안 깨짐을 확인.

| 구성 | 평균 지연 | dense 대비 | 활성 비율 | raw(vs G1) | edge(vs G1) | overshoot(vs G1) | F1(vs G1) | flat TV(vs G1) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 4.455ms | — | 1.000 | 0.982 | 1.069 | 0.900 | 이김 | 0.903 |
| t12 | 4.184ms | −6.1% | 0.776 | 0.983 | 1.070 | 0.900 | 이김 | 0.904 |
| t8 | 3.894ms | −12.6% | 0.553 | 0.984 | 1.072 | 0.907 | 이김 | 0.905 |
| **t4** | **3.806ms** | **−14.6%** | 0.336 | 0.983 | 1.071 | 0.906 | 이김 | 0.906 |

(이 표는 seed0 단일 기준 — §2의 3시드 정정이 여기도 적용된다: "G1 대비 우위"는
F1·flatTV로 한정해서 읽어야 한다. 스트리밍이 그 우위를 깨지 않는다는 결론 자체는
seed0 안에서 유효.)

## 4. 메커니즘 주장 (부록 P.0) — 논문 테제의 정밀화

원래 가설 "Mamba/SSM이 안 변하는 영역의 연산을 억제해 지연을 줄인다"는 **절반만
맞는다:**

- MambaVision stage2의 **Mamba 믹서 블록은 스킵 불가능** — 크로스스캔이 전체 토큰
  시퀀스를 순서대로 결합해서, 한 토큰을 스킵하면 뒤 모든 토큰의 은닉 상태가 달라짐.
  매 프레임 무조건 dense.
- **attention 블록만** Q0 방식 K/V 캐시로 토큰(타일) 단위 부분갱신 가능.

**정밀화된 주장:** "이 하이브리드 아키텍처에서 안 변하는 영역의 연산 억제는 SSM이
아니라 attention이 담당하며, 그 절감(−14.6%, t4)은 디코더 품질 개선(8탭+band)과
직교적으로 공존한다." SSM이 스킵 가능한지는 "Mamba냐 아니냐"가 아니라 그 SSM이
도는 축 위의 원소들이 서로 결합돼 있는지에 달려 있다.

## 5. 한계 (논문에 반드시 적어야 할 것)

- BEHAVE 6개 영상, 이미 학습에도 쓰인 데이터로 채점 — 홀드아웃 일반화는 미검증.
- 정확도 3시드(F1·flatTV만 재현), 스트리밍은 seed0 1개.
- G1 자체의 시드 편차는 모른다 — G1도 흔들릴 수 있고, 그러면 "F1·flatTV를 이긴다"는
  비교도 G1 쪽 분산 없이는 완전하지 않다(부록 Q.3, N.5와 같은 제약).
- Q0 관문(자체 배포 기준) 0/6 — 이 모델이 이기는 대상은 Q0가 아니라 G1(같은
  MambaVision 계열의 이전 최선 구성)이다.
- calibrate는 3시드 검증에서 seed0 고정 — finetune 시드 편차만 격리했고, 계산
  파이프라인 전체의 재현성은 별도.

## 6. 산출물 경로 요약

- 코드: `scripts/mv_decoder_probe_bigdecoder.py`, `scripts/mv_stream_engine_8taps.py`
  (둘 다 원본 `mv_decoder_probe.py`/`mv_stream_engine.py`를 import만 하고 무수정).
- 체크포인트/점수:
  `work_dirs/mv_decoder_probe_bigdecoder_20260917/mambavision_da2_hr_8taps_band8k_ft/`
  (대표, seed0), `..._s{1,2}_band8k_ft/`(3시드), `stream_8taps_band8k/{dense,t12,t8,t4}/`
  (스트리밍).
- 전체 실험 로그: `paper/SOKKANAEM_MV_DECODER_PROBE_20260915.md` 부록 M(구조 확대)–
  S(마지막 조합), 특히 O(손실×탭 결합), P(스트리밍), Q(3시드 정정), R(앙상블 시도).
