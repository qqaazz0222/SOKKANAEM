# NOVELTY

SOKKANAEM이 기존 연구 대비 실제로 새로운 부분만 추림. 배경·실험은 [IDEA.md](IDEA.md), 진행 상황은 [PROGRESS.md](PROGRESS.md), 수치는 [REPORT.md](REPORT.md) 참조.

## 핵심 통찰 (one-liner)

Mamba의 이산화 파라미터 Δ에 변화 마스크를 곱하면 (`Δ̃ = M · Δ`), **정적 패치의 연산 스킵이 근사가 아니라 hidden state의 정확한(exact) 항등 복사가 된다.** 별도 캐시·정합성 관리 없이 "스킵 = 상태 유지"가 수식 그 자체로 성립 (IDEA.md §1.3, §3.2).

## 관련 연구 대비 차별점

| 계열 | 대표 연구 | 공유하는 것 | SOKKANAEM만 갖는 것 |
|---|---|---|---|
| 단안/비디오 깊이 추정 | MiDaS, DPT, Depth Anything, Video Depth Anything, NVDS | 깊이 추정 태스크 | 프레임당 연산량이 변화율에 비례 (이들은 고정 비용) |
| 토큰 감축 (이미지) | ToMe, EViT, DynamicViT | 토큰 단위 연산 절감 | 단일 프레임 내부가 아니라 **프레임 간** 중복 활용, dense 출력 복원 문제 없음 (마스크는 공간 위치 고정, 손실 없음) |
| 변화 기반 스킵 | Skip-Convolutions, DeltaCNN, Eventful Transformer | "변한 것만 재계산" 원칙 | 스킵된 토큰의 정보를 **별도 캐시가 아니라 SSM hidden state 자체**가 보존 — 캐시 무효화/정합성 로직 불필요 |
| Vision Mamba | Vim, VMamba, VideoMamba | SSM 백본, $O(N)$ 스캔 | 입력 적응적 연산 스킵 없음(항상 전체 토큰 스캔) → 본 연구가 최초로 Δ-gating 결합 |

**한 문장 요약:** 변화 기반 스킵(Eventful Transformer 계열)과 SSM의 상태 유지 능력을 결합한 최초 시도.

## 기여 4가지와 그 근거

1. **Δ-Gating** — 조건부 SSM 게이팅을 이산화 수식 레벨에서 구현. `M=0 → Ā=I, B̄=0 → h_t=h_{t-1}` (근사 아님, 극한값). 검증: `tests/test_gating.py` (mask=0 ⇒ bit-exact state copy).
2. **후처리 없는 플리커 제거** — hidden state가 프레임 간 시각적 기억을 유지하므로 NVDS류 후처리
   없이 프레임 간 출력이 흔들리지 않는다. 실측(1,000 클립 동일 프로토콜, REPORT §4.15):
   t-delta 0.2455 vs DA3 1.80·VDA 2.18·DA v2 9.47 (**7.3–38.6배**). 명시적 시간 모듈(VDA)이나
   8프레임 동시 처리(DA3)로도 안 되는 부분. 스킵률↑에 t-delta가 단조 개선되는 것도 재확인 —
   스킵이 부작용이 아니라 안정성의 **원인**.
   **범위 제한(중요)**: 모션 보정 지표에서는 우위가 없다 — OPW에서 DA3가, GT 기준 TCE에서는
   DA3·VDA 둘 다 우리보다 낫다(REPORT §4.15). 기여 2는 "플리커 없음"으로 좁혀 주장해야 하고
   "시간 일관성 전반의 우위"로 확대하면 즉시 반박된다.
   대조 증거: 같은 마스크로 state는 동결하되 readout만 없애면(token drop) 같은 active%에서
   AbsRel 4.0배·t-delta 29배 붕괴 — 스킵이 공짜인 이유가 "state를 계속 읽는 것"임을 분리 실증
   (REPORT §4.14).
3. **입력 적응적 연산량** — *정직한 형태*: **시간축 state 경로의 비용이 변화율에 비례하고,
   종단 절감은 dense embed/decoder와 공간축에 의해 제한된다.** "연산량 ∝ 변화율"이라는 원래
   주장은 실측 FLOPs 회계에서 기각됨 — v1~v5 구조에서 Δ-gating만으로는 active 0%에서도
   풀연산의 96.4%(REPORT.md §4.11). 이유: 디코더가 MAC의 67.8%(§3.3의 자체 "백본 10% 이하"
   예산을 22배 위반)이고, Δ-gating은 static 토큰의 state 갱신만 없애고 readout(58.5%)은
   남긴다. v6(경량 디코더 + 학습된 sparse spatial 경로)에서 active 16.6% 시 38.6%로 개선.
   고정 카메라 active% 자체는 실촬로 입증됨(TUM static, $\tau$=0.05에서 5.9%, §4.12).
4. **센서리스 이동 카메라 확장 (§3.5)** — Low-Res GMC(호모그래피 정렬) + Feature-level Gating 하이브리드. IMU 없이 순수 RGB만으로 ego-motion 환경 대응. 저해상도 정렬의 잔차 오차를 Mamba의 순차 hidden state 전파가 흡수한다는 "노이즈 강인성" 자체가 학술적 어필 포인트 (IDEA.md §3.5 기대효과).

## 검증되지 않은 novelty 주장 (주의)

- 기여 2의 근거였던 t-delta는 **상수 출력이 전역 최적**인 퇴화 지표다(§4.6 붕괴가 0.0000 기록).
  flow warping(OPW)도 상수 필드엔 무력. 퇴화하지 않는 TCE와 const 제어행을 함께 도입했고
  (REPORT.md §4.10), 실촬 고정카메라에서는 **상수가 TCE에서도 모델을 이겼다**(§4.12) —
  "스킵이 시간 안정성의 원인" 주장은 외부 baseline 대비로는 유효하나, 절대적 의미의 시간
  일관성 우위로 확대 해석하면 안 된다.

- §3.5의 "노이즈 강인성 흡수"는 vkitti2(합성, 이미 clean geometry) 기준 실증이며, 실제 노이즈 있는 ego-motion 영상(블랙박스 등) 미검증 — REPORT.md 한계 참조.
- "학습-배포 분포 일치가 중요하다"는 통념은 3-arm ablation에서 **기각**됨 (iid random 마스크가 detector-driven fine-tune보다 우세) — 이 자체도 반직관적 결과로 novelty 주장에 포함 가능 (IDEA.md §4.5).
