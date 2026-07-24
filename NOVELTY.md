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
2. **구조적 시간 일관성** — hidden state가 프레임 간 시각적 기억을 유지하므로 후처리(NVDS류) 없이 플리커 제거. 실측: 스킵률↑ → t-delta 단조 개선(IDEA.md §4.5, 0.1033→0.0762). 스킵이 부작용이 아니라 안정성의 **원인**이라는 게 일반 상식과 반대되는 지점.
3. **입력 적응적 연산량** — 연산 비용이 장면 변화율에 비례. 고정 카메라(CCTV) 환경 타깃.
4. **센서리스 이동 카메라 확장 (§3.5)** — Low-Res GMC(호모그래피 정렬) + Feature-level Gating 하이브리드. IMU 없이 순수 RGB만으로 ego-motion 환경 대응. 저해상도 정렬의 잔차 오차를 Mamba의 순차 hidden state 전파가 흡수한다는 "노이즈 강인성" 자체가 학술적 어필 포인트 (IDEA.md §3.5 기대효과).

## 검증되지 않은 novelty 주장 (주의)

- §3.5의 "노이즈 강인성 흡수"는 vkitti2(합성, 이미 clean geometry) 기준 실증이며, 실제 노이즈 있는 ego-motion 영상(블랙박스 등) 미검증 — REPORT.md 한계 참조.
- "학습-배포 분포 일치가 중요하다"는 통념은 3-arm ablation에서 **기각**됨 (iid random 마스크가 detector-driven fine-tune보다 우세) — 이 자체도 반직관적 결과로 novelty 주장에 포함 가능 (IDEA.md §4.5).
