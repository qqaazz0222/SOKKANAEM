# PROGRESS

작업 상태 추적용. 실험 수치는 [REPORT.md](REPORT.md), 아이디어 배경은 [IDEA.md](IDEA.md) 참조.
로드맵 4단계는 IDEA.md §7 기준.

## 로드맵 상태

| 단계 | 상태 | 비고 |
|---|---|---|
| 1. PoC (4주) | ✅ 완료 (Go) | vkitti2, 스킵 55%에서 AbsRel 열화 +0.15% — 기준(5% 이내) 크게 상회 |
| 2. 본 학습 (8주) | 🟡 진행 중 | `work_dirs/main` 100k step 중 6,000 step에서 재개, tmux 세션 `main`에서 실행 중 |
| 3. 시스템 (4주) | ⬜ 미착수 | Triton 블록 희소 커널, Jetson Orin 실측, 데모 |
| 4. 논문화 | ⬜ 미착수 | CVPR/ICCV(efficiency) 또는 CoRL/IROS(실시간 시스템) |

## 현재 상태 (2026-07-24 기준)

- **PC 재설치로 conda 환경 소실 → 재구축 완료**: `conda env create -f environment.yml` + `uv pip install -e ".[dev,video]"`. pytest 14개 전부 통과.
  - **주의: `torch>=2.3` 무제한 상한 탓에 2.13.0+cu130이 잡힘 — `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`로 conv2d 자체가 안 됨.** `torch==2.8.0`(cu128, cudnn 9.10)으로 다운그레이드해 해결, 재발 방지로 `pyproject.toml`에 `torch<2.9` 상한 박음.
  - `work_dirs/*` 일부가 root 소유라 hyunsu 계정으로 못 쓰던 문제 있었음 — 사용자가 별도 터미널에서 `chown` 완료.
- **체크포인트 resume 기능 검증 후 사용**: `scripts/train.py`/`sokkanaem/model.py`의 `{model, optim, step}` 딕셔너리 저장 + `--resume` 복원 로직(이전 세션에서 미커밋 상태로 남아있던 것)을 합성 데이터로 스모크 테스트해 정상 동작 확인.
- **데이터셋 경로 버그 수정 (2026-07-24, 이전 세션)**: `configs/main.toml`, `configs/vkitti2.toml`가 존재하지 않는 로컬 경로를 가리키던 것을 `/archive/Dataset_SOKKANAEM` 하위로 통일.
- **본 학습 재개함**: `work_dirs/main/latest.pt`(체크포인트 저장 시점 step 6000 — train.log는 7500까지 찍혀 있었지만 2000-step 주기 저장이라 마지막 스냅샷은 6000)에서 `--config configs/main.toml --resume work_dirs/main/latest.pt`로 재개. tmux 세션 `main`에서 실행 중, `mixed dataset: 160973 clips from 3 sources` 확인(경로 수정 덕에 vkitti2도 정상 포함). 진행 상황은 `work_dirs/main/train.log` / `console.log`.

## 완료된 마일스톤 (날짜순)

- **2026-07-06** — Δ-gated Mamba PoC 초기 구현 (`ab6014b`)
- **2026-07-07** — 이동 카메라 확장(GMC+feature gating) 설계 및 구현, vkitti2 PoC 결과 확보(Go 판정), wall-clock 베이스라인, 청크 segment-sum 스캔(3.6× FPS), static-patch 캐싱(phase 3, opt-in)
- **2026-07-08** — CUDA graph 캡처(2.4–2.9×), detector-driven 마스크 학습 플래그, eval-only ablation(keyframe 주기, MSE vs cosine), 본 학습 인프라(주기적 ckpt, holdout split, aspect crop), 256px OOM 수정(gradient checkpoint), 장기 스트림 드리프트/스트리밍 active% 조사, 마스크 분포 3-arm ablation(iid random @ 0.5 확정)
- **2026-07-08** — 본 학습 데이터셋 다운로드 스크립트 작성 (`/archive`로 VDA 믹스)
- **2026-07-09** — 스톨 워치독 추가, tartanair 다운로더를 `hf_hub` → `wget -c`로 교체 (다운로드 중 재시작 버그 실측·수정)
- **2026-07-10** — tartanair2 + pointodyssey 어댑터, `configs/main.toml` 작성 → 본 학습 착수
- **(런타임, 로그 기준) ~2026-07-14** — 본 학습 7500 step까지 진행 후 중단
- **2026-07-24** — 데이터셋 경로 버그 수정, README에 데이터 위치 명시, checkpoint resume 기능 작업 중

## 다음 액션

1. 체크포인트 resume 변경(`scripts/train.py`, `sokkanaem/model.py`) 커밋 (검증 완료, 아직 미커밋)
2. 100k step 완주까지 모니터링 (holdout: vkitti2 Scene06, tartanair OldTownFall, pointodyssey val/test)
3. 완주 후 `scripts/eval.py`로 홀드아웃 전체 벤치마크 (§4.2 지표: AbsRel/RMSE/δ1 + TAE/OPW + FPS)
4. 3단계(시스템) 착수: Triton 희소 커널, Jetson Orin 실측
