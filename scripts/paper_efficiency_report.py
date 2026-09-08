"""Export matched-quality efficiency, pipeline costs, and sequence/seed statistics."""
import argparse
from collections import defaultdict
import html
import json
from pathlib import Path
import statistics
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.paper_study_report import csv_file, table
from scripts.paper_efficiency_study import PRIOR, rows, selected
from sokkanaem.metrics import pooled
from sokkanaem.performance_study import POINTS, paired_sequence_summary, pareto_flags
from sokkanaem.protocol import file_record

METRICS = ("absrel", "rmse", "delta1", "tce")


def aggregate_quality(records):
    groups = defaultdict(list)
    for row in records:
        for model, item in row["models"].items():
            for gauge, val in item["gauges"].items():
                if not gauge.endswith("/frame"):
                    groups[(model, gauge)].append((row, val))
    output = {}
    for key, members in groups.items():
        by_sequence, by_source = {}, {}
        for group, target in (("sequence", by_sequence), ("source", by_source)):
            for name in sorted({row[group] for row, val in members}):
                subset = [val for row, val in members if row[group] == name]
                target[name] = {**pooled([val["scores"]["_pooled"] for val in subset]),
                                "failed": sum(val["failed"] for val in subset),
                                "catastrophic": sum(val["catastrophic"] for val in subset), "n": len(subset)}
        output[key] = {"sequences": by_sequence, "sources": by_source,
            "balanced": {metric: statistics.fmean(s[metric] for s in by_source.values()) for metric in METRICS},
            "equal_sequence": {metric: statistics.fmean(s[metric] for s in by_sequence.values()) for metric in METRICS},
            "failed": sum(v["failed"] for v in by_sequence.values()),
            "catastrophic": sum(v["catastrophic"] for v in by_sequence.values()),
            "n": len(members)}
    return output


def balanced_mean(records, field):
    groups = defaultdict(list)
    for row in records:
        groups[row["source"]].append(row[field])
    return statistics.fmean(statistics.fmean(v) for v in groups.values())


def latency_summary(records):
    sequence_rows, summary, event_rows = [], {}, []
    for row in records:
        times = np.asarray([run["ms"] for run in row["runs"]])
        flat = times.ravel()
        unique_telemetry = row["runs"][0]["telemetry"]
        item = {"model": row["model"], "source": row["source"], "sequence": row["sequence"],
                "clip_id": row["clip_id"], "mean_ms": float(flat.mean()),
                "p50_ms": float(np.quantile(flat, .5)), "p95_ms": float(np.quantile(flat, .95)),
                "max_ms": float(flat.max()), "repeat_means": times.mean(1).tolist(),
                "repeat_min_ms": float(times.mean(1).min()), "repeat_max_ms": float(times.mean(1).max()),
                "peak_allocated_mib": max(r["peak_allocated_mib"] for r in row["runs"]),
                "state_mib": max(t["state_bytes"] for t in unique_telemetry) / 2**20,
                "keyframes": sum(t["keyframe"] for t in unique_telemetry),
                "dense_fallbacks": sum(t["dense_fallback"] for t in unique_telemetry),
                "gmc_calls": unique_telemetry[-1]["gmc_calls"],
                "gmc_fallbacks": unique_telemetry[-1]["gmc_fallbacks"],
                "frames": row["frames"], "repeats": len(row["runs"])}
        sequence_rows.append(item)
        categories = defaultdict(list)
        for run in row["runs"]:
            for ms, t in zip(run["ms"], run["telemetry"]):
                if POINTS[row["model"]]["kind"] != "native":
                    label = "single_frame"
                elif POINTS[row["model"]].get("dense"):
                    label = "dense"
                else:
                    label = ("keyframe" if t["keyframe"] else "dense_fallback" if t["dense_fallback"]
                             else "other_all_active" if t["active"] == 1 else "sparse")
                categories[label].append(ms)
        for event, values in categories.items():
            event_rows.append({"model": row["model"], "sequence": row["sequence"], "event": event,
                "unique_frames": len(values) // len(row["runs"]), "timed_frames_with_repeats": len(values),
                "mean_ms": statistics.fmean(values), "p95_ms": float(np.quantile(values, .95))})
    for model in sorted({r["model"] for r in sequence_rows}):
        members = [r for r in sequence_rows if r["model"] == model]
        s = {field: balanced_mean(members, field) for field in ("mean_ms", "p50_ms", "p95_ms")}
        s["equal_sequence_mean_ms"] = statistics.fmean(r["mean_ms"] for r in members)
        repeated = [balanced_mean([{**r, "repeat_mean": r["repeat_means"][i]} for r in members], "repeat_mean")
                    for i in range(members[0]["repeats"])]
        s.update(repeat_min_ms=min(repeated), repeat_max_ms=max(repeated),
                 peak_allocated_mib=max(r["peak_allocated_mib"] for r in members),
                 state_mib=max(r["state_mib"] for r in members),
                 frames=sum(r["frames"] for r in members), repeats=members[0]["repeats"])
        summary[model] = s
    return sequence_rows, summary, event_rows


def compare(quality, left, right, gauge, metric, scope):
    a, b = quality[(left, gauge)]["sequences"], quality[(right, gauge)]["sequences"]
    if set(a) != set(b):
        raise ValueError("paired comparison does not have identical sequence IDs")
    seq = sorted(a)
    result = paired_sequence_summary([a[s][metric] - b[s][metric] for s in seq])
    return {"scope": scope, "left": left, "right": right, "gauge": gauge, "metric": metric,
            "left_equal_sequence": statistics.fmean(a[s][metric] for s in seq),
            "right_equal_sequence": statistics.fmean(b[s][metric] for s in seq),
            "sequences": seq, **result}


def scatter(path, points):
    width, height, x0, y0, pw, ph = 980, 570, 70, 55, 780, 385
    xmax = max(p["mean_ms"] for p in points) * 1.1
    ymax = max(p["absrel"] for p in points) * 1.1
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<title>Development operating points: matched 1024 frames, PNG-to-CPU-depth FP32 eager</title>',
           f'<rect width="{width}" height="{height}" fill="white"/>',
           '<g font-family="Arial,sans-serif" font-size="12" fill="#222">',
           '<text x="70" y="26" font-size="17">Matched development quality and pipeline latency</text>']
    for i in range(6):
        x, y = x0 + pw * i / 5, y0 + ph * (1 - i / 5)
        out += [f'<path d="M{x},{y0}v{ph} M{x0},{y}h{pw}" fill="none" stroke="#ddd"/>',
                f'<text x="{x}" y="{y0 + ph + 20}" text-anchor="middle">{xmax * i / 5:.1f}</text>',
                f'<text x="{x0 - 8}" y="{y + 4}" text-anchor="end">{ymax * i / 5:.2f}</text>']
    labels = []
    for p in points:
        x, y = x0 + pw * p["mean_ms"] / xmax, y0 + ph * (1 - p["absrel"] / ymax)
        color = {"native": "#0077bb", "da2": "#cc3311", "midas": "#009988"}[POINTS[p["model"]]["kind"]]
        out.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{color}"><title>{html.escape(p["model"])}</title></circle>')
        # Nearby refresh/dense points should not put several labels on top of each other.
        w = 8 * len(str(p["id"]))
        candidates = [(dx, dy) for dy in (-8, 16, -24, 32, -40, 48, -56, 64)
                      for dx in (8, -w - 8, 30, -w - 30, 52, -w - 52)]
        for dx, dy in candidates:
            box = (x + dx, y + dy - 12, x + dx + w, y + dy + 2)
            overlap = any(not (box[2] + 3 < b[0] or box[0] - 3 > b[2] or box[3] + 3 < b[1] or box[1] - 3 > b[3]) for b in labels)
            if not overlap and box[0] >= x0 and box[2] < width - 20 and box[1] >= y0 and box[3] < y0 + ph:
                break
        else:
            raise ValueError("could not place a legible operating-point label")
        labels.append(box)
        out.append(f'<path d="M{x:.2f},{y:.2f}L{box[0] + w/2:.2f},{box[1] + 7:.2f}" stroke="{color}" stroke-width="0.6"/>')
        out.append(f'<text x="{box[0]:.2f}" y="{box[1] + 12:.2f}" fill="{color}">{p["id"]}</text>')
    out += ['<text x="420" y="484" text-anchor="middle">PNG-file to CPU output, mean ms/frame (lower is better)</text>',
            '<text x="20" y="240" transform="rotate(-90 20 240)" text-anchor="middle">Clip scale-shift AbsRel (lower is better)</text>',
            '<text x="70" y="515">Blue: native; red: DA2 Small; green: MiDaS Small. IDs map to the report table.</text>',
            '<text x="70" y="541">Equal-source means; same 4 x 256 frames. Different input resolutions are explicit operating points.</text>',
            '</g></svg>']
    path.write_text("\n".join(out) + "\n")


def validate(directory, contract):
    if contract["smoke"]:
        raise ValueError("smoke results cannot become paper evidence")
    for phase in ("quality", "latency", "profile", "seeds"):
        completion = json.loads((directory / f"{phase}_complete.json").read_text())
        if completion["smoke"]:
            raise ValueError("smoke completion")
        for rec in completion["results"]:
            if file_record(rec["path"])["sha256"] != rec["sha256"]:
                raise ValueError("phase result hash mismatch")
    for filename, length, names in (("quality.jsonl", 256, set(POINTS)),
                                   ("seeds_L8.jsonl", 8, {"seed1", "seed2"}),
                                   ("seeds_L256.jsonl", 256, {"seed1", "seed2"})):
        doc = json.loads(Path(f"manifests/acc_real_L{length}.json").read_text())
        declared = {s: [c for c in doc["clips"] if c["source"] == s] for s in doc["sources"]}
        expected = {f"{s}:{i}" for s, clips in declared.items() for i in range(len(clips))}
        measured = rows(directory / filename)
        if len(measured) != len(expected) or {r["id"] for r in measured} != expected:
            raise ValueError("incomplete/duplicated declared quality clips")
        for row in measured:
            if row["pairs"] != declared[row["source"]][row["index"]]["pairs"] or row["clip_len"] != length:
                raise ValueError("quality frame identity mismatch")
            if set(row["models"]) != (set() if row["no_gt"] else names):
                raise ValueError("models did not use identical scorable clips")
    expected = {(model, identity["id"]) for model in POINTS for identity, _, _ in selected()}
    quality_by_id = {r["id"]: r for r in rows(directory / "quality.jsonl")}
    prior_by_id = {r["id"]: r for r in rows(PRIOR / "long.jsonl")}
    for row in quality_by_id.values():
        for name in set(row["models"]) & set(prior_by_id[row["id"]]["models"]):
            if not row["models"][name].get("prior_score_equivalence"):
                raise ValueError("native streaming equivalence was not checked")
    for phase in ("latency", "profile"):
        measured = rows(directory / f"{phase}.jsonl")
        if len(measured) != len(expected) or {(r["model"], r["clip_id"]) for r in measured} != expected:
            raise ValueError("incomplete paired timing windows")
        repeats = 5 if phase == "latency" else 1
        if any(r["frames"] != 256 or len(r["runs"]) != repeats or not r["prediction_verified"]
               or any(len(run["ms"]) != 256 or len(run["telemetry"]) != 256
                      or not all(np.isfinite(t) and t > 0 for t in run["ms"])
                      for run in r["runs"]) for r in measured):
            raise ValueError("invalid timing repetitions")
        for row in measured:
            reference = quality_by_id[row["clip_id"]]
            model = reference["models"][row["model"]]
            if row["source"] != reference["source"] or row["sequence"] != reference["sequence"]:
                raise ValueError("timing/quality sequence identities differ")
            if any(run["prediction_sha256"] != model["prediction_sha256"]
                   or run["input_size"] != model["effective_size"] for run in row["runs"]):
                raise ValueError("timing predictions/inputs differ from quality")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("work_dirs/paper_study_9_11"))
    args = ap.parse_args()
    directory = args.out
    contract = json.loads((directory / "study.json").read_text())
    validate(directory, contract)
    quality_rows = rows(directory / "quality.jsonl")
    chosen_ids = {r[0]["id"] for r in selected()}
    matched_rows = [r for r in quality_rows if r["id"] in chosen_ids]
    quality, matched = aggregate_quality(quality_rows), aggregate_quality(matched_rows)
    latency_rows, times, events = latency_summary(rows(directory / "latency.jsonl"))
    tables = Path("paper/tables")
    tables.mkdir(exist_ok=True)
    accuracy = []
    for scope, scores in (("L256_all_13", quality), ("matched_timing_4x256", matched)):
        for (model, gauge), item in scores.items():
            for scheme in ("balanced", "equal_sequence"):
                accuracy.append({"scope": scope, "model": model, "gauge": gauge, "aggregation": scheme,
                    **item[scheme], "failed": item["failed"], "catastrophic": item["catastrophic"], "n": item["n"]})
    csv_file(tables / "study9_quality.csv", ["scope", "model", "gauge", "aggregation", *METRICS,
             "failed", "catastrophic", "n"], accuracy)
    operating = []
    for i, model in enumerate(POINTS, 1):
        item = matched[(model, "scaleshift")]
        meta = matched_rows[0]["models"][model]
        operating.append({"id": i, "model": model, "input": "x".join(map(str, meta["effective_size"])),
            "original_resolution_roi": meta["original_resolution_roi"], "params": meta["params"],
            **item["balanced"], "failed": item["failed"], "catastrophic": item["catastrophic"], **times[model]})
    for point, flag in zip(operating, pareto_flags([(p["absrel"], p["mean_ms"]) for p in operating])):
        point["pareto_absrel_latency"] = flag
    csv_file(tables / "study9_operating_points.csv", ["id", "model", "input", "original_resolution_roi", "params",
        *METRICS, "failed", "catastrophic", "mean_ms", "p50_ms", "p95_ms", "repeat_min_ms", "repeat_max_ms",
        "peak_allocated_mib", "state_mib", "frames", "repeats", "pareto_absrel_latency"], operating)
    csv_file(tables / "study10_latency_sequences.csv", ["model", "source", "sequence", "clip_id", "mean_ms", "p50_ms",
        "p95_ms", "max_ms", "repeat_min_ms", "repeat_max_ms", "peak_allocated_mib", "state_mib", "keyframes",
        "dense_fallbacks", "gmc_calls", "gmc_fallbacks", "frames", "repeats"], latency_rows)
    csv_file(tables / "study10_frame_events.csv", ["model", "sequence", "event", "unique_frames",
             "timed_frames_with_repeats", "mean_ms", "p95_ms"], events)
    profile_rows = []
    stage_names = ("read_decode_preprocess", "h2d", "detector", "gmc", "embedding", "backbone", "decoder",
                   "network_and_output_resize", "d2h")
    for row in rows(directory / "profile.jsonl"):
        run = row["runs"][0]
        components = {s: statistics.fmean(t["components_ms"].get(s, 0) for t in run["telemetry"]) for s in stage_names}
        total = statistics.fmean(run["ms"])
        if total + 1e-3 < sum(components.values()):
            raise ValueError("component timers overlap or exceed the measured total")
        profile_rows.append({"model": row["model"], "source": row["source"], "sequence": row["sequence"],
                             **components, "profile_total_ms": total, "other_ms": total - sum(components.values())})
    csv_file(tables / "study10_components.csv", ["model", "source", "sequence", *stage_names,
             "profile_total_ms", "other_ms"], profile_rows)
    scatter(Path("paper/figures/study9_efficiency.svg"), operating)

    paired, sequences = [], []
    old8_rows, old256_rows = rows(PRIOR / "accuracy.jsonl"), rows(PRIOR / "long.jsonl")
    old8, old256 = aggregate_quality(old8_rows), aggregate_quality(old256_rows)
    scopes = [("L8_existing_487", old8), ("L256_existing_13", old256), ("L256_new_13", quality)]
    for scope, scores in scopes:
        for (model, gauge), item in scores.items():
            for sequence, value in item["sequences"].items():
                sequences.append({"scope": scope, "model": model, "gauge": gauge, "sequence": sequence, **value})
            if model != "sparse_k30" and ("sparse_k30", gauge) in scores:
                for metric in ("absrel", "delta1", "tce"):
                    paired.append(compare(scores, "sparse_k30", model, gauge, metric, scope))
    # Direct mechanism pairs, in addition to comparisons with the default.
    for a, b in (("dense_carry", "dense_reset"), ("delta_sp", "drop_sp"), ("delta_no_sp", "drop_no_sp")):
        for gauge in ("none", "median", "scaleshift"):
            for metric in ("absrel", "delta1", "tce"):
                paired.append(compare(old256, a, b, gauge, metric, "L256_existing_13"))
    for model in POINTS:
        if model == "sparse_k30":
            continue
        a = {r["sequence"]: r["mean_ms"] for r in latency_rows if r["model"] == "sparse_k30"}
        b = {r["sequence"]: r["mean_ms"] for r in latency_rows if r["model"] == model}
        seq = sorted(a)
        paired.append({"scope": "latency_4x256", "left": "sparse_k30", "right": model,
            "gauge": "not_applicable", "metric": "ms_per_frame", "sequences": seq,
            "left_equal_sequence": statistics.fmean(a.values()), "right_equal_sequence": statistics.fmean(b.values()),
            **paired_sequence_summary([a[s] - b[s] for s in seq])})

    seed_summary, seed_by_run = [], []
    for length, originals in ((8, old8_rows), (256, old256_rows)):
        additional = rows(directory / f"seeds_L{length}.jsonl")
        by_id = {r["id"]: r for r in originals}
        for row in additional:
            source = by_id[row["id"]]
            if row["no_gt"] != source["no_gt"] or row["pairs"] != source["pairs"]:
                raise ValueError("seed runs used different scorable frames")
            if not row["no_gt"]:
                row["models"]["seed0"] = source["models"]["sparse_k30"]
        scores = aggregate_quality(additional)
        for (model, gauge), item in scores.items():
            seed_by_run.append({"length": length, "seed": int(model[-1]), "gauge": gauge,
                **item["balanced"], "failed": item["failed"], "catastrophic": item["catastrophic"], "n": item["n"]})
            for sequence, value in item["sequences"].items():
                sequences.append({"scope": f"seed_L{length}", "model": model, "gauge": gauge,
                                  "sequence": sequence, **value})
        for gauge in ("none", "median", "scaleshift"):
            for metric in METRICS:
                values = [scores[(f"seed{s}", gauge)]["balanced"][metric] for s in range(3)]
                seed_summary.append({"length": length, "gauge": gauge, "metric": metric, "seed_n": 3,
                    "mean": statistics.fmean(values), "sample_sd": statistics.stdev(values), "min": min(values), "max": max(values)})
    csv_file(tables / "study11_seed_runs.csv", ["length", "seed", "gauge", *METRICS, "failed", "catastrophic", "n"], seed_by_run)
    csv_file(tables / "study11_seed_summary.csv", ["length", "gauge", "metric", "seed_n", "mean", "sample_sd", "min", "max"], seed_summary)
    csv_file(tables / "study11_sequences.csv", ["scope", "model", "gauge", "sequence", *METRICS,
             "failed", "catastrophic", "n"], sequences)
    csv_file(tables / "study11_paired_ci.csv", ["scope", "left", "right", "gauge", "metric", "left_equal_sequence",
        "right_equal_sequence", "difference", "ci_low", "ci_high", "sign_flip_p", "sequence_n", "bootstrap_draws", "sign_flip_draws"], paired)
    (tables / "study11_paired_details.json").write_text(json.dumps(paired, ensure_ascii=False, allow_nan=False) + "\n")

    baseline = next(p for p in operating if p["model"] == "sparse_k30")
    competitors = [p for p in operating if POINTS[p["model"]]["kind"] != "native"]
    quality_matches = [p for p in competitors if p["absrel"] <= baseline["absrel"] * 1.05
                       and p["delta1"] >= baseline["delta1"] - .01 and p["failed"] <= baseline["failed"]]
    latency_matches = [p for p in competitors if p["mean_ms"] <= baseline["mean_ms"] * 1.10]
    fastest = min(quality_matches, key=lambda p: p["mean_ms"]) if quality_matches else None
    best = min(latency_matches, key=lambda p: p["absrel"]) if latency_matches else None
    matches = {"reference": "sparse_k30", "same_quality_tolerance": {"absrel_factor": 1.05, "delta1_loss": .01,
                "failures_increase": 0}, "latency_factor": 1.10,
               "fastest_quality_match": fastest, "best_latency_match": best}
    (tables / "study9_matched_points.json").write_text(json.dumps(matches, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

    doc = ["# 논문화 작업 9~11 결과", "", "2026-09-06. 고정 v11 개발 검증. 신규 학습·최종 테스트 추론·외부 제출 없음.", "",
        "## 9. 경량 비교군과 품질–시간 운용점", "",
        "15개 운용점을 L256 13개/3,328프레임에서 평가했다. 아래 표의 품질과 시간은 모두 개발 시퀀스당 "
        "첫 L256 하나씩, 동일한 4개/1,024프레임에 한정한다. 같은 가중치의 K/해상도 변경이며 모델을 재학습하지 않았다.", "",
        "AbsRel/δ1/TCE는 clip scale–shift 정합 후 값이며 실시간 거리 보정 없는 정확도와 다르다. "
        "소스별 유효 픽셀 합산 후 TUM/Bonn을 1:1로 평균했다. 시간도 소스 균형 평균이고 정합/GT/지표 계산은 시간에서 제외했다.", ""]
    doc += table(["ID", "조건", "실제 입력", "M params", "AbsRel↓", "δ1↑", "실패", "평균 ms↓", "P95 ms", "peak MiB"], [
        [p["id"], p["model"], p["input"], f"{p['params']/1e6:.2f}", f"{p['absrel']:.4f}", f"{p['delta1']:.4f}",
         p["failed"], f"{p['mean_ms']:.3f}", f"{p['p95_ms']:.3f}", f"{p['peak_allocated_mib']:.1f}"] for p in operating]) + ["",
        "P95는 각 시퀀스의 프레임별 반복 측정 95분위를 소스 균형 집계한 값이며 전체 프레임의 global P95가 아니다. "
        "Peak는 한 번에 한 모델만 로딩한 프로세스의 최대 allocated GPU memory이며 가중치·활성·상태·런타임 workspace를 포함한다.", "",
        "256 목표는 frozen common RGB, 나머지 목표는 원본 해상도의 동일 scoring ROI다. "
        "MiDaS/DA2는 metric 출력이나 미래 프레임을 사용하지 않는 상대 깊이 모델이다. "
        "GMC는 tau=0.1/0.05의 별도 진단 행으로 기본 pixel gate tau=0.05/0.025와 다르다.", "",
        "![품질–시간 운용점](figures/study9_efficiency.svg)", "",
        "### 사전 지정 허용오차 내 비교", "",
        "품질 허용선은 기본 대비 AbsRel ≤1.05×, δ1 손실 ≤0.01, 실패 수 증가 없음이다. "
        "동일 시간 탐색은 기본 latency의 1.10× 이하다. 이는 실측 grid 탐색이지 통계적 동등성 판정이 아니다.", ""]
    if fastest:
        doc += [f"- 품질 조건을 만족하는 가장 빠른 비교군: `{fastest['model']}`, AbsRel {fastest['absrel']:.4f}, "
                f"{fastest['mean_ms']:.3f} ms. 기본은 {baseline['mean_ms']:.3f} ms이며 비교군/기본 시간 비는 "
                f"{fastest['mean_ms']/baseline['mean_ms']:.2f}×다."]
    else:
        doc += ["- 평가한 비교군 grid에 품질 허용선을 만족하는 점이 없다. 같은 품질에서의 speedup은 산출하지 않는다."]
    if best:
        doc += [f"- 시간 조건을 만족하는 비교군 중 최소 AbsRel: `{best['model']}` {best['absrel']:.4f} "
                f"({best['mean_ms']:.3f} ms), 기본 {baseline['absrel']:.4f}."]
    else:
        doc += ["- 평가한 비교군 grid에 시간 허용선을 만족하는 점이 없다. 같은 시간에서의 품질 우위는 산출하지 않는다."]
    doc += ["", "[운용점 전체 CSV](tables/study9_operating_points.csv) · [전체 13개와 시간 subset의 품질 분리](tables/study9_quality.csv) · "
            "[운용점 선택 결과](tables/study9_matched_points.json)", "",
            "### 시간 subset의 대표성 한계", "",
            "시간 subset은 결과를 보기 전에 정한 각 시퀀스의 첫 클립이다. 전체 13개와 점수·순위가 달라질 수 있다. "
            "아래와 같이 같은 모델도 달라지므로 전체 정확도와 subset latency를 합쳐 같은 품질의 speedup으로 제시하지 않는다. "
            "기존 native 경로 7종×13클립은 5~8의 점수와 동등함을 확인했다.", ""]
    doc += table(["조건", "전체 13개 AbsRel", "시간 subset 4개 AbsRel"], [[name,
        f"{quality[(name, 'scaleshift')]['balanced']['absrel']:.4f}",
        f"{matched[(name, 'scaleshift')]['balanced']['absrel']:.4f}"]
        for name in ("sparse_k30", "dense_carry", "output_hold", "da2_256", "midas_256")]) + ["",
        "운용점의 우열은 이 개발 subset 및 지정 허용오차에 한정한다. 전체 스트림에서의 품질 보존이나 "
        "새 장면 일반화의 증거로 확대하지 않는다.", "", "## 10. 실제 경로와 비용 분해", "",
        "RTX 4090, batch=1, FP32 eager, TF32 off, PyTorch/OpenCV thread=2/1. PNG 파일 읽기/디코딩, "
        "crop/resize/정규화, H2D, 실제 검출기·GMC·키프레임·fallback·backbone·decoder, 256 출력 보간과 CPU 복사를 포함했다. "
        "모델 로딩/GT/RAFT/정합/채점/출력 파일 저장은 제외했다. 파일 캐시는 warm 상태이며 카메라·영상 코덱·네트워크·cold storage를 포함한 제품 지연은 아니다.", "",
        "각 256프레임 클립 warmup 1회 뒤 상태를 초기화해 5회 반복했다. 첫 프레임/키프레임도 포함한다. "
        "매번 가장 빠른 결과만 선택하지 않았다. 모든 반복의 출력 SHA-256이 품질 pass와 같음을 검사했다. "
        "각 반복 시작에 다른 GPU compute process가 있으면 중단하도록 했고 앞뒤 GPU 상태를 기록했다. "
        "이는 주기적 프로세스 검사이며 하드웨어를 독점 예약하거나 clock을 고정한 측정은 아니다.", "",
        "아래 비용 분해는 별도 1회 계측 pass에서 component 경계마다 GPU 동기화를 추가했다. "
        "합산 총시간은 이 계측 pass의 값이며 위 주 latency와 동일하지 않다. 상대 모델은 network와 출력 보간을 한 항목으로 묶었다.", ""]
    component_table = []
    for name in ("sparse_k30", "dense_carry", "output_hold", "gmc_k30", "da2_256", "midas_256"):
        members = [r for r in profile_rows if r["model"] == name]
        values = {k: balanced_mean(members, k) for k in (*stage_names, "profile_total_ms", "other_ms")}
        component_table.append([name, *[f"{values[k]:.3f}" for k in ("read_decode_preprocess", "h2d", "detector", "gmc",
                                "embedding", "backbone", "decoder", "network_and_output_resize", "d2h", "other_ms", "profile_total_ms")]])
    doc += table(["조건", "입력", "H2D", "검출", "GMC", "embed", "backbone", "decoder", "상대모델 전체", "D2H", "기타", "계측 합계 ms"], component_table) + ["",
        "[시퀀스별 latency·반복 범위·stream state](tables/study10_latency_sequences.csv) · "
        "[키프레임/fallback별 시간](tables/study10_frame_events.csv) · [비용 분해](tables/study10_components.csv)", "",
        "이 측정은 모든 모델의 FP32 eager 경로 비교다. FP16/compile/TensorRT 최적화 모델의 최고 성능이나 "
        "이전 GPU-resident 속도 표와 직접 혼합하지 않는다. MAC이 적다고 파일-to-depth 지연이 반드시 줄지는 않는다.", "",
        "## 11. 시퀀스 단위 paired 불확실성과 seed", "",
        "아래의 차이는 `left − right`이며 AbsRel/TCE/시간은 음수가 left에 유리하다. "
        "각 시퀀스 내 pixel-pooled 점수를 구한 뒤 네 시퀀스를 동일 가중 평균한다. 위 TUM/Bonn 1:1 평균과 estimand가 다르다.", "",
        "95% 구간은 paired sequence bootstrap의 4⁴=256개 복원추출 조합을 전수 열거한 percentile 구간이다. "
        "표본이 네 개라 coverage가 보장되는 정확한 구간이 아니며 탐색적이다. "
        "양측 sign-flip도 2⁴=16개를 전수 열거했다. 가능한 최소 p=0.125이므로 5% 유의성을 주장하지 않는다. "
        "인접 클립이나 실행 반복을 독립 시퀀스로 세지 않았고, 다중 비교 보정 없는 구간을 확정 검정으로 해석하지 않는다.", ""]
    doc += ["부호 반전 p는 귀무가설 아래 차이의 대칭성/교환가능성을 가정한 보조 진단이다. "
            "시퀀스를 무작위 배정한 실험이 아니므로 인과적 우월성 검정으로 읽지 않는다. "
            "이 작은 표본에서는 bootstrap 구간이 0을 제외해도 sign-flip 기준 유의성과 일치하지 않을 수 있다.", ""]
    show = []
    for scope, left, right, gauge, metric in (
        ("L8_existing_487", "sparse_k30", "dpt_common", "scaleshift", "absrel"),
        ("L256_existing_13", "sparse_k30", "dense_carry", "scaleshift", "absrel"),
        ("L256_existing_13", "dense_carry", "dense_reset", "scaleshift", "tce"),
        ("L256_existing_13", "delta_sp", "drop_sp", "scaleshift", "absrel"),
        ("L256_new_13", "sparse_k30", "midas_256", "scaleshift", "absrel"),
        ("latency_4x256", "sparse_k30", "dense_carry", "not_applicable", "ms_per_frame")):
        p = next(p for p in paired if (p["scope"], p["left"], p["right"], p["gauge"], p["metric"]) == (scope, left, right, gauge, metric))
        show.append([scope, left + " − " + right, metric, f"{p['difference']:+.5f}",
                     f"[{p['ci_low']:+.5f}, {p['ci_high']:+.5f}]", f"{p['sign_flip_p']:.3f}"])
    doc += table(["자료", "비교", "지표", "차이", "paired 95% 구간", "양측 p"], show) + ["",
        "[시퀀스별 점수](tables/study11_sequences.csv) · [전체 paired CI](tables/study11_paired_ci.csv) · "
        "[순서·시퀀스별 차이 JSON](tables/study11_paired_details.json)", "",
        "### 마지막 8k 단계의 seed 0/1/2", "",
        "기존 세 checkpoint를 현재 동일 채점 코드로 비교했다(seed0 기존 전수 결과 재사용, L256 streaming 동등성 재검사). "
        "세 모델은 같은 v10 부모에서 마지막 8,000단계만 seed가 다르므로 독립적인 전체 학습 3회가 아니다. "
        "기록된 학습 인자는 seed/저장 경로 외 동일하다. 기록된 commit에는 RAFT batch 분할 및 기본값 1.0의 bin temperature 추가 차이가 있다. "
        "이는 현재 추론 코드를 통일한 비교이며 당시 미기록 worktree 변경까지 증명하지 않는다.", ""]
    doc += table(["길이", "정합", "AbsRel mean±sample SD", "범위"], [[length, gauge,
        f"{r['mean']:.4f} ± {r['sample_sd']:.4f}", f"[{r['min']:.4f}, {r['max']:.4f}]"]
        for length in (8, 256) for gauge in ("none", "median", "scaleshift")
        for r in seed_summary if r["length"] == length and r["gauge"] == gauge and r["metric"] == "absrel"]) + ["",
        "SD는 마지막 학습 단계의 관측 변동성이지 전체 학습/새 시퀀스 일반화의 신뢰구간이 아니다. "
        "기존 개발 자료가 모델 선택에 반복 사용되었으므로 이 평균과 구간도 독립 test의 성능 추정으로 승격하지 않는다. "
        "[seed별 전체 지표](tables/study11_seed_runs.csv) · [평균·SD·범위](tables/study11_seed_summary.csv). "
        "각 checkpoint/config/부모 해시와 역사적 소스 fingerprint는 `work_dirs/paper_study_9_11/seed_provenance.json`에 있다.", "",
        "## 완료 범위와 재현", "",
        "9~11의 위 개발 분석을 완료했다. 성능 목표 충족·독립 최종 test 성공·투고 준비 완료를 뜻하지 않는다. "
        "이번 비교군은 DA2 Small과 MiDaS Small이며 둘 다 native보다 파라미터가 많다. "
        "서로 다른 사전학습/학습 자료와 학습량은 통제되지 않았고 데이터 중복도 완전히 검증되지 않았다. "
        "따라서 차이를 아키텍처 하나의 인과적 효과로 해석하지 않는다. "
        "FastDepth 공식 서버의 HTTP/HTTPS 연결 timeout으로 해당 실측은 포함하지 않았고, 초소형 경쟁군 전체에 대한 우월성을 주장하지 않는다. "
        "GMC는 이 네 개발 시퀀스의 진단일 뿐, 새로운 움직이는 카메라/장면의 검증을 대신하지 않는다.", "",
        "실행 계획: [STUDY_9_11_PLAN.md](STUDY_9_11_PLAN.md). 기존 가중치·평가 코드·최종 test 봉인 자료는 변경하지 않았다.", "",
        "```bash", "python scripts/prepare_efficiency_dependencies.py",
        "python scripts/paper_efficiency_study.py --phase quality",
        "python scripts/paper_efficiency_study.py --phase latency", "python scripts/paper_efficiency_study.py --phase profile",
        "python scripts/paper_efficiency_study.py --phase seeds", "python scripts/paper_efficiency_report.py", "```", "",
        "MiDaS는 [공식 소형 모델](https://pytorch.org/hub/intelisl_midas_v2/)과 "
        "[v2.1 구현](https://github.com/isl-org/MiDaS/tree/94afff70fe0873c3af70355e5388e4d482bfc807)을 사용했다. "
        "가중치와 transitive EfficientNet 구현은 로컬 revision/파일 해시로 기록했다. "
        "FastDepth의 공식 배포 위치는 [저자 저장소](https://github.com/dwofk/fast-depth)에 따른다.", ""]
    dense = next(p for p in operating if p["model"] == "dense_carry")
    members = [r for r in profile_rows if r["model"] == "sparse_k30"]
    pre = balanced_mean(members, "read_decode_preprocess")
    total = balanced_mean(members, "profile_total_ms")
    conclusion = ["## 논문 주장에 대한 판정", "",
        f"- **경량 모델의 이점과 희소화의 이점은 별개다.** 기본 {baseline['mean_ms']:.3f} ms와 동일 가중치 dense "
        f"{dense['mean_ms']:.3f} ms의 차이는 {baseline['mean_ms'] - dense['mean_ms']:+.3f} ms "
        f"({100 * (baseline['mean_ms']/dense['mean_ms'] - 1):+.2f}%)다. "
        "이 값을 이전의 MAC 절감률과 동일시하거나 큰 실사용 가속으로 표현하지 않는다.",
        f"- 기본 경로의 read/decode/preprocess는 별도 계측 총시간의 {100 * pre/total:.1f}%다. "
        "이 실행 경계에서는 네트워크 연산만 줄여 얻을 수 있는 이득이 제한된다.",
        f"- 기본의 stream state 최대 {baseline['state_mib']:.3f} MiB는 dense의 {dense['state_mib']:.3f} MiB보다 "
        "작지 않다. 캐시는 추가 저장 공간을 사용한다. 작은 전체 모델과 작은 stream state를 혼동하지 않는다.",
        "- 품질–시간의 허용오차 내 비교 결과는 위 실측 subset에 한정한다. "
        "희소화가 큰 end-to-end 가속을 제공한다는 주장, 독립 test 일반화, 전체 학습 seed 안정성은 별도로 검증해야 한다.", ""]
    insertion = doc.index("## 완료 범위와 재현")
    doc[insertion:insertion] = conclusion
    target = Path("paper/STUDY_9_11.md")
    target.write_text("\n".join(doc))
    artifacts = [target, Path("paper/figures/study9_efficiency.svg"),
                 *sorted(p for p in tables.iterdir() if p.name.startswith(("study9_", "study10_", "study11_")))]
    (directory / "report_artifacts.json").write_text(json.dumps({"generator": file_record(__file__),
        "inputs": [file_record(directory / p) for p in ("study.json", "seed_provenance.json", "quality_complete.json",
                   "latency_complete.json", "profile_complete.json", "seeds_complete.json")],
        "artifacts": [file_record(p) for p in artifacts]}, indent=2) + "\n")
    print(f"created {target} and {len(artifacts)-1} supporting artifacts")


if __name__ == "__main__":
    main()
