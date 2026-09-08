"""Export tasks 5–8 tables and matched-frame curves from completed study rows."""
import argparse
from collections import defaultdict
import csv
import html
import json
from pathlib import Path
import statistics
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sokkanaem.metrics import pooled, robust
from sokkanaem.protocol import file_record
from sokkanaem.study import BENCH_ARMS


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def mean(values):
    return statistics.fmean(values)


def aggregate(rows):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for model, result in row["models"].items():
            for gauge, value in result["gauges"].items():
                grouped[(model, gauge)][row["source"]].append((row, result, value))
    output = {}
    for (model, gauge), sources in grouped.items():
        per_source = {}
        for source, members in sources.items():
            sc = pooled([value["scores"]["_pooled"] for _, _, value in members])
            if gauge.endswith("/frame"):
                for key in ("tce", "opw", "temporal_delta"):
                    sc[key] = None
            per_source[source] = {**sc, **{"clip_" + k: v for k, v in
                robust([value["scores"]["absrel"] for _, _, value in members]).items()},
                "failed": sum(value["failed"] for _, _, value in members),
                "catastrophic": sum(value["catastrophic"] for _, _, value in members),
                "n": len(members)}
        first = next(iter(sources.values()))[0][1]
        balanced = {key: mean([s[key] for s in per_source.values()])
                    for key in per_source[next(iter(per_source))]
                    if key not in ("failed", "catastrophic", "n")
                    and per_source[next(iter(per_source))][key] is not None}
        balanced.update({key: sum(s[key] for s in per_source.values())
                         for key in ("failed", "catastrophic", "n")})
        for key in ("active_all", "active_post_first", "linear_conv_gmac_per_frame"):
            if key in first:
                balanced[key] = mean([mean([result[key] for _, result, _ in members])
                                      for members in sources.values()])
        output[(model, gauge)] = {"balanced": balanced, "sources": per_source,
                                  "params": first["params"], "input": first["effective_size"],
                                  "future_frames": first["future_frames"]}
    return output


def frame_curve(rows, model, gauge):
    grouped = defaultdict(list)
    for row in rows:
        if model in row["models"]:
            grouped[row["source"]].append(row["models"][model]["gauges"][gauge]["frames"])
    source_curves = []
    for members in grouped.values():
        rel = np.sum([m["rel_sum"] for m in members], axis=0)
        px = np.sum([m["px"] for m in members], axis=0)
        source_curves.append(np.divide(rel, px, out=np.full_like(rel, np.nan), where=px > 0))
    values = np.mean(source_curves, axis=0)
    return [float(v) if np.isfinite(v) else None for v in values]


def csv_file(path, columns, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def table(headers, rows):
    return ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] + [
        "| " + " | ".join(str(v) for v in row) + " |" for row in rows]


def svg_curves(path, curves):
    arms = ["sparse_k30", "dense_carry", "dense_reset", "sparse_k5", "sparse_k10", "sparse_k60"]
    colors = ["#cc3311", "#0077bb", "#009988", "#ee7733", "#aa3377", "#555555"]
    panels = [("none", "No GT fit"), ("scaleshift", "Clip scale-shift"),
              ("scaleshift/frame", "Frame scale-shift (shape only)")]
    out = ['<svg xmlns="http://www.w3.org/2000/svg" width="1160" height="390" viewBox="0 0 1160 390">',
           '<title>Matched 256-frame development clips: alignment and refresh controls</title>',
           '<rect width="1160" height="390" fill="white"/>',
           '<g font-family="Arial,sans-serif" font-size="11" fill="#222">']
    for panel, (gauge, title) in enumerate(panels):
        x0, y0, width, height = 55 + panel * 385, 42, 300, 240
        values = [v for arm in arms for v in curves[gauge][arm] if v is not None]
        top = max(values) * 1.06
        out.append(f'<text x="{x0}" y="24" font-size="14">{html.escape(title)}</text>')
        for i in range(5):
            y = y0 + height * (1 - i / 4)
            out.append(f'<path d="M{x0},{y}h{width}" stroke="#dddddd"/>')
            out.append(f'<text x="{x0 - 7}" y="{y + 4}" text-anchor="end">{top * i / 4:.2f}</text>')
        for tick in (0, 64, 128, 192, 255):
            x = x0 + width * tick / 255
            out.append(f'<text x="{x}" y="{y0 + height + 19}" text-anchor="middle">{tick}</text>')
        out.append(f'<text x="{x0 + width / 2}" y="{y0 + height + 36}" text-anchor="middle">frame index</text>')
        for arm, color in zip(arms, colors):
            points = " ".join(f"{x0 + width * i / 255:.2f},{y0 + height * (1 - v / top):.2f}"
                              for i, v in enumerate(curves[gauge][arm]) if v is not None)
            out.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.5"/>')
    for i, (arm, color) in enumerate(zip(arms, colors)):
        x = 50 + i * 185
        out.append(f'<path d="M{x},355h23" stroke="{color}" stroke-width="3"/>')
        out.append(f'<text x="{x + 29}" y="359">{arm}</text>')
    out.append('<text x="55" y="384">AbsRel; pixel-pooled per source, equal-source mean. Same 13 clips in every arm. Panels have separate y scales.</text></g></svg>')
    path.write_text("\n".join(out) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("work_dirs/paper_study_5_8"))
    args = ap.parse_args()
    directory = args.out
    contract = json.loads((directory / "study.json").read_text())
    if contract["smoke"]:
        raise ValueError("smoke subsets must not become paper results")
    for phase in ("accuracy", "long", "bench"):
        completion = json.loads((directory / f"{phase}_complete.json").read_text())
        result = completion["result"]
        if completion["smoke"] or file_record(result["path"])["sha256"] != result["sha256"]:
            raise ValueError(f"unverified phase: {phase}")
    accuracy_rows = read_rows(directory / "accuracy.jsonl")
    long_rows = read_rows(directory / "long.jsonl")
    for length, rows in ((8, accuracy_rows), (256, long_rows)):
        manifest = json.loads(Path(f"manifests/acc_real_L{length}.json").read_text())
        expected = {(s, i) for s in manifest["sources"] for i in range(
                    sum(c["source"] == s for c in manifest["clips"]))}
        if len(rows) != len(expected) or {(r["source"], r["index"]) for r in rows} != expected:
            raise ValueError("study did not evaluate the full declared manifest")
        expected_models = ({"sparse_k30", "dpt_common", "dpt_official", "da2_common",
                            "da2_official", "zoe_common", "zoe_official"} if length == 8
                           else set(contract["long_arms"]))
        declared = {s: [c for c in manifest["clips"] if c["source"] == s]
                    for s in manifest["sources"]}
        for row in rows:
            pairs = declared[row["source"]][row["index"]]["pairs"]
            if (row["id"] != f"{row['source']}:{row['index']}" or row["pairs"] != pairs
                    or row["clip_len"] != length or row["first_rgb"] != pairs[0][0]):
                raise ValueError("record identity differs from the declared manifest")
            if set(row["models"]) != (set() if row["no_gt"] else expected_models):
                raise ValueError("models were not evaluated on identical scorable clips")
    # GT-defined regions must contain the same pixels for every model/gauge.
    for row in accuracy_rows:
        reference = None
        for model in row["models"].values():
            for gauge in model["gauges"].values():
                counts = {k: v["px"] for k, v in gauge["regions"].items()}
                if reference is not None and counts != reference:
                    raise ValueError("region pixel counts differ across model/gauge rows")
                reference = counts
                for names in (("edge", "flat"), ("dynamic", "static"), ("near", "mid", "far")):
                    if sum(counts[n] for n in names) != counts["all"]:
                        raise ValueError("regions fail to partition valid GT")
    accuracy, long = aggregate(accuracy_rows), aggregate(long_rows)
    tables = Path("paper/tables")
    tables.mkdir(exist_ok=True)
    comparison = []
    for (model, gauge), item in accuracy.items():
        for source, scores in {**item["sources"], "balanced": item["balanced"]}.items():
            comparison.append({"model": model, "gauge": gauge, "source": source,
                               "params": item["params"], "input": "x".join(map(str, item["input"])),
                               "future_frames": item["future_frames"], **scores})
    csv_file(tables / "study5_accuracy.csv", ["model", "gauge", "source", "params", "input",
        "future_frames", "absrel", "rmse", "delta1", "tce", "failed", "catastrophic", "n"], comparison)
    tails, failures, region_rows = [], [], []
    for (model, gauge), item in accuracy.items():
        for source, scores in {**item["sources"], "balanced": item["balanced"]}.items():
            tails.append({"model": model, "gauge": gauge, "source": source, **scores})
        for source in item["sources"]:
            relevant = [(row, row["models"][model]["gauges"][gauge]) for row in accuracy_rows
                        if row["source"] == source and model in row["models"]]
            worst = {row["id"] for row, _ in sorted(relevant, key=lambda pair: pair[1]["scores"]["absrel"], reverse=True)[:5]}
            for row, val in relevant:
                if row["id"] in worst or val["failed"] or val["catastrophic"]:
                    failures.append({"model": model, "gauge": gauge, "source": source,
                        "clip_id": row["id"], "sequence": row["sequence"], "first_rgb": row["first_rgb"],
                        "absrel": val["scores"]["absrel"], "failed": val["failed"],
                        "catastrophic": val["catastrophic"], "neg_frac": val["neg_frac"],
                        "top5_within_source": row["id"] in worst})
            for region in ("all", "edge", "flat", "dynamic", "static", "near", "mid", "far"):
                entries = [v["regions"][region] for _, v in relevant]
                px = sum(e["px"] for e in entries)
                region_rows.append({"model": model, "gauge": gauge, "source": source, "region": region,
                    "absrel": sum(e["rel"] for e in entries) / px if px else None,
                    "delta1": sum(e["d1"] for e in entries) / px if px else None, "px": px})
    csv_file(tables / "study6_tails.csv", ["model", "gauge", "source", "clip_mean", "clip_median", "clip_p90",
             "clip_p95", "clip_max", "failed", "catastrophic", "n"], tails)
    csv_file(tables / "study6_failures.csv", ["model", "gauge", "source", "clip_id", "sequence", "first_rgb",
             "absrel", "failed", "catastrophic", "neg_frac", "top5_within_source"], failures)
    csv_file(tables / "study6_regions.csv", ["model", "gauge", "source", "region", "absrel", "delta1", "px"], region_rows)
    curves = {gauge: {model: frame_curve(long_rows, model, gauge) for model in contract["long_arms"]}
              for gauge in ("none", "median", "scaleshift", "median/frame", "scaleshift/frame")}
    (tables / "study7_frame_curves.json").write_text(json.dumps(curves, allow_nan=False) + "\n")
    long_table = []
    for (model, gauge), item in long.items():
        curve = curves[gauge][model]
        early = mean([v for v in curve[:8] if v is not None])
        late = mean([v for v in curve[-8:] if v is not None])
        long_table.append({"model": model, "gauge": gauge, **item["balanced"],
                           "first8_absrel": early, "last8_absrel": late,
                           "late_early_relative_change": late / early - 1})
    csv_file(tables / "study7_long.csv", ["model", "gauge", "absrel", "rmse", "delta1", "tce",
        "first8_absrel", "last8_absrel", "late_early_relative_change", "active_all", "n"], long_table)
    timings = json.loads((directory / "ablation_latency.json").read_text())
    first_per_sequence = {}
    for row in long_rows:
        first_per_sequence.setdefault(row["sequence"], row["id"])
    expected_windows = {(s, clip, arm) for s, clip in first_per_sequence.items() for arm in BENCH_ARMS}
    if (len(timings["rows"]) != len(expected_windows)
            or {(r["sequence"], r["clip_id"], r["arm"]) for r in timings["rows"]} != expected_windows
            or any(r["frames"] != contract["bench"]["frames"]
                   or len(r["window_ms_per_frame"]) != contract["bench"]["repeats"]
                   or not all(np.isfinite(t) and t > 0 for t in r["window_ms_per_frame"])
                   for r in timings["rows"])):
        raise ValueError("benchmark windows differ from the declared paired design")
    latency = {}
    for model in BENCH_ARMS:
        sources = defaultdict(list)
        for row in timings["rows"]:
            if row["arm"] == model:
                sources[row["clip_id"].split(":")[0]].append(mean(row["window_ms_per_frame"]))
        latency[model] = mean([mean(v) for v in sources.values()])
    ablations = [{"model": m, "gauge": g, **long[(m, g)]["balanced"], "ms_per_frame": latency[m]}
                 for g in ("none", "median", "scaleshift") for m in BENCH_ARMS]
    csv_file(tables / "study8_ablations.csv", ["model", "gauge", "absrel", "delta1", "tce", "failed", "active_all",
        "linear_conv_gmac_per_frame", "ms_per_frame"], ablations)
    svg_curves(Path("paper/figures/study7_long.svg"), curves)

    doc = ["# 논문화 작업 5~8 결과", "", "2026-09-06. 고정 v11, 신규 학습 없음. 개발 검증 분석이며 최종 테스트는 열지 않았다.", "",
           "## 실행·재현 범위", "",
           f"- L8: {len(accuracy_rows)}개 중 유효 GT 클립 {sum(not r['no_gt'] for r in accuracy_rows)}개. L256: {len(long_rows)}개, 동일 3,328 프레임.",
           "- 비교군: DPT-Large, DA V2 Small, ZoeDepth N-K. 모두 single-frame이며 미래 프레임을 사용하지 않는다.",
           "- 체크포인트와 원본 자료 해시, baseline revision, 소스 버전은 `work_dirs/paper_study_5_8/study.json` 및 `data_files.json`에 기록했다.",
           "- 새 4개 TUM test 시퀀스와 기존 가중치·v1 freeze 기록은 변경하지 않았다. 이번 결과로 최종 테스트 성능을 주장하지 않는다.",
           "- 공식 해상도 행은 원본 RGB의 채점 ROI에서 시작한다. 256px RGB 재확대 방식의 과거 행과 직접 같은 실험으로 간주하지 않는다.", "",
           "표의 AbsRel/RMSE/δ1/TCE는 각 소스(TUM/Bonn) 내부의 유효 픽셀 합산 점수를 구한 뒤 두 소스를 1:1로 평균한다. "
           "시퀀스 균등 평균이나 전체 클립 단순 평균과는 다르다. 실패 수는 실제 클립 수의 합이다. "
           "`실패`는 채점 유효 영역의 비양수 예측 또는 정합 후 비양수 시차가 존재함을 뜻하며 실행 중단 건수가 아니다. "
           "AbsRel>1인 catastrophic과 중복될 수 있다. RMSE의 단위는 m다. 정의는 [평가 계약](PROTOCOL.md)을 따른다.", "",
           "## 5. 동일 프레임 정확도", ""]
    for gauge in ("none", "median", "scaleshift"):
        doc += [f"### {gauge}", ""]
        rows = []
        for model in ("sparse_k30", "dpt_common", "da2_common", "zoe_common", "dpt_official", "da2_official", "zoe_official"):
            if (model, gauge) not in accuracy:
                continue
            item, b = accuracy[(model, gauge)], accuracy[(model, gauge)]["balanced"]
            rows.append([model, "×".join(map(str, item["input"])), f"{item['params'] / 1e6:.2f}",
                         f"{b['absrel']:.4f}", f"{b['rmse']:.4f}", f"{b['delta1']:.4f}", b["failed"], b["catastrophic"]])
        doc += table(["모델/입력 조건", "실제 입력", "M params", "AbsRel↓", "RMSE↓", "δ1↑", "실패", "AbsRel>1"], rows) + [""]
    own = accuracy[("sparse_k30", "scaleshift")]["balanced"]
    dpt = accuracy[("dpt_common", "scaleshift")]["balanced"]
    doc += [f"동일 입력 목표의 relative-shape에서 native AbsRel은 {own['absrel']:.4f}, DPT는 {dpt['absrel']:.4f}다. "
            f"Native의 상대 오차 차이는 {100 * (own['absrel'] / dpt['absrel'] - 1):+.1f}%다. "
            "None은 실제 거리 정확도, median은 GT 스케일 보정 후 정확도이므로 다른 정합 표의 순위를 섞지 않는다.", "",
            "DPT/DA2의 median 행은 offset이 미확정인 상대 disparity를 깊이로 변환한 뒤 1개 scale만 맞추는 보조 진단이다. "
            "큰 오차와 실패를 절대 깊이 성능으로 해석하거나 native 우월성의 근거로 쓰지 않는다. "
            "Scale–shift 역시 disparity 제곱오차를 최소화하므로 역수 변환 후 AbsRel/RMSE 개선을 보장하지 않는다.", "",
            "소스별 전체 수치: [study5_accuracy.csv](tables/study5_accuracy.csv).", "",
            "## 6. 취약 영역과 오차 꼬리", "",
            "아래는 common-input relative-shape의 소스 균형 AbsRel이다. 동적 영역은 GT 의미 분할이 아니라 "
            "동일 RAFT backward flow에서 dominant motion을 뺀 1.5px 초과 영역이다. 경계는 동일 GT에서 생성한다.", ""]
    region_table = []
    for region in ("edge", "flat", "dynamic", "static", "near", "mid", "far"):
        cells = []
        for model in ("sparse_k30", "dpt_common", "da2_common", "zoe_common"):
            values = [r["absrel"] for r in region_rows if r["model"] == model and r["gauge"] == "scaleshift"
                      and r["region"] == region and r["absrel"] is not None]
            cells.append(f"{mean(values):.4f}" if values else "NA")
        region_table.append([region, *cells])
    doc += table(["영역", "Native", "DPT", "DA2", "Zoe"], region_table) + [""]
    doc += table(["모델", "Clip P50", "Clip P95", "실패", "catastrophic"], [[m,
        f"{accuracy[(m, 'scaleshift')]['balanced']['clip_median']:.4f}",
        f"{accuracy[(m, 'scaleshift')]['balanced']['clip_p95']:.4f}",
        accuracy[(m, "scaleshift")]["balanced"]["failed"], accuracy[(m, "scaleshift")]["balanced"]["catastrophic"]]
        for m in ("sparse_k30", "dpt_common", "da2_common", "zoe_common")]) + ["",
        "P50/P95는 소스별 통계의 평균이며 global quantile이 아니다. 실패 클립은 평균에서 제외하지 않았다.", "",
        "[영역별 전체 표](tables/study6_regions.csv) · [오차 꼬리](tables/study6_tails.csv) · "
        "[실패 및 소스별 worst-5 클립](tables/study6_failures.csv).", "",
        "## 7. 동일 L256에서 장기 열화 분리", "",
        "모든 행은 같은 13개 클립이다. Dense carry/reset은 매 프레임 all-active이며, K 조건은 고정 가중치에서 "
        "키프레임 주기만 변경했다. 실제 RGB 변화량과 dense fallback은 활성률에 반영된다.", ""]
    arms7 = ("sparse_k30", "dense_carry", "dense_reset", "sparse_k5", "sparse_k10", "sparse_k60")
    doc += table(["조건", "활성률(전체)", "none AbsRel", "median", "scaleshift/clip", "scaleshift/frame", "clip TCE"], [[m,
        f"{long[(m, 'none')]['balanced']['active_all'] * 100:.1f}%", *[f"{long[(m, g)]['balanced']['absrel']:.4f}"
            for g in ("none", "median", "scaleshift", "scaleshift/frame")],
        f"{long[(m, 'scaleshift')]['balanced']['tce']:.4f}"] for m in arms7]) + ["",
        "![동일 프레임의 장기 오차 곡선](figures/study7_long.svg)", "",
        "곡선의 early/late 차이는 영상 내용 변화도 포함한다. 원인 해석은 같은 시점의 carry/reset/sparse 차이로 한다. "
            "Frame 정합은 실제 scale drift를 제거하므로 temporal 성능으로 읽지 않는다.", ""]
    carry = long[("dense_carry", "scaleshift")]["balanced"]["absrel"]
    reset = long[("dense_reset", "scaleshift")]["balanced"]["absrel"]
    sparse = long[("sparse_k30", "scaleshift")]["balanced"]["absrel"]
    doc += [f"같은 프레임의 carry−reset AbsRel은 {carry - reset:+.6f}, sparse−carry는 {sparse - carry:+.6f}다. "
            "이는 이 checkpoint와 데이터에서의 결과이며 모든 재귀 모델의 성질로 일반화하지 않는다.", "",
            f"Dense carry의 clip-fit TCE는 {long[('dense_carry', 'scaleshift')]['balanced']['tce']:.4f}, "
            f"reset은 {long[('dense_reset', 'scaleshift')]['balanced']['tce']:.4f}다. "
            "따라서 이 비교에서 상태 유지의 깊이 정확도 이득이 작다는 결과와 시간 일관성 이득이 있다는 결과를 구분한다.", "",
            "[길이/정합 분해 표](tables/study7_long.csv) · [프레임별 수치](tables/study7_frame_curves.json).", "",
            "## 8. 동일 mask에서 readout·출력 재사용·캐시 비교", "",
            "모든 행은 sparse K30의 post-fallback mask를 그대로 재생했다. `delta_sp`와 `drop_sp`는 "
            "temporal output cache를 끈 상태에서 readout 처리만 다르고 공간 캐시가 켜져 있다. "
            "`*_no_sp`는 공간 캐시도 끈다. `both_cache_replay`는 기본 모델 출력을 수치적으로 재현하는지 검사했다.", ""]
    doc += table(["조건", "AbsRel↓", "δ1↑", "TCE↓", "실패", "Linear+Conv GMAC", "ms/frame"], [[m,
        f"{long[(m, 'scaleshift')]['balanced']['absrel']:.4f}", f"{long[(m, 'scaleshift')]['balanced']['delta1']:.4f}",
        f"{long[(m, 'scaleshift')]['balanced']['tce']:.4f}", long[(m, "scaleshift")]["balanced"]["failed"],
        f"{long[(m, 'scaleshift')]['balanced']['linear_conv_gmac_per_frame']:.3f}", f"{latency[m]:.3f}"] for m in BENCH_ARMS]) + ["",
        "정확도/TCE는 clip scale–shift 표이며 [전체 정합 ablation](tables/study8_ablations.csv)에 none/median도 보존했다. "
        "MAC은 실제 실행된 Linear/Conv2d만 집계하며 fused SSM scan·정규화·elementwise·resize 등을 제외한다. "
        "이 수치를 전체 모델 FLOPs라고 부르면 안 된다. 현재 drop 구현은 temporal 출력을 나중에 mask하므로 "
        "state/readout 효과와 실제 linear 연산 절감은 다를 수 있다.", "",
        "`output_hold`는 stateless dense predictor의 active patch 출력만 채택하고 나머지는 이전 출력을 유지한다. "
        "완전 비활성 프레임을 제외하면 predictor는 여전히 dense다. 따라서 선택 비율을 곧바로 연산 절감률로 환산하지 않았다.", "",
        "정확도와 MAC은 전체 L256 3,328프레임에서, 시간 측정은 각 개발 시퀀스의 첫 L256 클립 중 "
        "앞 64프레임(총 256프레임), warmup 1회·반복 3회에서 구했다. 따라서 시간과 품질의 집계 프레임 집합은 다르다. "
        "GPU-resident FP32 eager, 고정 mask, 같은 프레임의 반복 window 평균을 소스 균형 집계했다. "
        "검출기·I/O·지표는 제외했으므로 이 값은 9~10번의 end-to-end latency가 아니다. "
        "GT 정합 계산 역시 측정 시간에서 제외했다.", "",
        "## 논문 주장에 대한 판정", ""]
    delta = long[("delta_sp", "scaleshift")]["balanced"]
    drop = long[("drop_sp", "scaleshift")]["balanced"]
    no_sp = long[("delta_no_sp", "scaleshift")]["balanced"]
    hold = long[("output_hold", "scaleshift")]["balanced"]
    base = long[("both_cache_replay", "scaleshift")]["balanced"]
    doc += table(["주장", "이번 결과의 판단", "후속 작업"], [
        ["작은 모델로 동급 형상 정확도", f"4.19M의 크기 이점은 확인되지만 common DPT 대비 AbsRel {100 * (own['absrel'] / dpt['absrel'] - 1):.1f}% 열세", "9번에서 경량 경쟁군과 비교"],
        ["기본 희소 경로의 장기 정확도 유지", f"K30은 동일 L256 dense carry 대비 AbsRel {100 * (sparse / carry - 1):.1f}% 증가", "11번 통계, 13번 refresh/캐시 개선 후보 검증"],
        ["Δ readout의 효과", f"공간 캐시 on·시간 캐시 off에서 drop 대비 AbsRel {100 * (1 - delta['absrel'] / drop['absrel']):.1f}% 감소", "같은 추론 개입 범위로 한정; 재학습 대조는 별도"],
        ["현재 캐시 조합의 최적성", f"캐시 off(delta_no_sp)는 기본 대비 AbsRel {100 * (1 - no_sp['absrel'] / base['absrel']):.1f}% 감소하지만 부분 MAC 증가", "품질·비용 trade-off로 보고"],
        ["단순 출력 재사용 대비 우월성", f"output_hold AbsRel {hold['absrel']:.4f} < 기본 {base['absrel']:.4f}; 제한된 시간 측정에서도 더 빠름", "보편적 우월성 주장은 보류; 9~10번 동일 운용점 검증"],
    ]) + ["", "이번 결과만으로 정확도 동등성·end-to-end 가속·독립 테스트 일반화를 주장하지 않는다. "
        "단순 출력 재사용은 기본보다 부분 MAC이 많으므로 모든 비용 축에서 우월하다는 뜻도 아니다. "
        "후속 후보를 제안하는 분석이며 기본 checkpoint나 K를 결과에 맞춰 변경하지 않았다.", "",
        "## 완료 상태와 남은 한계", "",
        "5~8의 개발 분석을 완료했다. 신규 학습·최종 test 평가·외부 제출은 하지 않았다. "
        "성능 목표 달성 여부는 위 결과로 판단해야 하며, 실험 완료 자체가 정확도 개선이나 투고 준비 완료를 뜻하지 않는다.", "",
        "시퀀스는 네 개뿐이며 인접 클립은 독립 표본이 아니다. 다중 학습 seed와 paired sequence CI는 11번에 남겨둔다. "
        "기여 분리 행은 같은 체크포인트의 추론 시 개입이며, 대체 방식을 별도로 재학습했을 때의 최선 성능 비교가 아니다. "
        "TCE/OPW의 가림 처리는 in-bounds와 GT validity warp만 사용하며 forward/backward flow 일치성 검사는 없다. "
        "미래 프레임을 사용하는 비디오 baseline 전체 비교와 다른 도메인 일반화도 이 표의 범위가 아니다.", "",
        "DPT loader의 누락된 네 파라미터는 첫 fusion layer의 사용되지 않는 residual branch에 해당한다. "
        "그 모듈이 실행되면 중단하는 hook을 설치했고, 전체 평가가 그 검사 아래 완료됐다. "
        "로딩 정보와 실제 라이브러리 소스 해시는 `baseline_loading.json`에 있다.", "",
        "재실행:", "", "```bash", "python scripts/paper_study.py --out work_dirs/paper_study_5_8",
        "python scripts/paper_study_report.py --out work_dirs/paper_study_5_8", "```", ""]
    Path("paper/STUDY_5_8.md").write_text("\n".join(doc))
    outputs = [Path("paper/STUDY_5_8.md"), Path("paper/figures/study7_long.svg"), *sorted(tables.glob("study[5-8]_*.csv")),
               tables / "study7_frame_curves.json"]
    (directory / "report_artifacts.json").write_text(json.dumps({"generator": file_record(__file__),
        "inputs": [file_record(directory / p) for p in ("study.json", "accuracy_complete.json",
                   "long_complete.json", "bench_complete.json", "baseline_loading.json")],
        "artifacts": [file_record(p) for p in outputs]}, indent=2) + "\n")
    print("created paper/STUDY_5_8.md and", len(outputs) - 1, "supporting tables/curve artifacts")


if __name__ == "__main__":
    main()
