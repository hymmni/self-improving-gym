r"""STG 롤아웃 규모 실험: 같은 평가 세트를 쓰는 두 조건의 짝지은 차이와 부트스트랩 95% 구간(2026-11-21).

조건마다의 구간(summary.csv)은 평가 개입 15개·실패 분기 30개라 넓다. 두 조건은 같은 평가 에피소드로 채점되므로,
같은 (order, seed) 예측기끼리 짝을 짓고 양쪽에 같은 층별 재추출(eval_stg_scaling.bootstrap_values)을 적용해
"A − B"의 분포를 낸다. 평가 에피소드를 고르는 운이 양쪽에 똑같이 들어가 상쇄된다.

- 지표: ① success_mae, ② false_progress(8프레임 창). diff = 짝마다 (A − B)의 평균, [lo, hi] = 재추출마다 짝 평균 차이의 2.5/97.5%.
- excludes_zero: 구간이 0을 포함하지 않는 칸.
- 비교 묶음: --set main(개입 시점, 성공 넣기/빼기, (가)−(나), 입력 − all), --set sensitivity(학습 step 민감도: --roots에
  본 스윕과 sensitivity 결과를 같이 준다).

    /opt/venvs/torch/bin/python -m square_assembly.scripts.stg_scaling_paired --set main \
        --roots outputs/stg_rollout_scaling_20261120 --out outputs/stg_rollout_scaling_20261120/paired_main
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from square_assembly.scripts.eval_stg_scaling import bootstrap_values, ratio_metrics
from square_assembly.scripts.stg_scaling_data import KINDS
from square_assembly.scripts.stg_scaling_sweep import OBS_COMPARE, POLICY_BINS, SENS_GRID_STEPS

METRICS = ("success_mae", "false_progress")


def _matches(run, flt):
    return all(run.get(k) == v for k, v in flt.items())


def _resolve(runs, flt):
    """필터의 n="max"를 그 필터에 맞는 실행들의 최대 n으로 바꾼다."""
    if flt.get("n") != "max":
        return flt
    rest = {k: v for k, v in flt.items() if k != "n"}
    return {**rest, "n": max(r["n"] for r in runs if _matches(r, rest))}


def paired_rows(runs, units_of, comparisons, n_boot=1000, seed=0):
    """comparison = {"name", "policy", "a": 필터, "b": 필터, "by": ("n",) 또는 ()}. 필터의 n은 숫자 또는 "max".
    by=("n",)이면 n마다 한 줄, by=()이면 a·b의 n을 "a→b"로 적은 한 줄. units_of(run) -> 그 예측기의 unit_stats."""
    rows = []
    for comp in comparisons:
        base = {"policy": comp["policy"]}
        fa, fb = _resolve(runs, {**base, **comp["a"]}), _resolve(runs, {**base, **comp["b"]})
        side_b = {}
        for r in runs:
            if _matches(r, fb):
                side_b[tuple(r[k] for k in comp["by"]) + (r["order"], r["seed"])] = r
        groups = {}
        for r in runs:
            key = tuple(r[k] for k in comp["by"])
            if _matches(r, fa) and key + (r["order"], r["seed"]) in side_b:
                groups.setdefault(key, []).append((r, side_b[key + (r["order"], r["seed"])]))
        for key, pairs in sorted(groups.items()):
            pairs.sort(key=lambda p: (p[0]["order"], p[0]["seed"]))
            stats = [units_of(a) for a, _ in pairs] + [units_of(b) for _, b in pairs]
            vals, points = bootstrap_values(stats, n_boot, seed), [ratio_metrics(s) for s in stats]
            p = len(pairs)
            label = key[0] if comp["by"] == ("n",) else f"{fa.get('n', '')}→{fb.get('n', '')}"
            for metric in METRICS:
                a = np.array([pt[metric] for pt in points[:p]], float)
                b = np.array([pt[metric] for pt in points[p:]], float)
                with np.errstate(all="ignore"):
                    diffs = np.nanmean(vals[metric][:, :p] - vals[metric][:, p:], axis=1)
                lo, hi = (float(v) for v in np.nanpercentile(diffs, [2.5, 97.5]))
                rows.append({"comparison": comp["name"], "policy": comp["policy"], "n": label, "metric": metric,
                             "n_pairs": p, "a_mean": float(np.nanmean(a)), "b_mean": float(np.nanmean(b)),
                             "diff": float(np.nanmean(a - b)), "lo": lo, "hi": hi,
                             "excludes_zero": bool(lo > 0 or hi < 0)})
    return rows


MAIN = {"pairing": "same", "with_success": True, "obs_parts": "all"}


def main_comparisons(policy):
    comps = [{"name": f"{a}-{b}", "a": {**MAIN, "kind": a}, "b": {**MAIN, "kind": b}, "by": ("n",)}
             for a, b in (("delay", "timeout"), ("delay", "immediate"), ("timeout", "immediate"))]
    for kind in KINDS:
        comps.append({"name": f"{kind}:with-without_success", "a": {**MAIN, "kind": kind},
                      "b": {**MAIN, "kind": kind, "with_success": False}, "by": ("n",)})
    for kind in ("delay", "immediate"):
        same, full = {**MAIN, "kind": kind}, {**MAIN, "kind": kind, "pairing": "all"}
        comps.append({"name": f"{kind}:same-all", "a": same, "b": full, "by": ("n",)})
        comps.append({"name": f"{kind}:same-all", "a": {**same, "n": "max"}, "b": {**full, "n": "max"}, "by": ()})
    for obs in OBS_COMPARE:
        comps.append({"name": f"{obs}-all", "a": {**MAIN, "kind": "delay", "obs_parts": obs},
                      "b": {**MAIN, "kind": "delay"}, "by": ("n",)})
    return [{**c, "policy": policy} for c in comps]


def sensitivity_comparisons(policy):
    comps = []
    arms = [("main", {"arm": "main"}), ("E", {"arm": "E"})] + [(f"G{s}", {"arm": "G", "steps": s})
                                                              for s in SENS_GRID_STEPS]
    for kind in KINDS:
        for name, arm in arms:   # n=0 − 최대 n: 양수면 데이터를 늘려 나아진 것
            comps.append({"name": f"{kind}:{name}:n0-nmax", "a": {**MAIN, **arm, "kind": kind, "n": 0},
                          "b": {**MAIN, **arm, "kind": kind, "n": "max"}, "by": ()})
        comps.append({"name": f"{kind}:E-main", "a": {**MAIN, "arm": "E", "kind": kind},
                      "b": {**MAIN, "arm": "main", "kind": kind}, "by": ("n",)})
        for s in SENS_GRID_STEPS:
            comps.append({"name": f"{kind}:G{s}-main", "a": {**MAIN, "arm": "G", "steps": s, "kind": kind},
                          "b": {**MAIN, "arm": "main", "kind": kind}, "by": ("n",)})
    for a, b in (("delay", "timeout"), ("delay", "immediate"), ("timeout", "immediate")):
        comps.append({"name": f"E:{a}-{b}", "a": {**MAIN, "arm": "E", "kind": a},
                      "b": {**MAIN, "arm": "E", "kind": b}, "by": ("n",)})
    return [{**c, "policy": policy} for c in comps]


def load_runs(roots):
    """루트마다 jobs.json의 실행. arm이 없는 실행(본 스윕)은 arm="main", steps=그 스윕의 num_steps로 둔다."""
    runs = []
    for root in roots:
        plan = json.loads(Path(root, "jobs.json").read_text())
        for r in plan["runs"]:
            runs.append({"arm": "main", "steps": plan["num_steps"], **r,
                         "metrics": str(Path(root, "runs", r["job"], "metrics.json"))})
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--roots", nargs="+", default=["outputs/stg_rollout_scaling_20261120"])
    ap.add_argument("--set", choices=["main", "sensitivity"], default="main")
    ap.add_argument("--out", required=True, help="이 경로에 .csv와 .json을 쓴다(확장자 없이)")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    runs = [r for r in load_runs(args.roots) if Path(r["metrics"]).exists()]
    cache = {}

    def units_of(r):
        if r["metrics"] not in cache:
            cache[r["metrics"]] = json.loads(Path(r["metrics"]).read_text())["units"]
        return cache[r["metrics"]]

    build = main_comparisons if args.set == "main" else sensitivity_comparisons
    rows = paired_rows(runs, units_of, [c for p in POLICY_BINS for c in build(p)], n_boot=args.n_boot)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(f"{args.out}.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    with open(f"{args.out}.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} rows ({sum(r['excludes_zero'] for r in rows)} exclude 0) -> {args.out}.csv/json")


if __name__ == "__main__":
    main()
