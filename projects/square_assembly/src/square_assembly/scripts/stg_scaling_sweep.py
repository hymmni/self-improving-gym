r"""STG 롤아웃 규모 실험(2026-11-20) 스윕: job 목록 만들기, 기계별로 나눠 돌리기, 결과 모으기.

스펙: docs/superpowers/specs/2026-11-20-stg-rollout-scaling-design.md, 플랜: docs/superpowers/plans/2026-11-20-stg-rollout-scaling.md
캐시와 평가용 목록은 stg_scaling_data.py가 먼저 만든다.

- 논리 실행(run) = (정책, 개입 시점 kind, 짝 pairing, 성공 포함, 입력, n, 순서 order, 시드). 조건·n마다 순서 5 × 시드 2.
- job = 실제로 학습할 것. 같은 캐시·입력·시드·학습 에피소드면 run 여럿이 job 하나를 공유한다(n=0, 아직 추가 에피소드가
  안 들어온 (나), 뽑힌 n개에 성공이 없는 성공 제외 조건).
- job 순서: 주 곡선(입력 all) 먼저, 입력 비교는 나중. 그 안에서는 시드 슬롯 순 — 중간에 끊겨도 시드 수만 적은 완전한 곡선이 남는다.

    PY=/opt/venvs/torch/bin/python
    $PY -m square_assembly.scripts.stg_scaling_sweep plan --root outputs/stg_rollout_scaling_20261120 --num-steps S
    $PY -m square_assembly.scripts.stg_scaling_sweep run --root ... --shards 0 1 --num-shards 3 --workers 12 \
        --no-new-after 2026-11-21T09:50+09:00
    $PY -m square_assembly.scripts.stg_scaling_sweep aggregate --root ...

run은 metrics.json이 있는 job을 건너뛰므로, 다른 기계의 runs/를 합친 뒤 다시 실행하면 남은 job만 돈다.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from square_assembly.scripts.stg_scaling_data import KINDS, by_index, eval_cache_path, train_cache_path

POLICY_BINS = {"random30": 522, "turn30": 888}   # 그 정책 학습 데모 최대 길이 × 2
OBS_COMPARE = ("feat", "lowdim", "agentview", "wrist", "wrist_lowdim", "agentview_lowdim")
N_ORDERS, N_SEEDS = 5, 2


def conditions():
    """(kind, pairing, with_success, obs_parts, priority). 주 곡선 10개 + 입력 비교 6개."""
    out = []
    for kind in KINDS:
        for pairing in (("same",) if kind == "timeout" else ("same", "all")):
            for with_success in (True, False):
                out.append((kind, pairing, with_success, "all", 0))
    out += [("delay", "same", True, obs, 1) for obs in OBS_COMPARE]
    return out


def order_names(pool, order):
    names = by_index(pool)
    return [names[i] for i in np.random.RandomState(order).permutation(len(names))]


def select(order, eligible, solo, n, with_success):
    chosen = [x for x in order if x in eligible][:n]
    return chosen if with_success else [x for x in chosen if x not in solo]


def n_grid(pool_size):
    return sorted(set(range(0, pool_size, 10)) | {pool_size})


def plan_runs(policy, pools, heldout, caches, num_steps, root):
    """pools = stg_scaling_data.rollout_pools, caches[kind] = {"path", "demos", "intv": {원본: 캐시 이름}, "solo": {...}}."""
    held = set(heldout["intervention"]) | set(heldout["policy_success"])
    solo = set(pools["solo"]) - held
    eligible_by = {("delay", "all"): set(pools["intv"]) - held, ("immediate", "all"): set(pools["immediate"]) - held}
    common = set(pools["common"]) - held
    full = eligible_by[("delay", "all")] | solo
    orders = [order_names(full, o) for o in range(N_ORDERS)]
    jobs, runs = {}, []
    for kind, pairing, with_success, obs, priority in conditions():
        eligible = (common if pairing == "same" else eligible_by[(kind, pairing)]) | solo
        assert eligible <= full
        for n in n_grid(len(eligible)):
            for o in range(N_ORDERS):
                for s in range(N_SEEDS):
                    seed = N_SEEDS * o + s
                    rollouts = select(orders[o], eligible, solo, n, with_success)
                    cache_kind = kind if n else "delay"   # n=0(데모만)은 조건끼리 같은 job
                    c = caches[cache_kind]
                    names = c["demos"] + sorted(c["solo" if r in solo else "intv"][r] for r in rollouts)
                    key = json.dumps([policy, cache_kind, obs, seed, sorted(names)])
                    digest = hashlib.sha1(key.encode()).hexdigest()[:10]
                    job_id = f"{policy}/{obs}/{cache_kind}/r{len(rollouts):02d}_s{seed}_{digest}"
                    job = jobs.setdefault(job_id, {
                        "id": job_id, "policy": policy, "kind": cache_kind, "cache": c["path"],
                        "eval_cache": eval_cache_path(root, policy), "obs_parts": obs, "seed": seed,
                        "num_bins": POLICY_BINS[policy], "num_steps": num_steps, "n_rollouts": len(rollouts),
                        "train_episodes": names, "priority": priority})
                    job["priority"] = min(job["priority"], priority)
                    runs.append({"policy": policy, "kind": kind, "pairing": pairing, "with_success": with_success,
                                 "obs_parts": obs, "n": n, "order": o, "seed": seed, "job": job_id,
                                 "rollouts": rollouts})
    return sorted(jobs.values(), key=lambda j: (j["priority"], j["seed"], j["id"])), runs


def read_cache_names(path):
    import h5py

    out = {"path": path, "demos": [], "intv": {}, "solo": {}}
    with h5py.File(path, "r") as f:
        for name in by_index(f["data"]):
            a = f["data"][name].attrs
            if a["role"] == "demo":
                out["demos"].append(name)
            else:
                out[str(a["role"])][str(a["source_name"])] = name
    return out


def build_jobs(root, num_steps):
    from square_assembly.scripts.stg_scaling_data import load_heldout, rollout_pools

    jobs, runs = [], []
    for policy in POLICY_BINS:
        caches = {k: read_cache_names(train_cache_path(root, policy, k)) for k in KINDS}
        j, r = plan_runs(policy, rollout_pools(policy), load_heldout(policy), caches, num_steps, root)
        jobs += j
        runs += r
    return sorted(jobs, key=lambda j: (j["priority"], j["seed"], j["id"])), runs


def shard(jobs, shards, num_shards):
    """우선순위 순서를 지킨 채 i % num_shards ∈ shards인 job만. 기계별 몫은 shard 개수로 조절한다."""
    return [j for i, j in enumerate(jobs) if i % num_shards in set(shards)]


def _src_dir():
    return str(Path(__file__).resolve().parents[2])


def run_job(job, root, python, threads=1, deadline=None):
    """학습(train_dstg_vip) -> 평가(eval_stg_scaling). metrics.json이 있으면 건너뛰고, 실패하면 남기지 않는다."""
    import datetime
    import subprocess
    import time

    out = Path(root) / "runs" / job["id"]
    if (out / "metrics.json").exists():
        return "skipped"
    if deadline is not None and datetime.datetime.now().astimezone() >= deadline:
        return "deadline"
    out.mkdir(parents=True, exist_ok=True)
    (out / "job.json").write_text(json.dumps(job, indent=1))
    (out / "train_episodes.txt").write_text("\n".join(job["train_episodes"]) + "\n")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [_src_dir(), os.environ.get("PYTHONPATH")])),
               OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    ckpt = out / "predictor.pt"
    train = [python, "-m", "square_assembly.scripts.train_dstg_vip", f"cache_path={job['cache']}",
             f"mode_hdf5={job['cache']}", f"out={ckpt}", f"num_bins_override={job['num_bins']}",
             "success_tail_only=true", "val_fraction=0", f"num_steps={job['num_steps']}", f"seed={job['seed']}",
             f"obs_parts={job['obs_parts']}", f"train_episodes_file={out / 'train_episodes.txt'}", "use_wandb=false",
             f"hydra.run.dir={out / 'hydra'}"]
    evaluate = [python, "-m", "square_assembly.scripts.eval_stg_scaling", "--predictor", str(ckpt),
                "--eval-cache", job["eval_cache"], "--out", str(out)]
    import socket

    timing = {"start": time.time(), "host": socket.gethostname()}   # 어느 기계에서 돌았는지(로컬/Eddy) 구분용
    for name, cmd in (("train", train), ("eval", evaluate)):
        if name == "train" and ckpt.exists():
            continue
        t0 = time.time()
        with open(out / f"{name}.log", "w") as log:
            if subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env).returncode:
                return "failed"
        timing[name] = time.time() - t0
    timing["end"] = time.time()
    (out / "timing.json").write_text(json.dumps(timing))
    return "done"


def run_jobs(jobs, root, workers, python, threads=1, deadline=None):
    """workers개를 동시에 돌린다. deadline(aware datetime) 이후엔 새 job을 시작하지 않는다(돌던 것은 끝까지)."""
    import threading
    from collections import Counter
    from concurrent.futures import ThreadPoolExecutor

    counts, lock = Counter(), threading.Lock()

    def one(job):
        status = run_job(job, root, python, threads, deadline)
        with lock:
            counts[status] += 1
            if status in ("done", "failed"):
                print(f"[{sum(counts.values())}/{len(jobs)}] {status} {job['id']}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, jobs))
    return dict(counts)


def _parse_deadline(text):
    import datetime

    t = datetime.datetime.fromisoformat(text)
    return t if t.tzinfo else t.astimezone()   # 시간대가 없으면 이 기계의 현지 시각


GROUP = ("policy", "kind", "pairing", "with_success", "obs_parts", "n")


def aggregate(root, n_boot=1000, plots=True):
    """논리 실행을 GROUP으로 묶어 예측기 평균·시드 표준편차·부트스트랩 구간을 summary.csv/json으로(+그림)."""
    import csv

    from square_assembly.scripts.eval_stg_scaling import RATIOS, bootstrap_ci, ratio_metrics

    plan = json.loads(Path(root, "jobs.json").read_text())
    groups = {}
    for r in plan["runs"]:
        groups.setdefault(tuple(r[k] for k in GROUP), []).append(r)
    cache, rows = {}, []
    for key, runs in sorted(groups.items(), key=lambda kv: [str(v) for v in kv[0]]):
        ms = []
        for r in runs:
            if r["job"] not in cache:
                path = Path(root, "runs", r["job"], "metrics.json")
                cache[r["job"]] = json.loads(path.read_text()) if path.exists() else None
            if cache[r["job"]] is not None:
                ms.append(cache[r["job"]])
        if not ms:
            continue
        row = dict(zip(GROUP, key), n_predictors=len(ms), n_runs=len(runs),
                   n_rollouts_mean=float(np.mean([len(r["rollouts"]) for r in runs])),
                   n_train_samples_mean=float(np.mean([m["n_train_samples"] for m in ms])))
        ci = bootstrap_ci([m["units"] for m in ms], n_boot=n_boot)
        points = [ratio_metrics(m["units"]) for m in ms]
        for name in RATIOS:
            vals = [p[name] for p in points if p[name] is not None]
            row[f"{name}_mean"] = float(np.mean(vals)) if vals else None
            row[f"{name}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
            row[f"{name}_lo"], row[f"{name}_hi"] = ci[name] if ci[name] else (None, None)
        extra = {"auroc_fail_vs_intv": lambda f: f.get("auroc_fail_vs_intv"),
                 **{f"{g}_k{k}_{m}": (lambda f, g=g, k=k, m=m: f.get("reward", {}).get(g, {}).get(f"k{k}", {}).get(m))
                    for g in ("policy_success", "intv", "demo") for k in (1, 8)
                    for m in ("reward_sign_acc", "spearman")}}
        for name, get in extra.items():
            vals = [v for v in (get(m["frame"]) for m in ms) if v is not None and np.isfinite(v)]
            row[f"{name}_mean"] = float(np.mean(vals)) if vals else None
            row[f"{name}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
        rows.append(row)
    Path(root, "summary.json").write_text(json.dumps(rows, indent=1))
    with open(Path(root, "summary.csv"), "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else list(GROUP))
        writer.writeheader()
        writer.writerows(rows)
    if plots and rows:
        plot_curves(rows, Path(root, "figures"))
    return rows


SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6")   # dataviz 기본 순서
METRICS = (("success_mae", "(1) success-path MAE [steps]"), ("false_progress", "(2) false-progress rate"))


def plot_curves(rows, out_dir):
    """정책마다 주 곡선(개입 시점 × 성공), 짝 어블레이션, 입력 비교, 학습 프레임 수 x축. 오차 막대 = 부트스트랩 95%."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    pick = lambda **kw: [r for r in rows if all(r[k] == v for k, v in kw.items())]   # noqa: E731
    for policy in sorted({r["policy"] for r in rows}):
        figs = {
            "main": [(f"{kind} {'+success' if ws else '-success'}", SERIES[i], "-" if ws else "--",
                      pick(policy=policy, kind=kind, pairing="same", with_success=ws, obs_parts="all"))
                     for i, kind in enumerate(KINDS) for ws in (True, False)],
            "pairing": [(f"{kind} {pairing}", SERIES[i], "-" if pairing == "same" else "--",
                         pick(policy=policy, kind=kind, pairing=pairing, with_success=True, obs_parts="all"))
                        for i, kind in enumerate(KINDS[:2]) for pairing in ("same", "all")],
            "inputs": [(obs, SERIES[i], "-", pick(policy=policy, kind="delay", pairing="same", with_success=True,
                                                  obs_parts=obs))
                       for i, obs in enumerate(("all",) + OBS_COMPARE)],
        }
        for fig_name, series in figs.items():
            for x_key in ("n", "n_train_samples_mean"):
                if x_key != "n" and fig_name != "main":
                    continue
                fig, axes = plt.subplots(1, 2, figsize=(11, 4))
                for ax, (metric, title) in zip(axes, METRICS):
                    for label, color, style, rs in series:
                        rs = sorted((r for r in rs if r[f"{metric}_mean"] is not None), key=lambda r: r[x_key])
                        if not rs:
                            continue
                        x = [r[x_key] for r in rs]
                        y = np.array([r[f"{metric}_mean"] for r in rs])
                        lo = np.array([r[f"{metric}_lo"] if r[f"{metric}_lo"] is not None else np.nan for r in rs])
                        hi = np.array([r[f"{metric}_hi"] if r[f"{metric}_hi"] is not None else np.nan for r in rs])
                        ax.errorbar(x, y, yerr=[y - lo, hi - y], color=color, linestyle=style, linewidth=2,
                                    marker="o", markersize=4, capsize=2, label=label)
                    ax.set_title(title, fontsize=10)
                    ax.set_xlabel("rollouts n" if x_key == "n" else "mean training frames")
                    ax.grid(alpha=0.25, linewidth=0.5)
                    ax.spines[["top", "right"]].set_visible(False)
                axes[0].legend(fontsize=8, frameon=False)
                fig.suptitle(f"{policy} — {fig_name}", fontsize=11)
                fig.tight_layout()
                suffix = "" if x_key == "n" else "_frames"
                fig.savefig(out_dir / f"{policy}_{fig_name}{suffix}.png", dpi=120)
                plt.close(fig)


def main():
    import sys

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan", help="jobs.json 만들기")
    p.add_argument("--root", default="outputs/stg_rollout_scaling_20261120")
    p.add_argument("--num-steps", type=int, required=True)
    r = sub.add_parser("run", help="job 돌리기(끝난 것은 건너뜀)")
    r.add_argument("--root", default="outputs/stg_rollout_scaling_20261120")
    r.add_argument("--shards", type=int, nargs="+", default=None, help="이 기계가 맡을 shard 번호들(기본: 전부)")
    r.add_argument("--num-shards", type=int, default=1)
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--threads", type=int, default=1, help="job마다 OMP/MKL 스레드 수")
    r.add_argument("--no-new-after", default=None, help="이 시각 이후 새 job을 시작하지 않는다(ISO, 예 2026-11-21T09:50+09:00)")
    r.add_argument("--only-ids", nargs="+", default=None)
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--python", default=sys.executable)
    a = sub.add_parser("aggregate", help="summary.csv/json과 figures/")
    a.add_argument("--root", default="outputs/stg_rollout_scaling_20261120")
    a.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    if args.command == "plan":
        jobs, runs = build_jobs(args.root, args.num_steps)
        Path(args.root, "jobs.json").write_text(json.dumps({"num_steps": args.num_steps, "jobs": jobs, "runs": runs}))
        print(f"runs {len(runs)} -> jobs {len(jobs)} "
              f"(main {sum(j['priority'] == 0 for j in jobs)}, inputs {sum(j['priority'] == 1 for j in jobs)})")
    elif args.command == "run":
        jobs = json.loads(Path(args.root, "jobs.json").read_text())["jobs"]
        if args.shards is not None:
            jobs = shard(jobs, args.shards, args.num_shards)
        if args.only_ids:
            jobs = [j for j in jobs if j["id"] in set(args.only_ids)]
        jobs = jobs[: args.limit] if args.limit else jobs
        deadline = _parse_deadline(args.no_new_after) if args.no_new_after else None
        print(f"{len(jobs)} jobs, workers={args.workers}, no new jobs after {deadline}", flush=True)
        print(run_jobs(jobs, args.root, args.workers, args.python, args.threads, deadline), flush=True)
    else:
        rows = aggregate(args.root, args.n_boot)
        print(f"{len(rows)} groups -> {args.root}/summary.csv, figures/")


if __name__ == "__main__":
    main()
