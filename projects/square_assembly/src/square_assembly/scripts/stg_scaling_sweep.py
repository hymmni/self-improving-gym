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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--root", default="outputs/stg_rollout_scaling_20261120")
    p.add_argument("--num-steps", type=int, required=True)
    args = ap.parse_args()
    if args.command == "plan":
        jobs, runs = build_jobs(args.root, args.num_steps)
        Path(args.root, "jobs.json").write_text(json.dumps({"num_steps": args.num_steps, "jobs": jobs, "runs": runs}))
        print(f"runs {len(runs)} -> jobs {len(jobs)} "
              f"(main {sum(j['priority'] == 0 for j in jobs)}, inputs {sum(j['priority'] == 1 for j in jobs)})")


if __name__ == "__main__":
    main()
