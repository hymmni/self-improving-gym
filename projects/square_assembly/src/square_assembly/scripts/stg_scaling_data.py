r"""STG 롤아웃 규모 실험(2026-11-20)의 평가용 목록과 VIP 캐시를 만든다.

스펙: docs/superpowers/specs/2026-11-20-stg-rollout-scaling-design.md, 플랜: docs/superpowers/plans/2026-11-20-stg-rollout-scaling.md

- heldout: 정책마다 개입 15개(세 조건 공통 에피소드) + 정책 단독 성공 6개를 수집 순서 구간마다 하나씩 뽑아
  configs/stg_scaling/heldout_{policy}.json에 고정한다. 이미 있으면 다시 뽑아 같은지만 확인한다.
- caches: 출처별 VIP 캐시(cache_vip_feats.py로 `{root}/caches/src/<hdf5 이름>.vip.h5`에 미리 굽는다)를 이어 붙여
  조건별 학습 캐시 `{root}/caches/{policy}_train_{kind}.vip.h5`(데모 30 + 그 조건의 개입 + 정책 단독 성공, 평가용 제외)와
  평가 캐시 `{root}/caches/{policy}_eval.vip.h5`를 만든다. 그룹마다 action_mode를 넣어 캐시가 mode_hdf5를 겸하고,
  attrs에 원본 이름(source_name)·역할(role)·채점 시작 프레임(score_from)을 남긴다.

    /opt/venvs/torch/bin/python -m square_assembly.scripts.stg_scaling_data heldout --policy random30
    /opt/venvs/torch/bin/python -m square_assembly.scripts.stg_scaling_data caches --policy random30 \
        --root outputs/stg_rollout_scaling_20261120
"""

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np

from square_assembly.datasets.labels import LABEL_DEMO, LABEL_INTV

SOURCES = {
    "random30": dict(
        demos="data/square_quest_random30.hdf5",
        delay="data/square_quest_random30_delay_intv.hdf5",
        immediate="data/square_quest_random30_immediate.hdf5",
        timeout="data/square_quest_random30_timeout.hdf5",
        branch_t0="data/square_quest_random30_branch_t0.hdf5",
        branch_t_press="data/square_quest_random30_branch_t_press.hdf5",
        eval_demos="data/quest_square_demos.hdf5",
        eval_demo_names=[f"demo_{k}" for k in range(30, 50)],   # demo_0~29는 정책 학습 데모
    ),
    "turn30": dict(
        demos="data/square_quest_turn30.hdf5",
        delay="data/square_quest_turn30_delay_intv.hdf5",
        immediate="data/square_quest_turn30_immediate.hdf5",
        timeout="data/square_quest_turn30_timeout.hdf5",
        branch_t0="data/square_quest_turn30_branch_t0.hdf5",
        branch_t_press="data/square_quest_turn30_branch_t_press.hdf5",
        eval_demos="data/quest_square_turn_eval_demos.hdf5",
        eval_demo_names=None,   # 파일 전체(20개)
    ),
}
KINDS = ("delay", "immediate", "timeout")
N_HELDOUT_INTV, N_HELDOUT_SOLO = 15, 6
HELDOUT_DIR = Path(__file__).resolve().parents[1] / "configs" / "stg_scaling"
META_KEYS = ("model", "embed_dim", "crop", "size", "rgb_keys", "lowdim_keys")


def by_index(names):
    """demo_k를 k 순으로(문자열 순이면 demo_10이 demo_2 앞에 온다)."""
    return sorted(names, key=lambda n: int(n.rsplit("_", 1)[1]))


def pick_heldout(names, k, seed=0):
    """수집 순서로 정렬해 k개 연속 구간으로 나누고, 구간마다 하나씩 뽑는다 — 수집 순서가 고르게 섞인다."""
    rng = np.random.RandomState(seed)
    return [str(block[rng.randint(len(block))]) for block in np.array_split(np.array(by_index(names)), k)]


def _successes(path):
    with h5py.File(path, "r") as f:
        return {n for n in f["data"] if bool(f["data"][n].attrs.get("is_success", True))}


def rollout_pools(policy):
    """지연 파일의 개입(intv)·정책 단독 성공(solo), 즉시·타임아웃 파일의 이름, 세 조건 공통(common)."""
    src = SOURCES[policy]
    intv, solo = [], []
    with h5py.File(src["delay"], "r") as f:
        for name in f["data"]:
            g = f["data"][name]
            if not bool(g.attrs.get("is_success", True)):
                continue
            (intv if (np.asarray(g["action_mode"]) == LABEL_INTV).any() else solo).append(name)
    immediate, timeout = _successes(src["immediate"]), _successes(src["timeout"])
    common = set(intv) & immediate & timeout
    return {"intv": by_index(intv), "solo": by_index(solo), "immediate": by_index(immediate),
            "timeout": by_index(timeout), "common": by_index(common)}


def heldout_path(policy):
    return HELDOUT_DIR / f"heldout_{policy}.json"


def make_heldout(policy):
    pools = rollout_pools(policy)
    return {"policy": policy, "intervention": pick_heldout(pools["common"], N_HELDOUT_INTV),
            "policy_success": pick_heldout(pools["solo"], N_HELDOUT_SOLO), "seed": 0,
            "method": "demo_k의 k 순으로 정렬 -> np.array_split(k) -> 구간마다 RandomState(0)로 하나 "
                      "(개입은 지연·즉시·타임아웃 공통 에피소드에서)"}


def load_heldout(policy):
    return json.loads(heldout_path(policy).read_text())


def _score_from(role, mode, attrs):
    if role.startswith("intv"):
        return int(np.flatnonzero(mode == LABEL_INTV)[0])
    if role.startswith("branch"):
        return int(attrs["branch_step"])
    return 0


def assemble_cache(out, entries):
    """entries = [{"cache", "hdf5", "name", "role"}] 순서대로 그룹 demo_0.. 에 이어 붙인다.

    feat·lowdim은 출처별 캐시에서, action_mode·is_success·(분기면) hacky·branch_step은 원본 hdf5에서 가져온다.
    출처 캐시의 인코더 메타(카메라 순서·crop·size 등)가 하나라도 다르면 특징이 섞이므로 거부한다.
    """
    ref = None
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with h5py.File(out, "w") as dst:
        for i, e in enumerate(entries):
            with h5py.File(e["cache"], "r") as c, h5py.File(e["hdf5"], "r") as h:
                meta = json.loads(c.attrs["meta"])
                key = {k: meta.get(k) for k in META_KEYS}
                ref = ref or key
                if key != ref:
                    raise ValueError(f"cache meta differs: {e['cache']} {key} != {ref}")
                g, src = c["data"][e["name"]], h["data"][e["name"]]
                length = int(g.attrs["length"])
                mode = (np.asarray(src["action_mode"]) if "action_mode" in src
                        else np.full(length, LABEL_DEMO, dtype=np.int64))
                if len(mode) != length:
                    raise ValueError(f"{e['hdf5']}:{e['name']} action_mode length {len(mode)} != cached {length}")
                og = dst.create_group(f"data/demo_{i}")
                og["feat"], og["lowdim"], og["action_mode"] = g["feat"][:], g["lowdim"][:], mode
                og.attrs.update(length=length, is_success=bool(src.attrs.get("is_success", True)),
                                source=e["hdf5"], source_name=e["name"], role=e["role"],
                                score_from=_score_from(e["role"], mode, src.attrs))
                for k in ("hacky", "branch_step"):
                    if k in src.attrs:
                        og.attrs[k] = src.attrs[k]
        dst.attrs["meta"] = json.dumps({**ref, "assembled_from": sorted({e["cache"] for e in entries})})


def source_cache(root, hdf5):
    return os.path.join(root, "caches", "src", Path(hdf5).stem + ".vip.h5")


def train_cache_path(root, policy, kind):
    return os.path.join(root, "caches", f"{policy}_train_{kind}.vip.h5")


def eval_cache_path(root, policy):
    return os.path.join(root, "caches", f"{policy}_eval.vip.h5")


def build_caches(policy, root):
    src, pools, held = SOURCES[policy], rollout_pools(policy), load_heldout(policy)
    held_names = set(held["intervention"]) | set(held["policy_success"])

    def entries(key, names, role):
        return [dict(cache=source_cache(root, src[key]), hdf5=src[key], name=n, role=role) for n in names]

    with h5py.File(src["demos"], "r") as f:
        demos = entries("demos", by_index(f["data"]), "demo")
    solo = entries("delay", [n for n in pools["solo"] if n not in held_names], "solo")
    for kind in KINDS:
        names = pools["intv" if kind == "delay" else kind]
        train = demos + entries(kind, [n for n in names if n not in held_names], "intv") + solo
        assert not {e["name"] for e in train if e["role"] != "demo"} & held_names, "held-out leaked into training"
        assemble_cache(train_cache_path(root, policy, kind), train)
        print(f"{policy} train_{kind}: demos {len(demos)} intv {len(train) - len(demos) - len(solo)} "
              f"solo {len(solo)}", flush=True)

    with h5py.File(src["eval_demos"], "r") as f:
        eval_demo_names = src["eval_demo_names"] or by_index(f["data"])
    ev = entries("eval_demos", eval_demo_names, "demo") + entries("delay", held["policy_success"], "policy_success")
    for kind in KINDS:
        ev += entries(kind, held["intervention"], f"intv_{kind}")
    for kind in ("branch_t0", "branch_t_press"):
        ev += entries(kind, held["intervention"], kind)
    assemble_cache(eval_cache_path(root, policy), ev)
    print(f"{policy} eval: {len(ev)} episodes", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["heldout", "caches"])
    ap.add_argument("--policy", choices=sorted(SOURCES), required=True)
    ap.add_argument("--root", default="outputs/stg_rollout_scaling_20261120")
    args = ap.parse_args()
    if args.command == "heldout":
        new, path = make_heldout(args.policy), heldout_path(args.policy)
        if path.exists() and json.loads(path.read_text()) != new:
            raise SystemExit(f"{path}가 이미 있고 다시 뽑은 목록과 다르다 — 고정 목록이므로 덮어쓰지 않는다")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(new, indent=2, ensure_ascii=False) + "\n")
        pools = rollout_pools(args.policy)
        print(json.dumps({k: len(v) for k, v in pools.items()}), json.dumps(new, ensure_ascii=False), sep="\n")
    else:
        build_caches(args.policy, args.root)


if __name__ == "__main__":
    main()
