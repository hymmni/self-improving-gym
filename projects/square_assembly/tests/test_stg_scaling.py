"""STG 롤아웃 규모 실험(2026-11-20): 평가용 목록, 캐시 조립, 부분집합 선택, 평가 지표, 실행 묶음."""

import json

import h5py
import numpy as np
import pytest

from square_assembly.datasets.dino_feature_dataset import DinoFeatureWindows


def test_pick_heldout_spreads_over_collection_order():
    from square_assembly.scripts.stg_scaling_data import pick_heldout
    names = [f"demo_{k}" for k in range(30)]
    np.random.RandomState(1).shuffle(names)
    picked = pick_heldout(names, 6, seed=0)
    blocks = sorted(int(n.split("_")[1]) // 5 for n in picked)
    assert blocks == [0, 1, 2, 3, 4, 5]                      # 수집 순서 구간마다 정확히 하나(숫자 순 정렬)
    assert picked == pick_heldout(sorted(names), 6, seed=0)  # 입력 순서와 무관, 같은 seed면 같다


def _fake_source(path, episodes, meta=None):
    """출처별 캐시(feat, lowdim)와 원본 hdf5(action_mode, attrs)를 같은 이름으로 만든다."""
    cache, hdf5 = path.with_suffix(".vip.h5"), path.with_suffix(".hdf5")
    meta = meta or {"model": "vip", "crop": 76, "size": 224, "rgb_keys": ["agentview_image", "robot0_eye_in_hand_image"],
                    "lowdim_keys": ["robot0_eef_pos"]}
    with h5py.File(cache, "w") as c, h5py.File(hdf5, "w") as h:
        for name, (modes, attrs) in episodes.items():
            length = len(modes)
            g = c.create_group(f"data/{name}")
            g["feat"] = np.full((length, 4), float(name.split("_")[1]), np.float16)
            g["lowdim"] = np.zeros((length, 2), np.float32)
            g.attrs["length"], g.attrs["is_success"] = length, bool(attrs.get("is_success", True))
            s = h.create_group(f"data/{name}")
            if modes is not None and not all(m == -1 for m in modes):
                s["action_mode"] = modes
            s["actions"] = np.zeros((length, 7))
            s.attrs.update(attrs)
        c.attrs["meta"] = json.dumps(meta)
    return str(cache), str(hdf5)


def test_assemble_cache_keeps_source_names_modes_and_rejects_mixed_meta(tmp_path):
    from square_assembly.scripts.stg_scaling_data import assemble_cache
    intv_c, intv_h = _fake_source(tmp_path / "intv", {"demo_7": ([0, 0, -10, 1, 1], {"is_success": True})})
    br_c, br_h = _fake_source(tmp_path / "branch", {"demo_7": ([0] * 6, {"is_success": False, "hacky": False,
                                                                         "branch_step": 2})})
    demo_c, demo_h = _fake_source(tmp_path / "demo", {"demo_0": ([-1] * 3, {"is_success": True})})
    out = tmp_path / "out.vip.h5"
    assemble_cache(str(out), [dict(cache=demo_c, hdf5=demo_h, name="demo_0", role="demo"),
                              dict(cache=intv_c, hdf5=intv_h, name="demo_7", role="intv_delay"),
                              dict(cache=br_c, hdf5=br_h, name="demo_7", role="branch_t0")])
    with h5py.File(out) as f:
        got = {n: dict(f["data"][n].attrs) for n in f["data"]}
        assert [got[n]["source_name"] for n in ("demo_0", "demo_1", "demo_2")] == ["demo_0", "demo_7", "demo_7"]
        assert [got[n]["role"] for n in ("demo_0", "demo_1", "demo_2")] == ["demo", "intv_delay", "branch_t0"]
        assert [int(got[n]["score_from"]) for n in ("demo_0", "demo_1", "demo_2")] == [0, 3, 2]
        assert not got["demo_2"]["is_success"] and not got["demo_2"]["hacky"]
        np.testing.assert_array_equal(f["data/demo_0/action_mode"][:], [-1] * 3)
        np.testing.assert_array_equal(f["data/demo_1/action_mode"][:], [0, 0, -10, 1, 1])
        assert json.loads(f.attrs["meta"])["rgb_keys"] == ["agentview_image", "robot0_eye_in_hand_image"]
    ds = DinoFeatureWindows(str(out), 2, fail_bin=99, mode_hdf5=str(out), strict_modes=True)
    assert len(ds) == 3 + 5 + 6

    other_c, other_h = _fake_source(tmp_path / "other", {"demo_1": ([-1] * 2, {})},
                                    meta={"model": "vip", "crop": 76, "size": 224, "lowdim_keys": ["robot0_eef_pos"],
                                          "rgb_keys": ["robot0_eye_in_hand_image", "agentview_image"]})
    with pytest.raises(ValueError, match="meta"):
        assemble_cache(str(tmp_path / "bad.vip.h5"), [dict(cache=demo_c, hdf5=demo_h, name="demo_0", role="demo"),
                                                      dict(cache=other_c, hdf5=other_h, name="demo_1", role="demo")])


def _fake_plan(num_steps=100):
    """개입 demo_0..39(공통 demo_0..34), 정책 단독 성공 demo_40..49, 평가용 개입 demo_0·demo_10, 성공 demo_40."""
    from square_assembly.scripts.stg_scaling_sweep import plan_runs
    intv = [f"demo_{k}" for k in range(40)]
    pools = {"intv": intv, "solo": [f"demo_{k}" for k in range(40, 50)], "immediate": intv,
             "timeout": intv[:35], "common": intv[:35]}
    heldout = {"intervention": ["demo_0", "demo_10"], "policy_success": ["demo_40"]}
    caches = {}
    for kind in ("delay", "immediate", "timeout"):
        names = pools["intv" if kind == "delay" else kind]
        caches[kind] = {"path": f"c/{kind}.vip.h5", "demos": ["demo_0", "demo_1"],
                        "intv": {n: f"intv_{n}" for n in names}, "solo": {n: f"solo_{n}" for n in pools["solo"]}}
    return plan_runs("random30", pools, heldout, caches, num_steps, root="R")


def test_selection_nests_and_keeps_intervention_sets_paired():
    from square_assembly.scripts.stg_scaling_sweep import order_names, select
    pool = [f"demo_{k}" for k in range(30)]
    order, solo, eligible = order_names(pool, 3), {f"demo_{k}" for k in range(25, 30)}, set(pool) - {"demo_7"}
    small, big = select(order, eligible, solo, 10, True), select(order, eligible, solo, 20, True)
    assert big[:10] == small and "demo_7" not in big
    assert select(order, eligible, solo, 20, False) == [x for x in big if x not in solo]
    jobs, runs = _fake_plan()
    by = {}
    for r in runs:
        if r["pairing"] == "same" and r["obs_parts"] == "all":
            by.setdefault((r["n"], r["order"], r["seed"], r["with_success"]), set()).add(tuple(r["rollouts"]))
    assert all(len(v) == 1 for v in by.values())          # 같은 에피소드 조건이면 세 개입 시점이 같은 원본 에피소드


def test_n_grid_ends_at_the_pool():
    from square_assembly.scripts.stg_scaling_sweep import n_grid
    assert n_grid(68) == [0, 10, 20, 30, 40, 50, 60, 68]
    assert n_grid(60) == [0, 10, 20, 30, 40, 50, 60]


def test_build_jobs_counts_and_dedupes():
    from square_assembly.scripts.stg_scaling_sweep import n_grid
    jobs, runs = _fake_plan()
    same, full = 33 + 9, 38 + 9   # 공통 35 - 평가 2 + 성공 9 / 개입 40 - 2 + 9
    expected = (6 * len(n_grid(same)) + 4 * len(n_grid(full)) + 6 * len(n_grid(same))) * 10
    assert len(runs) == expected
    ids = [j["id"] for j in jobs]
    assert len(ids) == len(set(ids)) and len(jobs) < len(runs)
    zero = {r["job"] for r in runs if r["n"] == 0 and r["obs_parts"] == "all"}
    assert len(zero) == 10                                  # n=0은 조건끼리 공유, 시드 10개
    job = {j["id"]: j for j in jobs}
    assert all(job[r["job"]]["train_episodes"][:2] == ["demo_0", "demo_1"] for r in runs)
    assert all({"demo_0", "demo_10", "demo_40"}.isdisjoint(r["rollouts"]) for r in runs)
    assert [j["priority"] for j in jobs] == sorted(j["priority"] for j in jobs)   # 주 곡선이 먼저
    assert {j["num_bins"] for j in jobs} == {522} and {j["num_steps"] for j in jobs} == {100}
