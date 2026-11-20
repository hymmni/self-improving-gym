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


def _fake_plan(num_steps=100, plan=None):
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
                        "intv": {n: f"intv_{n}" for n in names}, "solo": {n: f"solo_{n}" for n in pools["solo"]},
                        "frames": {"demo_0": 100, "demo_1": 100, **{f"intv_{n}": 50 for n in names},
                                   **{f"solo_{n}": 20 for n in pools["solo"]}}}
    if plan is not None:
        return plan("random30", pools, heldout, caches, root="R")
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


def _frames(series):
    """{에피소드: (d, label)} -> 평가 함수가 받는 프레임 배열들."""
    d, label, ep, t = [], [], [], []
    for name, (dd, ll) in series.items():
        d += list(dd); label += list(ll); ep += [name] * len(dd); t += list(range(len(dd)))
    return np.array(d, float), np.zeros(len(d)), np.array(label, float), np.array(ep), np.array(t)


def _ep(role, source, score_from=0, is_success=True, hacky=False):
    return {"role": role, "source_name": source, "score_from": score_from, "is_success": is_success, "hacky": hacky}


def test_success_mae_pools_policy_success_and_intervention_tails_only():
    from square_assembly.scripts.eval_stg_scaling import ratio_metrics, unit_stats
    d, nll, label, ep, t = _frames({"a": ([3, 1, 0], [2, 1, 0]), "b": ([9, 9, 1, 2], [3, 2, 1, 0]),
                                    "c": ([5, 5], [1, 0])})
    episodes = {"a": _ep("policy_success", "demo_4"), "b": _ep("intv_delay", "demo_5", score_from=2),
                "c": _ep("demo", "demo_30")}
    m = ratio_metrics(unit_stats(d, nll, label, ep, t, episodes))
    assert m["success_mae"] == pytest.approx((1 + 0 + 0 + 0 + 2) / 5)   # 개입 전 프레임(9, 9)과 데모는 빠진다
    assert m["mae/intv_delay"] == pytest.approx(1.0) and m["mae/demo"] == pytest.approx(4.5)
    assert m["mae/intv_timeout"] is None


def test_false_progress_uses_chunk_means_and_skips_successful_branches():
    from square_assembly.scripts.eval_stg_scaling import ratio_metrics, unit_stats
    spike = [0, 0, 10] + [0] * 7 + [5] * 10            # 시작 한 프레임만 높다: W=1은 거짓 진전, W=8은 아니다
    series = {"up": (list(range(20)), [0] * 20), "down": (list(range(20))[::-1], [0] * 20),
              "spike": (spike, [0] * 20), "hk": ([9] * 20, [0] * 20), "ok": ([9] * 20, [0] * 20)}
    episodes = {"up": _ep("branch_t0", "demo_1", 2, False), "down": _ep("branch_t0", "demo_2", 2, False),
                "spike": _ep("branch_t_press", "demo_3", 2, False),
                "hk": _ep("branch_t_press", "demo_4", 2, True, hacky=True), "ok": _ep("branch_t0", "demo_5", 2, True)}
    stats = unit_stats(*_frames(series), episodes)
    m = ratio_metrics(stats)
    assert m["false_progress"] == pytest.approx(1 / 3) and m["false_progress_w1"] == pytest.approx(2 / 3)
    assert m["false_progress/branch_t0"] == pytest.approx(0.5) and m["false_progress/branch_t_press"] == 0
    assert sum(s.get("hacky", 0) for s in stats.values()) == 1
    assert sum(s.get("branch_success", 0) for s in stats.values()) == 1
    assert sum(s.get("nb", 0) for s in stats.values()) == 3


def test_bootstrap_ci_brackets_the_point_and_is_reproducible():
    from square_assembly.scripts.eval_stg_scaling import bootstrap_ci, ratio_metrics
    rng = np.random.RandomState(0)
    stats_list = []
    for _ in range(2):
        st = {f"rollout/demo_{k}": {"abs/intv_delay": rng.uniform(0, 100), "n/intv_delay": 10,
                                    "fp": float(rng.rand() > 0.5), "nb": 2} for k in range(8)}
        st["solo/demo_9"] = {"abs/policy_success": 30.0, "n/policy_success": 10}
        stats_list.append(st)
    ci = bootstrap_ci(stats_list, n_boot=500, seed=0)
    point = np.mean([ratio_metrics(s)["success_mae"] for s in stats_list])
    assert ci["success_mae"][0] <= point <= ci["success_mae"][1]
    assert ci == bootstrap_ci(stats_list, n_boot=500, seed=0)
    assert ci["mae/policy_success"] == [3.0, 3.0]          # 단위가 하나뿐인 층은 그 값 그대로
    assert ci["mae/demo"] is None


def test_auroc_ranks_failed_branch_frames_above_intervention_frames():
    from square_assembly.scripts.eval_stg_scaling import frame_metrics
    d, _, label, ep, t = _frames({"f": ([0, 0] + [300] * 10, [0] * 12), "i": ([500, 500] + list(range(9, -1, -1)),
                                                                          list(range(11, -1, -1)))})
    episodes = {"f": _ep("branch_t0", "demo_1", 2, False), "i": _ep("intv_delay", "demo_1", 2)}
    m = frame_metrics(d, label, ep, t, episodes)
    assert m["auroc_fail_vs_intv"] == 1.0                  # 채점 구간 앞(0, 500)은 빠진다
    assert m["reward"]["intv"]["k1"]["reward_sign_acc"] == 1.0


def test_shards_cover_every_job_once():
    from square_assembly.scripts.stg_scaling_sweep import shard
    jobs = [{"id": str(i)} for i in range(23)]
    parts = [shard(jobs, [s], 3) for s in range(3)]
    assert sorted(j["id"] for p in parts for j in p) == sorted(j["id"] for j in jobs)
    assert shard(jobs, [0, 2], 3) == [j for j in jobs if int(j["id"]) % 3 != 1]   # 순서(우선순위)는 유지


def _fake_python(tmp_path):
    """train_dstg_vip이면 out=에 파일을 만들고(FAIL_TRAIN이면 실패), eval이면 --out 폴더에 metrics.json을 쓴다."""
    script = tmp_path / "fakepy.sh"
    script.write_text("""#!/bin/bash
if [ "$2" = "square_assembly.scripts.train_dstg_vip" ]; then
  [ -n "$FAIL_TRAIN" ] && exit 1
  for a in "$@"; do case $a in out=*) touch "${a#out=}";; esac; done
else
  while [ $# -gt 0 ]; do [ "$1" = "--out" ] && echo '{}' > "$2/metrics.json"; shift; done
fi
""")
    script.chmod(0o755)
    return str(script)


def test_run_job_skips_finished_and_retries_failed(tmp_path, monkeypatch):
    from square_assembly.scripts.stg_scaling_sweep import run_jobs
    job = {"id": "random30/all/delay/r00_s0_abc", "cache": "c.h5", "eval_cache": "e.h5", "obs_parts": "all",
           "seed": 0, "num_bins": 522, "num_steps": 5, "train_episodes": ["demo_0"], "priority": 0}
    python, root = _fake_python(tmp_path), str(tmp_path / "root")
    done = tmp_path / "root/runs" / job["id"] / "metrics.json"
    monkeypatch.setenv("FAIL_TRAIN", "1")
    assert run_jobs([job], root, workers=1, python=python) == {"failed": 1}
    assert not done.exists()
    monkeypatch.delenv("FAIL_TRAIN")
    assert run_jobs([job], root, workers=1, python=python) == {"done": 1} and done.exists()
    import socket
    assert json.loads((done.parent / "timing.json").read_text())["host"] == socket.gethostname()
    assert run_jobs([job], root, workers=1, python=python) == {"skipped": 1}
    other = dict(job, id="random30/all/delay/r00_s1_def")
    import datetime
    past = datetime.datetime.now().astimezone() - datetime.timedelta(minutes=1)
    assert run_jobs([other], root, workers=1, python=python, deadline=past) == {"deadline": 1}


def test_aggregate_averages_seeds_and_reports_ci(tmp_path):
    from square_assembly.scripts.stg_scaling_sweep import aggregate
    root = tmp_path / "root"
    runs = []
    for seed, mae in ((0, 10.0), (1, 20.0)):
        job = f"random30/all/delay/r10_s{seed}_x"
        runs.append({"policy": "random30", "kind": "delay", "pairing": "same", "with_success": True,
                     "obs_parts": "all", "n": 10, "order": 0, "seed": seed, "job": job, "rollouts": ["demo_1"] * 10})
        units = {f"rollout/demo_{k}": {"abs/intv_delay": mae * 5, "n/intv_delay": 5, "fp": 1.0, "nb": 2}
                 for k in range(3)}
        (root / "runs" / job).mkdir(parents=True)
        (root / "runs" / job / "metrics.json").write_text(json.dumps(
            {"units": units, "n_train_samples": 1000 + seed, "frame": {"auroc_fail_vs_intv": 0.6}}))
    (root / "jobs.json").write_text(json.dumps({"num_steps": 5, "jobs": [], "runs": runs}))
    rows = aggregate(str(root), n_boot=50, plots=False)
    assert len(rows) == 1
    row = rows[0]
    assert row["n_predictors"] == 2 and row["success_mae_mean"] == pytest.approx(15.0)
    assert row["success_mae_std"] == pytest.approx(np.std([10, 20], ddof=1))
    assert row["success_mae_lo"] == pytest.approx(15.0) and row["false_progress_mean"] == pytest.approx(0.5)
    assert row["n_train_samples_mean"] == pytest.approx(1000.5) and row["auroc_fail_vs_intv_mean"] == pytest.approx(0.6)
    assert (root / "summary.csv").exists()


def test_sensitivity_plan_matches_epochs_and_adds_a_step_grid():
    """(E) n마다 학습 프레임에 비례한 step(최대 n = 4,000, 하한 300), (G) n ∈ {0, 10, 최대} × step {1000, 2000, 8000}."""
    from square_assembly.scripts.stg_scaling_sweep import plan_sensitivity
    jobs, runs = _fake_plan(plan=plan_sensitivity)
    main_jobs, main_runs = _fake_plan(num_steps=4000)
    job = {j["id"]: j for j in jobs}
    full = 200 + 33 * 50 + 9 * 20                                  # 최대 n: 데모 2 + 공통 개입 33 + 성공 9
    e = [r for r in runs if r["arm"] == "E"]
    assert {r["kind"] for r in e} == {"delay", "immediate", "timeout"} and {r["pairing"] for r in e} == {"same"}
    assert all(job[r["job"]]["num_steps"] == 4000 for r in e if r["n"] == 42)
    assert all(job[r["job"]]["num_steps"] == max(300, round(4000 * 200 / full)) for r in e if r["n"] == 0)
    g = [r for r in runs if r["arm"] == "G"]
    assert {r["n"] for r in g} == {0, 10, 42} and {r["steps"] for r in g} == {1000, 2000, 8000}
    assert all(job[r["job"]]["num_steps"] == r["steps"] for r in g)
    same = {(r["kind"], r["n"], r["order"], r["seed"]): r["rollouts"] for r in main_runs
            if r["pairing"] == "same" and r["with_success"] and r["obs_parts"] == "all"}
    assert all(r["rollouts"] == same[(r["kind"], r["n"], r["order"], r["seed"])] for r in runs)   # 본 스윕과 같은 에피소드
    assert len(job) == len({j["id"] for j in jobs}) and len(e) == 3 * 6 * 10 and len(g) == 3 * 3 * 3 * 10
