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

