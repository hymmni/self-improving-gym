"""너트 초기 yaw 제외 구간: env 생성 → 수집 hdf5 → 병합 env_args → 학습 task_cfg까지 이어지는지."""

import json

import h5py
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

from square_assembly.scripts.merge_demo_hdf5 import merge


def test_square_env_never_samples_excluded_nut_yaw():
    from square_assembly.envs.robomimic.factory import make_lowdim_env

    env = make_lowdim_env("NutAssemblySquare", "Panda", ["robot0_eef_pos"], env_kwargs={"nut_yaw_exclude": [-10, 125]})
    raw = env.env
    bid = raw.sim.model.body_name2id(raw.nuts[0].root_body)
    yaws = []
    for _ in range(40):
        env.reset()
        q = raw.sim.data.body_xquat[bid]
        yaws.append(Rot.from_quat([*q[1:], q[0]]).as_euler("xyz", degrees=True)[2])
    yaws = np.mod(np.asarray(yaws), 360)
    assert not np.any((yaws > 350) | (yaws < 125))   # [-10, 125] = [350, 360) ∪ [0, 125)
    assert yaws.max() - yaws.min() > 120             # 나머지 구간에서는 고르게 뽑는다
    assert env.serialize()["env_kwargs"].get("nut_yaw_exclude") is None   # robosuite kwargs로 새지 않는다


def _write(path, exclude=None, env_args=None):
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        g = data.create_group("demo_0")
        g.attrs["num_samples"], g.attrs["is_success"] = 2, True
        g.create_dataset("actions", data=np.zeros((2, 7)))
        g.create_group("obs").create_dataset("robot0_eef_pos", data=np.zeros((2, 3)))
        data.attrs["total"] = 2
        if exclude is not None:
            data.attrs["nut_yaw_exclude"] = exclude
        if env_args is not None:
            data.attrs["env_args"] = json.dumps(env_args)


def test_merge_carries_nut_yaw_exclude_into_env_args(tmp_path):
    ref = tmp_path / "ref.hdf5"
    _write(ref, env_args={"env_name": "NutAssemblySquare", "env_kwargs": {"lite_physics": False}})
    a, b, out = tmp_path / "a.hdf5", tmp_path / "b.hdf5", tmp_path / "out.hdf5"
    _write(a, exclude=[-10, 125]); _write(b, exclude=[-10, 125])
    merge([str(a), str(b)], str(out), env_args_from=str(ref))
    with h5py.File(out) as f:
        kw = json.loads(f["data"].attrs["env_args"])["env_kwargs"]
    assert kw == {"lite_physics": False, "nut_yaw_exclude": [-10, 125]}


def test_merge_rejects_mixed_nut_yaw_ranges(tmp_path):
    a, b = tmp_path / "a.hdf5", tmp_path / "b.hdf5"
    _write(a, exclude=[-10, 125]); _write(b)
    with pytest.raises(ValueError):
        merge([str(a), str(b)], str(tmp_path / "out.hdf5"))


def test_derive_keeps_nut_yaw_exclude():
    from square_assembly.utils.task_utils import _RELEVANT_ENV_KWARGS

    assert "nut_yaw_exclude" in _RELEVANT_ENV_KWARGS


def test_merge_filters_unrestricted_source_by_initial_nut_yaw(tmp_path):
    import robosuite as suite

    env = suite.make("NutAssemblySquare", robots="Panda", has_renderer=False, has_offscreen_renderer=False,
                     use_camera_obs=False)
    sampler = env.placement_initializer.samplers["SquareNutSampler"]
    src, out = tmp_path / "old.hdf5", tmp_path / "out.hdf5"
    with h5py.File(src, "w") as f:   # 제한 없이 모은 옛 파일: 너트 yaw 0°, 100°(제외 구간), 200°
        data = f.create_group("data")
        for i, deg in enumerate([0.0, 100.0, 200.0]):
            sampler.rotation = np.deg2rad(deg)
            env.reset()
            g = data.create_group(f"demo_{i}")
            g.attrs["num_samples"], g.attrs["is_success"] = 1, True
            g.create_dataset("actions", data=np.full((1, 7), deg))
            g.create_dataset("states", data=env.sim.get_state().flatten()[None])
            g.create_group("obs").create_dataset("robot0_eef_pos", data=np.zeros((1, 3)))
    n, _ = merge([str(src)], str(out), nut_yaw_exclude=(-10, 125))
    with h5py.File(out) as f:
        kept = [f["data"][k]["actions"][0, 0] for k in ("demo_0", "demo_1")[:n]]
        assert list(f["data"].attrs["nut_yaw_exclude"]) == [-10.0, 125.0]
    assert n == 1 and kept == [200.0]   # 0°도 [-10, 125] 안이라 빠진다
