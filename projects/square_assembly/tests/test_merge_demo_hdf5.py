"""merge_demo_hdf5: 라벨 승계·성공 필터·에피소드 번호 재부여 검증."""

import h5py
import numpy as np
import pytest

from square_assembly.datasets.labels import LABEL_DEMO, LABEL_INTV
from square_assembly.scripts.merge_demo_hdf5 import merge


def _write(path, episodes):
    """episodes: [(T, is_success, action_mode 또는 None)] -> 수집 산출물과 같은 구조의 hdf5."""
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        for i, (T, success, modes) in enumerate(episodes):
            grp = data.create_group(f"demo_{i}")
            grp.attrs["num_samples"] = T
            grp.attrs["is_success"] = success
            grp.create_dataset("actions", data=np.zeros((T, 7)))
            if modes is not None:
                grp.create_dataset("action_mode", data=np.asarray(modes))
            obs = grp.create_group("obs")
            obs.create_dataset("agentview_image", data=np.zeros((T, 4, 4, 3), dtype=np.uint8))
            obs.create_dataset("robot0_eef_pos", data=np.zeros((T, 3)))
        data.attrs["total"] = sum(e[0] for e in episodes)


def test_merge_labels_and_renumbering(tmp_path):
    demos = tmp_path / "demos.hdf5"
    intv = tmp_path / "intv.hdf5"
    out = tmp_path / "merged.hdf5"
    _write(demos, [(3, True, None), (2, True, None)])          # 순수 시연(라벨 없음)
    _write(intv, [(4, True, [0, 0, LABEL_INTV, LABEL_INTV])])  # 개입 수집분

    n_demo, n_frame = merge([str(demos), str(intv)], str(out))
    assert (n_demo, n_frame) == (3, 9)

    with h5py.File(out) as f:
        assert sorted(f["data"].keys()) == ["demo_0", "demo_1", "demo_2"]
        assert f["data"].attrs["total"] == 9
        # 라벨 없던 소스는 전부 DEMO(-1)로 채워진다
        np.testing.assert_array_equal(np.asarray(f["data/demo_0/action_mode"]), [LABEL_DEMO] * 3)
        # 라벨 있던 소스는 그대로 승계된다
        np.testing.assert_array_equal(
            np.asarray(f["data/demo_2/action_mode"]), [0, 0, LABEL_INTV, LABEL_INTV]
        )
        assert f["data/demo_2"].attrs["num_samples"] == 4
        assert list(f["data/demo_0/obs"].keys()) == ["agentview_image", "robot0_eef_pos"]


def test_failures_are_dropped_unless_asked(tmp_path):
    src = tmp_path / "mixed.hdf5"
    _write(src, [(3, True, None), (5, False, None), (2, True, None)])

    n_demo, n_frame = merge([str(src)], str(tmp_path / "a.hdf5"))
    assert (n_demo, n_frame) == (2, 5)

    n_demo, n_frame = merge([str(src)], str(tmp_path / "b.hdf5"), only_success=False)
    assert (n_demo, n_frame) == (3, 10)


def test_demo_order_is_numeric_not_lexicographic(tmp_path):
    """demo_10이 demo_2보다 뒤에 와야 한다(문자열 정렬이면 뒤집힌다)."""
    src = tmp_path / "many.hdf5"
    _write(src, [(i + 1, True, None) for i in range(11)])

    merge([str(src)], str(tmp_path / "out.hdf5"))
    with h5py.File(tmp_path / "out.hdf5") as f:
        lengths = [int(f[f"data/demo_{i}"].attrs["num_samples"]) for i in range(11)]
    assert lengths == list(range(1, 12))


def test_env_args_is_copied_from_reference(tmp_path):
    src = tmp_path / "s.hdf5"
    ref = tmp_path / "ref.hdf5"
    _write(src, [(2, True, None)])
    _write(ref, [(1, True, None)])
    with h5py.File(ref, "a") as f:
        f["data"].attrs["env_args"] = '{"env_name": "NutAssemblySquare"}'

    merge([str(src)], str(tmp_path / "out.hdf5"), env_args_from=str(ref))
    with h5py.File(tmp_path / "out.hdf5") as f:
        assert "NutAssemblySquare" in f["data"].attrs["env_args"]


def test_missing_env_args_reference_is_an_error(tmp_path):
    src = tmp_path / "s.hdf5"
    ref = tmp_path / "ref.hdf5"
    _write(src, [(2, True, None)])
    _write(ref, [(1, True, None)])  # env_args 없는 파일을 참조로 주면 조용히 넘어가면 안 된다

    with pytest.raises(KeyError):
        merge([str(src)], str(tmp_path / "out.hdf5"), env_args_from=str(ref))
