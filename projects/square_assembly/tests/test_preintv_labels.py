"""PREINTV 구간 라벨 재작성 — flat/rise/drop이 실제로 라벨 기울기를 바꾸는지 못 박는다.

카운트다운 라벨은 정책이 망가지는 PREINTV 구간에서도 매 스텝 1씩 줄어 "좋아지고 있다"고
말한다. 여기가 조용히 틀리면 학습은 멀쩡히 돌고 보상만 반대 부호가 되므로 고정해 둔다.
"""

import h5py
import numpy as np
import pytest

from square_assembly.datasets.dino_feature_dataset import DinoFeatureWindows
from square_assembly.datasets.labels import LABEL_PREINTV, LABEL_ROLLOUT

LENGTH = 10
PREINTV_RANGE = range(4, 7)  # t=4,5,6


@pytest.fixture
def paths(tmp_path):
    cache, source = tmp_path / "c.h5", tmp_path / "s.hdf5"
    with h5py.File(cache, "w") as f:
        g = f.create_group("data/demo_0")
        g.attrs["length"] = LENGTH
        g.attrs["is_success"] = True
        g.create_dataset("feat", data=np.zeros((LENGTH, 3), np.float32))
        g.create_dataset("lowdim", data=np.zeros((LENGTH, 2), np.float32))
    mode = np.full(LENGTH, LABEL_ROLLOUT, np.int64)
    mode[list(PREINTV_RANGE)] = LABEL_PREINTV
    with h5py.File(source, "w") as f:
        f.create_dataset("data/demo_0/action_mode", data=mode)
    return str(cache), str(source)


def labels_of(cache, source, preintv):
    ds = DinoFeatureWindows(cache, 2, mode_hdf5=source, preintv=preintv)
    return ds.labels()


def test_countdown_label_claims_progress_while_the_policy_is_failing(paths):
    assert list(labels_of(*paths, "none")) == [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]


def test_flat_holds_the_label_still_across_the_preintv_run(paths):
    lab = labels_of(*paths, "flat")
    assert list(lab[4:7]) == [5, 5, 5]          # 구간 시작(t=4)의 참값에 고정 = 진전 0
    assert list(lab[:4]) == [9, 8, 7, 6]        # 구간 밖은 그대로
    assert list(lab[7:]) == [2, 1, 0]


def test_rise_makes_the_label_grow_so_the_reward_goes_negative(paths):
    lab = labels_of(*paths, "rise")
    assert list(lab[4:7]) == [5, 6, 7]
    assert np.all(np.diff(lab[4:7]) > 0)        # 남은 비용이 늘어난다 = 보상 음수


def test_drop_reports_exactly_the_preintv_frames(paths):
    ds = DinoFeatureWindows(paths[0], 2, mode_hdf5=paths[1], preintv="drop")
    assert ds.preintv_indices() == list(PREINTV_RANGE)


def test_preintv_handling_needs_the_action_mode_source(paths):
    with pytest.raises(ValueError, match="mode_hdf5"):
        DinoFeatureWindows(paths[0], 2, preintv="flat")
