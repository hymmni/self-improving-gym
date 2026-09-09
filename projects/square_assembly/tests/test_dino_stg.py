"""DINO 특징 캐시·데이터셋·STG 헤드 검증.

이 파일은 robomimic 없이 돌아야 한다(로컬 PC에 robomimic이 없다). 사전학습
가중치 다운로드도 하지 않는다(pretrained=False) — 네트워크 없이 어디서나 돈다.
실제 가중치 다운로드는 캐시 스크립트 스모크 실행이 확인한다.
"""

import numpy as np
import torch

from square_assembly.scripts.cache_dino_feats import DEFAULT_MODEL, build_encoder, encode_frames


def test_encode_frames_shape_and_dtype():
    model = build_encoder(DEFAULT_MODEL, pretrained=False, device="cpu")
    imgs = np.random.randint(0, 256, size=(3, 84, 84, 3), dtype=np.uint8)

    feat = encode_frames(model, imgs, crop=76, device="cpu", batch_size=2)

    assert feat.shape == (3, 2 * model.embed_dim)  # [CLS ; mean(patch)]
    assert feat.dtype == np.float16
    assert np.isfinite(feat).all()


def test_encode_frames_is_deterministic():
    """얼린 인코더이므로 같은 프레임은 항상 같은 특징이어야 한다 — 캐싱 방식 자체의 전제."""
    model = build_encoder(DEFAULT_MODEL, pretrained=False, device="cpu")
    imgs = np.random.randint(0, 256, size=(2, 84, 84, 3), dtype=np.uint8)

    a = encode_frames(model, imgs, crop=76, device="cpu", batch_size=2)
    b = encode_frames(model, imgs, crop=76, device="cpu", batch_size=1)

    np.testing.assert_array_equal(a, b)


import types

import h5py
import pytest

from square_assembly.datasets.dino_feature_dataset import DinoFeatureWindows, episode_split_by_name


def _write_fake_cache(path, lengths, feat_dim=4, low_dim=2, success=None):
    """가짜 특징 캐시 — 값이 (demo, t)로부터 결정되게 넣어 윈도우 정렬을 눈으로 검증 가능하게."""
    with h5py.File(path, "w") as f:
        for i, L in enumerate(lengths):
            name = f"demo_{i}"
            feat = np.stack([np.full(feat_dim, i * 1000 + t, dtype=np.float16) for t in range(L)])
            low = np.stack([np.full(low_dim, -(i * 1000 + t), dtype=np.float32) for t in range(L)])
            g = f.create_group(f"data/{name}")
            g.create_dataset("feat", data=feat)
            g.create_dataset("lowdim", data=low)
            g.attrs["length"] = L
            g.attrs["is_success"] = True if success is None else bool(success[i])
        f.attrs["meta"] = "{}"


def test_sample_count_and_labels_match_baseline_rule(tmp_path):
    """샘플 수와 라벨은 robomimic(frame_stack=To, pad_frame_stack=True) +
    get_time_to_success와 같아야 한다: 데모당 L개 샘플, t=0..L-1, 라벨 = L-1-t."""
    cache = tmp_path / "c.h5"
    _write_fake_cache(cache, lengths=[5, 3])
    ds = DinoFeatureWindows(str(cache), obs_horizon=2)

    assert len(ds) == 8  # 5 + 3
    assert ds.samples[:3] == [("demo_0", 0), ("demo_0", 1), ("demo_0", 2)]
    np.testing.assert_array_equal(ds.labels(), [4, 3, 2, 1, 0, 2, 1, 0])


def test_window_pads_by_repeating_the_first_frame(tmp_path):
    """t=0에는 앞쪽 이력이 없으므로 첫 프레임이 복제돼야 한다(pad_frame_stack=True와 동일)."""
    cache = tmp_path / "c.h5"
    _write_fake_cache(cache, lengths=[5], feat_dim=4, low_dim=2)
    ds = DinoFeatureWindows(str(cache), obs_horizon=2)

    x0, y0 = ds[0]  # t=0 -> 윈도우 [frame 0, frame 0]
    assert x0.shape == (2 * 6,)
    np.testing.assert_array_equal(x0[:4].numpy(), [0, 0, 0, 0])
    np.testing.assert_array_equal(x0[6:10].numpy(), [0, 0, 0, 0])
    assert y0 == 4

    x2, y2 = ds[2]  # t=2 -> 윈도우 [frame 1, frame 2]
    np.testing.assert_array_equal(x2[:4].numpy(), [1, 1, 1, 1])
    np.testing.assert_array_equal(x2[6:10].numpy(), [2, 2, 2, 2])
    np.testing.assert_array_equal(x2[4:6].numpy(), [-1, -1])  # frame 1 lowdim
    assert y2 == 2


def test_split_matches_train_dstg_episode_split(tmp_path):
    """val 분할이 기준선과 글자 그대로 같아야 val_mae 비교가 성립한다.
    train_dstg._episode_split을 가짜 dataset으로 직접 호출해 대조한다(robomimic 불필요)."""
    from square_assembly.scripts.train_dstg import _episode_split

    lengths = [7, 5, 9, 4, 6, 8, 3, 5, 6, 7, 4, 5]
    cache = tmp_path / "c.h5"
    _write_fake_cache(cache, lengths=lengths)
    ds = DinoFeatureWindows(str(cache), obs_horizon=2)
    train_idx, val_idx, _ = ds.split_indices(val_fraction=0.2, seed=0)

    demo_ids = [name for name, _ in ds.samples]

    class _FakeDataset:
        def __init__(self, ids):
            self._seq_dataset = types.SimpleNamespace(
                _index_to_demo_id={i: d for i, d in enumerate(ids)}
            )

        def __len__(self):
            return len(self._seq_dataset._index_to_demo_id)

    ref_train, ref_val, _, _ = _episode_split(_FakeDataset(demo_ids), 0.2, 0)

    assert train_idx == ref_train
    assert val_idx == ref_val


def test_failure_demo_needs_a_fail_bin(tmp_path):
    """실패 데모가 섞였는데 fail_bin이 없으면 조용히 틀린 라벨을 붙이는 대신 즉시 죽어야 한다."""
    cache = tmp_path / "c.h5"
    _write_fake_cache(cache, lengths=[4, 4], success=[True, False])

    with pytest.raises(ValueError, match="fail_bin"):
        DinoFeatureWindows(str(cache), obs_horizon=2)

    ds = DinoFeatureWindows(str(cache), obs_horizon=2, fail_bin=99)
    np.testing.assert_array_equal(ds.labels(), [3, 2, 1, 0, 99, 99, 99, 99])


def test_frame_stats_use_training_demos_only(tmp_path):
    """표준화 통계가 val 데모를 보면 누출이다 — train 인덱스만 봐야 한다."""
    cache = tmp_path / "c.h5"
    _write_fake_cache(cache, lengths=[4] * 10)
    ds = DinoFeatureWindows(str(cache), obs_horizon=2)
    train_idx, val_idx, val_demos = ds.split_indices(val_fraction=0.2, seed=0)

    mean, std = ds.compute_frame_stats(train_idx)
    train_frames = np.concatenate(
        [ds.frames[n] for n in sorted(ds.frames) if n not in val_demos], axis=0
    )
    np.testing.assert_allclose(mean, train_frames.mean(0), rtol=1e-5)
    assert (std > 0).all()

    ds.apply_frame_stats(mean, std)
    x, _ = ds[train_idx[0]]
    assert torch.isfinite(x).all()
