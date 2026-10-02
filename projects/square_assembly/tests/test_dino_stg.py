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


from square_assembly.policies.diffusion.dino_stg_predictor import (
    DinoStgHead,
    load_checkpoint,
    save_checkpoint,
)


def test_head_shape_and_gradients():
    head = DinoStgHead(in_dim=12, num_bins=7, head_hidden=(8, 8))
    x = torch.randn(3, 12)

    logits = head(x)
    assert logits.shape == (3, 7)

    logits.sum().backward()
    grads = [p.grad for p in head.parameters()]
    assert all(g is not None for g in grads)
    assert any(torch.any(g != 0) for g in grads)


def test_checkpoint_roundtrip_preserves_predictions(tmp_path):
    """온라인 추론이 학습과 같은 전처리를 재현할 수 있어야 하므로, 전처리 메타가
    체크포인트에 남고 헤드가 그대로 복원돼야 한다."""
    head = DinoStgHead(in_dim=12, num_bins=7, head_hidden=(8, 8))
    x = torch.randn(3, 12)
    expected = head(x)

    path = tmp_path / "predictor.pt"
    save_checkpoint(
        str(path),
        head,
        {
            "in_dim": 12,
            "num_bins": 7,
            "head_hidden": [8, 8],
            "obs_horizon": 2,
            "frame_mean": np.zeros(6, dtype=np.float32),
            "frame_std": np.ones(6, dtype=np.float32),
            "cache_meta": {"model": "vit_small_patch14_reg4_dinov2.lvd142m", "crop": 76, "size": 224},
        },
    )

    loaded, ckpt = load_checkpoint(str(path))
    torch.testing.assert_close(loaded(x), expected)
    assert ckpt["cache_meta"]["crop"] == 76
    assert ckpt["num_bins"] == 7


@pytest.mark.parametrize("parts,dim", [("all", 6), ("feat", 4), ("lowdim", 2)])
def test_obs_parts_selects_which_columns_enter_the_window(tmp_path, parts, dim):
    """obs_parts는 프레임 벡터에서 이미지 특징/고유수용 감각 중 어느 쪽을 쓸지 고른다 —
    예측기가 진행도를 어디서 읽는지 가르는 실험 축(2026-09-23)."""
    cache = tmp_path / "c.h5"
    _write_fake_cache(cache, lengths=[3], feat_dim=4, low_dim=2)
    ds = DinoFeatureWindows(str(cache), obs_horizon=2, obs_parts=parts)

    assert ds.frame_dim == dim
    x, _ = ds[2]                      # t=2 -> 윈도우 [frame 1, frame 2]
    assert x.shape == (2 * dim,)
    # feat은 +t, lowdim은 -t로 채워져 있어 어느 쪽이 들어왔는지 부호로 구분된다.
    expected = {"all": [1, -1], "feat": [1, 1], "lowdim": [-1, -1]}[parts]
    assert [np.sign(x[0].item()), np.sign(x[dim - 1].item())] == expected


def test_obs_parts_rejects_unknown_value(tmp_path):
    cache = tmp_path / "c.h5"
    _write_fake_cache(cache, lengths=[2])
    with pytest.raises(ValueError, match="obs_parts"):
        DinoFeatureWindows(str(cache), obs_horizon=2, obs_parts="images")


def test_success_tail_selection_uses_last_onset_and_excludes_prefix_from_inputs_and_stats(tmp_path):
    cache, source = tmp_path / "c.h5", tmp_path / "m.h5"
    _write_fake_cache(cache, lengths=[8, 4, 4, 4], success=[True, True, True, False])
    with h5py.File(source, "w") as f:
        for name, modes in {"demo_0": [0, 1, 1, 0, 0, 1, 1, 0],
                            "demo_1": [-1] * 4, "demo_2": [0] * 4,
                            "demo_3": [0] * 4}.items():
            f.create_dataset(f"data/{name}/action_mode", data=modes)
    ds = DinoFeatureWindows(cache, 2, fail_bin=99, mode_hdf5=source)
    # Last onset is frame 5, not last human frame 6; final policy frame 7 stays.
    kept = ds.retain_success_tails(list(range(20)))
    assert kept == [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    assert len(ds) == 20  # Excluded prefixes remain available for evaluation.
    np.testing.assert_array_equal(ds.labels()[kept], [2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0])
    np.testing.assert_array_equal(ds[5][0].numpy(), [5] * 4 + [-5] * 2 + [5] * 4 + [-5] * 2)
    np.testing.assert_array_equal(ds[4][0].numpy(), [3] * 4 + [-3] * 2 + [4] * 4 + [-4] * 2)
    mean, std = ds.compute_frame_stats(kept + [5], sample_only=True)
    # Three tail frames (5,6,7), four demo (1000..1003), four rollout (2000..2003).
    expected = 12030 / 11
    np.testing.assert_allclose(mean, [expected] * 4 + [-expected] * 2)
    assert np.isfinite(std).all() and (std > 0).all()


def test_success_tail_selection_does_not_clamp_heldout_history(tmp_path):
    cache, source = tmp_path / "c.h5", tmp_path / "m.h5"
    _write_fake_cache(cache, lengths=[4, 4])
    with h5py.File(source, "w") as f:
        for name in ("demo_0", "demo_1"):
            f.create_dataset(f"data/{name}/action_mode", data=[0, 0, 1, 1])
    ds = DinoFeatureWindows(cache, 2, mode_hdf5=source)
    assert ds.retain_success_tails([0, 1, 2, 3]) == [2, 3]
    assert ds[2][0][0].item() == 2  # Train start pads within the tail.
    assert ds[6][0][0].item() == 1001  # Held-out start still sees its real history.


def test_success_tail_selection_requires_complete_mode_metadata(tmp_path):
    cache = tmp_path / "c.h5"
    _write_fake_cache(cache, lengths=[4])
    ds = DinoFeatureWindows(cache, 2)
    with pytest.raises(ValueError, match="action_mode"):
        ds.retain_success_tails([0, 1, 2, 3])


@pytest.mark.parametrize("modes", [[0, 0], [0, 0, 1, 1, 0]])
def test_strict_modes_rejects_source_length_mismatch_before_padding(tmp_path, modes):
    cache, source = tmp_path / "c.h5", tmp_path / "m.h5"
    _write_fake_cache(cache, lengths=[4])
    with h5py.File(source, "w") as f:
        f.create_dataset("data/demo_0/action_mode", data=modes)
    with pytest.raises(ValueError, match="action_mode"):
        DinoFeatureWindows(cache, 2, mode_hdf5=source, strict_modes=True)


def test_agentview_selects_named_camera_and_ignores_other_inputs(tmp_path):
    import json

    cache = tmp_path / "c.h5"
    _write_fake_cache(cache, lengths=[3])
    with h5py.File(cache, "r+") as f:
        # Camera order deliberately differs from the real VIP cache.
        f.attrs["meta"] = json.dumps({"rgb_keys": ["robot0_eye_in_hand_image", "agentview_image"]})
        f["data/demo_0/feat"][:] = [[90, 91, 10, 11], [92, 93, 20, 21], [94, 95, 30, 31]]
    before = DinoFeatureWindows(cache, 2, obs_parts="agentview")
    assert before.frame_dim == 2
    np.testing.assert_array_equal(before[2][0].numpy(), [20, 21, 30, 31])
    np.testing.assert_array_equal(before[0][0].numpy(), [10, 11, 10, 11])
    mean, std = before.compute_frame_stats([0, 1, 2])
    np.testing.assert_array_equal(mean, [20, 21])
    with h5py.File(cache, "r+") as f:
        f["data/demo_0/feat"][:, :2] = np.nan
        del f["data/demo_0/lowdim"]  # Proprioception must not even be read.
    after = DinoFeatureWindows(cache, 2, obs_parts="agentview")
    for i in range(3):
        torch.testing.assert_close(after[i][0], before[i][0])
        assert after[i][1] == before[i][1]
    mean_after, std_after = after.compute_frame_stats([0, 1, 2])
    np.testing.assert_array_equal(mean_after, mean)
    np.testing.assert_array_equal(std_after, std)


@pytest.mark.parametrize("keys,width", [([], 4), (["wrist"], 4),
                                      (["agentview_image", "agentview_image"], 4),
                                      (["agentview_image", "wrist"], 3)])
def test_agentview_rejects_missing_ambiguous_or_misaligned_camera_metadata(tmp_path, keys, width):
    import json

    cache = tmp_path / "c.h5"
    _write_fake_cache(cache, lengths=[3], feat_dim=width)
    with h5py.File(cache, "r+") as f:
        f.attrs["meta"] = json.dumps({"rgb_keys": keys})
    with pytest.raises(ValueError, match="camera|rgb_keys"):
        DinoFeatureWindows(cache, 2, obs_parts="agentview")


def test_rollout_limit_keeps_every_expert_demo_and_nests_across_sizes():
    """전문가 시연만(A) vs 롤아웃·개입을 n개 더한 것(B) 비교용 — 시연은 늘 다 남고, n이 커지면 남는 집합이 포함 관계다."""
    from square_assembly.scripts.train_dstg_vip import rollout_demos_to_drop
    modes = {f"demo_{i}": np.full(5, -1) for i in range(3)}              # 전문가 시연
    modes.update({f"demo_{i}": np.array([0, 0, 1, 1, 1]) for i in range(3, 7)})   # 개입 성공
    modes.update({f"demo_{i}": np.zeros(5, dtype=int) for i in range(7, 9)})      # 무개입 성공
    names, rollouts = list(modes), {f"demo_{i}" for i in range(3, 9)}

    assert rollout_demos_to_drop(names, modes, 0, seed=0) == rollouts                # A: 시연만
    kept = [rollouts - rollout_demos_to_drop(names, modes, n, seed=0) for n in (2, 4, 6)]
    assert [len(k) for k in kept] == [2, 4, 6] and kept[0] < kept[1] < kept[2]
    assert rollout_demos_to_drop(names, modes, 99, seed=0) == set()


def test_rollout_limit_can_keep_only_no_intervention_rollouts():
    """시연 + 사람 개입 없이 성공한 롤아웃만 — 개입 데모는 limit와 무관하게 전부 빠진다."""
    from square_assembly.scripts.train_dstg_vip import rollout_demos_to_drop
    modes = {f"demo_{i}": np.full(5, -1) for i in range(3)}
    modes.update({f"demo_{i}": np.array([0, -10, 1, 1, 1]) for i in range(3, 7)})   # 개입 성공
    modes.update({f"demo_{i}": np.zeros(5, dtype=int) for i in range(7, 10)})       # 무개입 성공
    names, intv, clean = list(modes), {f"demo_{i}" for i in range(3, 7)}, {f"demo_{i}" for i in range(7, 10)}

    drop = rollout_demos_to_drop(names, modes, 2, seed=0, kind="no_intervention")
    assert intv <= drop and len(clean - drop) == 2
    assert rollout_demos_to_drop(names, modes, 99, seed=0, kind="no_intervention") == intv


def test_clamp_labels_folds_long_remaining_steps_into_the_last_bin(tmp_path):
    """num_bins보다 먼 라벨은 마지막 bin("그 이상")으로 모이고, 바뀐 샘플을 알려 준다 — 나머지는 그대로."""
    cache = tmp_path / "feats.h5"
    _write_fake_cache(cache, lengths=[6, 3])
    dataset = DinoFeatureWindows(cache, obs_horizon=2)
    before = dataset.labels().copy()

    clamped = dataset.clamp_labels(num_bins=4)

    assert clamped.tolist() == (before >= 4).tolist() and clamped.sum() == 2
    assert dataset.labels().tolist() == np.minimum(before, 3).tolist()
    assert dataset[int(np.flatnonzero(clamped)[0])][1] == 3
