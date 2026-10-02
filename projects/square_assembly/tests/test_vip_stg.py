"""VIP 특징 캐시 검증. robomimic·네트워크 없이 돌아야 한다(pretrained=False, DINO 쪽과 동일 원칙).

다운스트림 데이터셋/헤드(DinoFeatureWindows, DinoStgHead)는 캐시 hdf5의 feat/lowdim
포맷에만 의존하고 인코더를 모르므로 test_dino_stg.py가 이미 검증한다 — 여기서는
cache_vip_feats.py 고유 로직(전처리, 출력 shape/dtype)만 본다.
"""

import numpy as np

from square_assembly.scripts.cache_vip_feats import EMBED_DIM, build_encoder, encode_frames


def test_encode_frames_shape_and_dtype():
    model = build_encoder(pretrained=False, device="cpu")
    imgs = np.random.randint(0, 256, size=(3, 84, 84, 3), dtype=np.uint8)

    feat = encode_frames(model, imgs, crop=76, device="cpu", batch_size=2)

    assert feat.shape == (3, EMBED_DIM)
    assert feat.dtype == np.float16
    assert np.isfinite(feat).all()


def test_encode_frames_is_deterministic():
    """얼린 인코더이므로 같은 프레임은 항상 같은 특징이어야 한다 — 캐싱 방식 자체의 전제.

    ViT(DINO)와 달리 ResNet(VIP)은 conv의 im2col 크기가 배치 크기에 따라 달라져
    CPU에서 마지막 몇 비트가 흔들릴 수 있다(부동소수점 결합법칙 위반, 값 자체는 동일
    분포) — 그래서 완전 동일이 아니라 허용오차 비교로 배치 불변성만 확인한다.
    """
    model = build_encoder(pretrained=False, device="cpu")
    imgs = np.random.randint(0, 256, size=(2, 84, 84, 3), dtype=np.uint8)

    a = encode_frames(model, imgs, crop=76, device="cpu", batch_size=2)
    b = encode_frames(model, imgs, crop=76, device="cpu", batch_size=1)

    np.testing.assert_allclose(a.astype(np.float32), b.astype(np.float32), rtol=1e-2, atol=1e-2)


def test_reward_wrapper_matches_the_cached_feature_path(tmp_path):
    """롤아웃용 obs 배치로 낸 d가, 같은 프레임을 캐시해서 낸 d(학습·평가 경로)와 같아야 한다."""
    import h5py
    import json
    import torch

    from square_assembly.datasets.dino_feature_dataset import DinoFeatureWindows
    from square_assembly.datasets.normalization import MinMaxNormalizer
    from square_assembly.policies.diffusion.dino_stg_predictor import DinoStgHead, save_checkpoint
    from square_assembly.policies.diffusion.vip_stg_reward import VipStgReward

    torch.manual_seed(0)
    rng = np.random.RandomState(0)
    model = build_encoder(pretrained=False, device="cpu")
    length, cams, low_keys = 4, ["cam_a", "cam_b"], ["pos", "grip"]
    imgs = {k: rng.randint(0, 256, size=(length, 84, 84, 3), dtype=np.uint8) for k in cams}
    low = {"pos": rng.uniform(-1, 1, (length, 3)).astype(np.float32),
           "grip": rng.uniform(0, .04, (length, 2)).astype(np.float32)}
    cache = tmp_path / "cache.h5"
    with h5py.File(cache, "w") as f:   # cache_vip_feats.main과 같은 포맷
        g = f.create_group("data/demo_0")
        g.create_dataset("feat", data=np.concatenate([encode_frames(model, imgs[k], 76, device="cpu") for k in cams], -1))
        g.create_dataset("lowdim", data=np.concatenate([low[k] for k in low_keys], -1))
        g.attrs["is_success"], g.attrs["length"] = True, length
        f.attrs["meta"] = json.dumps({"crop": 76, "size": 224, "rgb_keys": cams, "lowdim_keys": low_keys})

    dataset = DinoFeatureWindows(cache, 2)
    mean, std = dataset.compute_frame_stats(range(length))
    dataset.apply_frame_stats(mean, std)
    head = DinoStgHead(2 * dataset.frame_dim, 10, head_hidden=(8,))
    ckpt = tmp_path / "predictor.pt"
    save_checkpoint(ckpt, head, {"in_dim": 2 * dataset.frame_dim, "num_bins": 10, "head_hidden": [8], "obs_horizon": 2,
                                 "frame_mean": mean, "frame_std": std,
                                 "cache_meta": {"crop": 76, "size": 224, "rgb_keys": cams, "lowdim_keys": low_keys}})
    bins = torch.arange(10, dtype=torch.float32)
    with torch.no_grad():
        expected = torch.stack([(head(dataset[t][0][None]).softmax(-1) * bins).sum() for t in range(length)])

    normalizer = MinMaxNormalizer({"obs": {"pos": {"min": [-1.] * 3, "max": [1.] * 3},
                                           "grip": {"min": [0.] * 2, "max": [.04] * 2}},
                                   "action": {"min": [0.], "max": [1.]}})
    reward = VipStgReward(ckpt, normalizer, encoder=model)
    window = np.array([[max(t - 1, 0), t] for t in range(length)])   # 앞쪽 패딩 = 첫 프레임 복제
    obs = {k: torch.from_numpy(imgs[k][window]).permute(0, 1, 4, 2, 3).float() / 255 for k in cams}
    obs.update(normalizer.normalize_obs({k: torch.from_numpy(low[k][window]) for k in low_keys}))

    assert reward.obs_keys == cams + low_keys and reward.obs_horizon == 2
    np.testing.assert_allclose(reward.d(obs).numpy(), expected.numpy(), atol=2e-2)
