"""공유 백본(VIP) DP + STG 헤드: 손실 마스크, 고정 층, 데이터 역할·라벨, DstgReward 호환."""

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

from square_assembly.datasets.dino_feature_dataset import episode_split_by_name
from square_assembly.datasets.joint_stg_dataset import JointBatchSampler, JointStgDataset
from square_assembly.datasets.labels import LABEL_DEMO, LABEL_INTV, LABEL_ROLLOUT
from square_assembly.policies.diffusion.diffusion_policy_stg import DiffusionPolicyImageStg, export_dstg_ckpt
from square_assembly.policies.diffusion.dstg_reward import DstgReward
from square_assembly.utils.checkpoints import save_epoch_checkpoint, save_run_config

RGB, LOW = ["agentview_image", "robot0_eye_in_hand_image"], ["robot0_eef_pos"]
POLICY_CFG = dict(name="t", obs_horizon=2, pred_horizon=4, action_horizon=2, down_dims=[16, 32], kernel_size=3,
                  n_groups=4, diffusion_step_embed_dim=8, num_train_timesteps=10, beta_schedule="squaredcos_cap_v2",
                  num_inference_steps=10, crop_hw=[14, 14], vip_pretrained=False, vip_stats_ckpt=None, vip_size=32,
                  vip_frozen_stages=["conv1", "bn1", "layer1", "layer2"], stg={"num_bins": 20})


def _policy(seed=0):
    torch.manual_seed(seed)
    return DiffusionPolicyImageStg(
        rgb_keys=RGB, lowdim_keys=LOW, obs_dims={"robot0_eef_pos": 3}, obs_horizon=2, action_dim=3, pred_horizon=4,
        image_hw=(16, 16), crop_hw=(14, 14), down_dims=(16, 32), kernel_size=3, n_groups=4,
        diffusion_step_embed_dim=8, num_train_timesteps=10, num_inference_steps=10,
        num_stg_bins=20, vip_pretrained=False, vip_size=32)


def _batch(b=4):
    obs = {k: torch.rand(b, 2, 3, 16, 16) for k in RGB}
    obs["robot0_eef_pos"] = torch.randn(b, 2, 3)
    return {"obs": obs, "action": torch.randn(b, 4, 3), "action_mask": torch.ones(b, 4, dtype=torch.bool),
            "stg_label": torch.randint(0, 20, (b,)), "bc_mask": torch.tensor([True, True, False, False]),
            "stg_mask": torch.tensor([False, False, True, True])}


def _grads(module):
    return [p.grad for p in module.parameters() if p.requires_grad]


def test_bc_rows_train_unet_and_stg_rows_train_head():
    policy = _policy()
    policy.stg_weight = 1.0
    batch = _batch()
    batch["bc_mask"][:] = False                       # STG 행만 → UNet엔 그래디언트 없음
    policy.compute_loss(batch).backward()
    assert all(g is None or not g.any() for g in _grads(policy.unet))
    assert any(g is not None and g.any() for g in _grads(policy.stg_head))
    assert any(g is not None and g.any() for g in _grads(policy.encoders))

    policy.zero_grad(set_to_none=True)
    policy.stg_weight = 0.0                           # λ=0 → 헤드엔 그래디언트 없음
    policy.compute_loss(_batch()).backward()
    assert all(g is None or not g.any() for g in _grads(policy.stg_head))
    assert any(g is not None and g.any() for g in _grads(policy.unet))


def test_vip_front_stages_frozen_and_bn_stays_eval():
    policy = _policy()
    convnet = policy.encoders[RGB[0]].convnet
    assert convnet is policy.encoders[RGB[1]].convnet  # 카메라 두 대가 백본 하나를 공유
    assert not any(p.requires_grad for p in convnet.layer2.parameters())
    assert all(p.requires_grad for p in convnet.layer3.parameters())
    policy.train()
    assert not any(m.training for m in convnet.modules() if isinstance(m, torch.nn.BatchNorm2d))
    assert policy.encoders[RGB[0]].training           # crop은 학습 모드(RandomCrop) 그대로


def test_grad_checkpoint_matches_plain_encoder():
    torch.manual_seed(0)
    policy = DiffusionPolicyImageStg(
        rgb_keys=RGB, lowdim_keys=LOW, obs_dims={"robot0_eef_pos": 3}, obs_horizon=2, action_dim=3, pred_horizon=4,
        image_hw=(16, 16), crop_hw=(14, 14), down_dims=(16, 32), kernel_size=3, n_groups=4,
        diffusion_step_embed_dim=8, num_train_timesteps=10, num_inference_steps=10,
        num_stg_bins=20, vip_pretrained=False, vip_size=32, vip_frozen_stages=(), vip_grad_checkpoint=True).train()
    enc = policy.encoders[RGB[0]]
    assert all(e.grad_checkpoint for e in policy.encoders.values())
    x = torch.rand(2, 3, 16, 16)
    outs, grads = [], []
    for ckpt in (False, True):
        enc.grad_checkpoint = ckpt
        policy.zero_grad(set_to_none=True)
        torch.manual_seed(1)                          # 같은 RandomCrop
        out = enc(x)
        out.square().sum().backward()
        outs.append(out.detach())
        grads.append([p.grad.clone() for p in enc.convnet.parameters()])
    assert torch.allclose(outs[0], outs[1], atol=1e-6)
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(*grads))


def test_exported_head_gives_same_d_through_dstg_reward(tmp_path):
    policy = _policy().eval()
    task = OmegaConf.create({"name": "t", "rgb_keys": RGB, "lowdim_keys": LOW, "obs_dims": {"robot0_eef_pos": 3},
                             "action_dim": 3, "image_size": 16})
    save_run_config(str(tmp_path), task, OmegaConf.create(POLICY_CFG), "diffusion_stg")
    ckpt = save_epoch_checkpoint(str(tmp_path), 1, policy)
    export_dstg_ckpt(ckpt, str(tmp_path / "predictor.pt"))

    reward = DstgReward(str(tmp_path / "predictor.pt"))
    obs = _batch()["obs"]
    with torch.no_grad():
        probs = torch.softmax(policy.stg_head(policy.get_global_cond(obs)), -1)
    expected = (probs * torch.arange(20.0)).sum(-1)
    assert torch.allclose(reward.d(obs), expected, atol=1e-5)


def _write_hdf5(path, episodes):
    """episodes: [(T, action_mode)] — 프레임 값 = 전역 프레임 번호라 윈도우를 눈으로 확인할 수 있다."""
    g = 0
    with h5py.File(path, "w") as f:
        for i, (T, modes) in enumerate(episodes):
            grp = f.create_group(f"data/demo_{i}")
            grp.attrs["is_success"] = True
            frames = np.arange(g, g + T)
            grp.create_dataset("actions", data=np.repeat(frames[:, None], 3, 1).astype(np.float64))
            grp.create_dataset("action_mode", data=np.asarray(modes))
            grp.create_dataset("obs/agentview_image", data=np.zeros((T, 4, 4, 3), np.uint8) + (frames % 200)[:, None, None, None].astype(np.uint8))
            grp.create_dataset("obs/robot0_eef_pos", data=np.repeat(frames[:, None], 3, 1).astype(np.float64))
            g += T


def test_dataset_roles_labels_and_sampler(tmp_path):
    D, I, R = LABEL_DEMO, LABEL_INTV, LABEL_ROLLOUT
    episodes = [(3, [D] * 3), (4, [D] * 4), (5, [R, R, I, I, R]), (4, [R, I, R, R]), (3, [R] * 3)]
    _write_hdf5(tmp_path / "m.hdf5", episodes)
    ds = JointStgDataset(tmp_path / "m.hdf5", ["agentview_image", "robot0_eef_pos"], 2, 4, None, ["agentview_image"],
                         val_fraction=0.2, split_seed=0, num_bins=20)
    names = [f"demo_{i}" for i in range(5)]
    val = episode_split_by_name(names, 0.2, 0)
    n = len(ds._index)
    assert n == 19 and len(ds) == 2 * n

    # BC 역할: 시연(demo_0, demo_1) 프레임 전부 — 검증 분할과 무관(기본 정책과 같은 BC 데이터)
    assert [ds._index[i] for i in ds.bc_idx] == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (1, 3)]
    # STG 역할: 검증 밖 데모의, 첫 개입 이후(개입 없으면 처음부터) 프레임
    starts = {0: 0, 1: 0, 2: 2, 3: 1, 4: 0}
    expect = [i for i, (d, t) in enumerate(ds._index) if names[d] not in val and t >= starts[d]]
    assert list(ds.stg_idx) == expect

    item = ds[n + 9]                                  # demo_2, t=2 (STG 역할)
    assert ds._index[9] == (2, 2)
    assert item["stg_label"] == 5 - 1 - 2 and not item["bc_mask"] and item["stg_mask"] == (names[2] not in val)
    assert item["obs"]["robot0_eef_pos"][:, 0].tolist() == [8.0, 9.0]   # 윈도우 = 전역 프레임 8, 9
    bc_item = ds[0]
    assert bc_item["bc_mask"] and not bc_item["stg_mask"]
    assert bc_item["obs"]["robot0_eef_pos"][:, 0].tolist() == [0.0, 0.0]  # 첫 프레임 반복 패딩

    sampler = JointBatchSampler(ds.bc_idx, ds.stg_idx, n, bc_batch=3, stg_batch=2, seed=0)
    batches = list(sampler)
    assert len(batches) == len(sampler) == 7 // 3
    for b in batches:
        assert all(i in set(ds.bc_idx) for i in b[:3]) and all(i - n in set(ds.stg_idx) for i in b[3:])


def test_camera_stats_scale_is_one_scalar_per_camera(tmp_path):
    from square_assembly.policies.diffusion.diffusion_policy_stg import _camera_stats
    mean = torch.arange(2 * 1024 + 3, dtype=torch.float32)
    std = torch.cat([torch.full((1024,), 1e-4), torch.full((1024,), 2.0), torch.ones(3)])
    std[0] = 3.0
    torch.save({"cache_meta": {"rgb_keys": RGB}, "frame_mean": mean.tolist(), "frame_std": std.tolist()}, tmp_path / "c.pt")
    stats = _camera_stats(tmp_path / "c.pt", RGB[::-1])
    m, s = stats[RGB[1]]
    assert torch.equal(m, mean[1024:2048]) and torch.allclose(s, torch.full((1024,), 2.0))
    assert torch.allclose(stats[RGB[0]][1], std[:1024].pow(2).mean().sqrt().expand(1024))   # 거의 상수인 차원에 끌려가지 않음
