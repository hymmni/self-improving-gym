"""DstgPredictor(d(o,g) := E[steps-to-go | o]) 및 RobomimicSequenceDataset.get_time_to_success
검증. phases/5-mani-sim-ddpo/step1.md 참고.
"""

import os

import h5py
import pytest
import torch

from square_assembly.datasets.robomimic_dataset import RobomimicSequenceDataset
from square_assembly.policies.diffusion.diffusion_policy_image import DiffusionPolicyImage
from square_assembly.policies.diffusion.dstg_predictor import DstgPredictor

DEVICE = "cpu"
IMAGE_HW = (32, 32)
REAL_HDF5 = "/home/moai/hymm_ws/square_dataset/square_image_v15.hdf5"


def _make_policy(seed=0, image_hw=IMAGE_HW):
    torch.manual_seed(seed)
    policy = DiffusionPolicyImage(
        rgb_keys=["cam"],
        lowdim_keys=["state"],
        obs_dims={"state": 4},
        obs_horizon=2,
        action_dim=3,
        pred_horizon=4,
        num_kp=4,
        image_hw=image_hw,
        down_dims=(32, 64),
        num_train_timesteps=10,
        num_inference_steps=10,
    )
    return policy.to(DEVICE)


def _make_obs(batch_size=2, obs_horizon=2, device=DEVICE, image_hw=IMAGE_HW):
    return {
        "cam": torch.rand(batch_size, obs_horizon, 3, *image_hw, device=device),
        "state": torch.randn(batch_size, obs_horizon, 4, device=device),
    }


def test_frozen_encoder_no_grad():
    policy = _make_policy()
    predictor = DstgPredictor(policy, num_bins=50, head_hidden=(16, 16)).to(DEVICE)
    obs = _make_obs()

    logits = predictor(obs)
    assert logits.shape == (2, 50)
    logits.sum().backward()

    for p in predictor.frozen_policy.encoders.parameters():
        assert p.grad is None or torch.all(p.grad == 0)

    head_grads = [p.grad for p in predictor.head.parameters()]
    assert all(g is not None for g in head_grads)
    assert any(torch.any(g != 0) for g in head_grads)


def test_head_only_in_optimizer_group():
    """옵티마이저 파라미터 그룹이 head로만 좁혀졌는지(이중 안전판) — DstgPredictor 자체가
    아니라 train_dstg.py가 predictor.head.parameters()만 넘기는 관례를 재확인."""
    policy = _make_policy()
    predictor = DstgPredictor(policy, num_bins=50, head_hidden=(16, 16)).to(DEVICE)
    head_param_ids = {id(p) for p in predictor.head.parameters()}
    frozen_param_ids = {id(p) for p in predictor.frozen_policy.parameters()}
    assert head_param_ids.isdisjoint(frozen_param_ids)
    assert not any(p.requires_grad for p in predictor.frozen_policy.parameters())


@pytest.fixture
def small_hdf5(tmp_path):
    if not os.path.exists(REAL_HDF5):
        pytest.skip(f"실제 square 데이터셋이 이 머신에 없음: {REAL_HDF5}")
    out_path = tmp_path / "square_subset.hdf5"
    with h5py.File(REAL_HDF5, "r") as src, h5py.File(out_path, "w") as dst:
        data_grp = dst.create_group("data")
        for k, v in src["data"].attrs.items():
            data_grp.attrs[k] = v
        demo_ids = list(src["data"].keys())[:3]
        for demo_id in demo_ids:
            src.copy(f"data/{demo_id}", data_grp, name=demo_id)
    return str(out_path), demo_ids


def test_get_time_to_success(small_hdf5):
    hdf5_path, demo_ids = small_hdf5
    with h5py.File(hdf5_path, "r") as f:
        demo_lens = {d: f[f"data/{d}/actions"].shape[0] for d in demo_ids}

    dataset = RobomimicSequenceDataset(
        hdf5_path=hdf5_path,
        obs_keys=["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
        obs_horizon=2,
        pred_horizon=1,
    )
    labels = dataset.get_time_to_success()
    assert labels.shape == (len(dataset),)

    # L-1-t 규칙: 라벨 = 그 데모 길이 - 1 - (행동 청크가 시작하는 프레임 인덱스).
    for i in range(len(dataset)):
        demo_id, index_in_demo = dataset._demo_id_and_index_in_demo(i)
        assert labels[i] == demo_lens[demo_id] - 1 - index_in_demo

    # 데모별로 마지막 프레임(성공 시점)의 라벨은 0, 첫 프레임의 라벨은 demo_len - 1.
    for demo_id in demo_ids:
        demo_label_idxs = [i for i in range(len(dataset)) if dataset._demo_id_and_index_in_demo(i)[0] == demo_id]
        demo_labels = labels[demo_label_idxs]
        assert demo_labels.min() == 0
        assert demo_labels.max() == demo_lens[demo_id] - 1


def test_train_encoder_lets_grads_reach_the_encoder_but_not_the_unet():
    hw = (64, 64)  # 32x32는 ResNet 뒤 1x1이라 SpatialSoftmax가 상수 — 그래디언트가 0이 된다
    predictor = DstgPredictor(_make_policy(image_hw=hw), num_bins=50, head_hidden=(16, 16), train_encoder=True).to(DEVICE)
    predictor(_make_obs(image_hw=hw)).sum().backward()
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in predictor.frozen_policy.encoders.parameters())
    assert not any(p.requires_grad for p in predictor.frozen_policy.unet.parameters())
    with torch.no_grad():  # 보상 계산(no_grad) 경로에선 그래프를 안 만든다
        assert not predictor(_make_obs(image_hw=hw)).requires_grad


def test_success_tail_uses_last_human_onset_and_keeps_prefix_out_of_history():
    import numpy as np

    from square_assembly.datasets.stg_labels import success_tail_start
    from square_assembly.scripts.train_dstg import _LabeledWindow

    assert success_tail_start(np.array([0, 1, 1, 0, -10, 1, 1, 0])) == 5  # 개입 두 번 — 마지막 시작
    assert success_tail_start(np.zeros(4, int)) == 0                      # 무개입 성공은 전부

    class Base:  # obs 창 = 프레임 [t-1, t] (To=2)
        def __getitem__(self, t):
            return {"obs": {"x": torch.tensor([[max(t - 1, 0)], [t]], dtype=torch.float32)},
                    "demo_id": "demo_0", "index_in_demo": t}

    labels = np.arange(8)[::-1]
    tail = _LabeledWindow(Base(), labels, range(8), {"demo_0": 5})
    assert tail[5]["obs"]["x"].flatten().tolist() == [5, 5]  # 잘라낸 4번 프레임이 이력에 안 들어온다
    assert tail[6]["obs"]["x"].flatten().tolist() == [5, 6]
    assert tail[3]["obs"]["x"].flatten().tolist() == [2, 3]  # tail 밖 샘플은 그대로
    assert _LabeledWindow(Base(), labels, range(8))[5]["obs"]["x"].flatten().tolist() == [4, 5]  # held-out은 안 건드림
