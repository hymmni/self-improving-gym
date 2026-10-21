"""Diffusion Policy 원저자(Chi et al.) 공개 체크포인트를 그대로 돌리는 호환 정책 + 변환기.

공개 체크포인트(diffusion-policy.cs.columbia.edu, 예: image/can_ph/diffusion_policy_cnn)는 UNet은 우리 것과 같고
이미지 인코더(robomimic VisualCore)와 액션 규약만 다르다. 부모(DiffusionPolicyImage)와 다른 점:
- 인코더: GroupNorm 그룹 수 C//16(우리는 16), SpatialSoftmax 뒤 Linear(64->64) + ReLU(robomimic ObservationEncoder
  기본 활성화), 입력은 [0,1] 대신 [-1,1](DP 이미지 정규화기).
- 청크 실행 시작점: DP는 청크의 To-1번(=1)부터 실행한다(pad_before=To-1로 학습). predict_action_chunk가 앞 To-1개를
  잘라 돌려주므로 rollout은 그대로 [0:action_horizon]을 실행하면 된다.
- 액션: 월드 절대 pos3 + rot6d + gripper1(10차원). env(delta 컨트롤러)에 넣기 전 delta로 바꾸는 건
  envs/robomimic/factory.py AbsRot6dActionEnv.
"""

import json
import os
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from square_assembly.policies.diffusion.diffusion_policy_image import DiffusionPolicyImage, SpatialSoftmax


class RobomimicRgbEncoder(nn.Module):
    """robomimic VisualCore(ResNet18Conv + SpatialSoftmax + Linear) + ObservationEncoder ReLU. 입력 (B,3,H,W) [0,1]."""

    def __init__(self, crop_hw, num_kp=32, feature_dim=64):
        super().__init__()
        self.random_crop = torchvision.transforms.RandomCrop(tuple(crop_hw))
        self.center_crop = torchvision.transforms.CenterCrop(tuple(crop_hw))
        resnet = torchvision.models.resnet18(weights=None)
        for m in list(resnet.modules()):
            for name, child in m.named_children():
                if isinstance(child, nn.BatchNorm2d):
                    setattr(m, name, nn.GroupNorm(child.num_features // 16, child.num_features))
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        with torch.no_grad():
            c, h, w = self.backbone(torch.zeros(1, 3, *crop_hw)).shape[1:]
        self.pool = SpatialSoftmax((c, h, w), num_kp=num_kp)
        self.linear = nn.Linear(num_kp * 2, feature_dim)
        self.out_dim = feature_dim

    def forward(self, x):
        x = self.random_crop(x) if self.training else self.center_crop(x)
        return F.relu(self.linear(self.pool(self.backbone(x * 2 - 1)).flatten(1)))


class DiffusionPolicyImageDP(DiffusionPolicyImage):
    def __init__(self, crop_hw, num_kp=32, **kwargs):
        super().__init__(crop_hw=crop_hw, num_kp=num_kp, **kwargs)
        self.encoders = nn.ModuleDict({k: RobomimicRgbEncoder(crop_hw, num_kp=num_kp) for k in self.rgb_keys})

    @torch.no_grad()
    def predict_action_chunk(self, obs):
        return super().predict_action_chunk(obs)[:, self.obs_horizon - 1:]


def _map_key(k):
    """DP EMA state_dict 키 -> 우리 키. 버릴 키는 None."""
    if k.startswith("model."):
        # DP: 잔차 블록 conv = blocks.0/1, Down/Upsample1d가 conv를 .conv로 감쌈. 우리: conv1/conv2, 맨 conv.
        k = k[len("model."):].replace(".blocks.0.", ".conv1.").replace(".blocks.1.", ".conv2.")
        return "unet." + re.sub(r"(_modules\.\d+\.2)\.conv\.", r"\1.", k)
    if not k.startswith("obs_encoder.obs_nets."):
        return None  # normalizer(따로 변환), mask_generator, obs_randomizers
    cam, rest = k[len("obs_encoder.obs_nets."):].split(".", 1)
    if rest.startswith("backbone.nets."):
        return f"encoders.{cam}.backbone.{rest[len('backbone.nets.'):]}"
    if rest in ("pool.nets.weight", "pool.nets.bias"):
        return f"encoders.{cam}.{rest}"
    if rest in ("nets.3.weight", "nets.3.bias"):
        return f"encoders.{cam}.linear.{rest.split('.')[-1]}"
    return None  # nets.0/1.*(backbone·pool의 중복 경로), pool.pos_x/pos_y/temperature(우리 pos_grid와 같은 값)


def convert_dp_ckpt(dp_ckpt_path, out_dir, task_name="can", env_hdf5=None):
    """DP 워크스페이스 체크포인트(EMA 가중치) -> out_dir/{policy_epoch<N>.pt, run_config.yaml, normalization_stats.json}.

    task는 configs/task/<task_name>.yaml + env_hdf5(robomimic v1.5 env_args, delta 컨트롤러 그대로)에서 만든다.
    정규화는 DP LinearNormalizer(y = x*scale + offset)를 같은 아핀 변환이 되는 min/max로 옮긴다.
    """
    from omegaconf import OmegaConf

    from square_assembly.utils.checkpoints import save_run_config
    from square_assembly.utils.task_utils import derive_task_meta_from_hdf5

    payload = torch.load(dp_ckpt_path, map_location="cpu", weights_only=False)
    cfg = payload["cfg"]
    sd = payload["state_dicts"]["ema_model"]
    shape_meta = cfg.policy.shape_meta
    rgb_keys = [k for k, v in shape_meta.obs.items() if v.get("type", "low_dim") == "rgb"]
    lowdim_keys = [k for k, v in shape_meta.obs.items() if v.get("type", "low_dim") == "low_dim"]
    # DP global_cond는 shape_meta 순서로 이어 붙인다 — 우리 _encode는 rgb 전부 다음 lowdim이라 순서가 같아야 한다.
    assert list(shape_meta.obs.keys()) == rgb_keys + lowdim_keys, list(shape_meta.obs.keys())

    here = os.path.dirname(__file__)
    task_cfg = OmegaConf.load(os.path.join(here, "../../configs/task", f"{task_name}.yaml"))
    policy_cfg = OmegaConf.load(os.path.join(here, "../../configs/policy/diffusion_unet.yaml"))
    if env_hdf5:
        task_cfg.hdf5_path = env_hdf5
    task_cfg.rgb_keys, task_cfg.lowdim_keys = rgb_keys, lowdim_keys
    derive_task_meta_from_hdf5(task_cfg)
    OmegaConf.set_struct(task_cfg, False)
    task_cfg.action_dim = int(shape_meta.action.shape[0])
    task_cfg.abs_rot6d_action = True
    p = cfg.policy
    policy_cfg.update(dict(obs_horizon=p.n_obs_steps, pred_horizon=p.horizon, action_horizon=p.n_action_steps,
                           down_dims=list(p.down_dims), kernel_size=p.kernel_size, n_groups=p.n_groups,
                           diffusion_step_embed_dim=p.diffusion_step_embed_dim, crop_hw=list(p.crop_shape),
                           num_train_timesteps=p.noise_scheduler.num_train_timesteps,
                           beta_schedule=p.noise_scheduler.beta_schedule, num_inference_steps=p.num_inference_steps))
    assert p.cond_predict_scale and p.obs_as_global_cond and p.noise_scheduler.prediction_type == "epsilon"

    from square_assembly.factory import registry
    import square_assembly.policies.diffusion.diffusion_policy_registrations  # noqa: F401
    policy = registry.create_policy("diffusion_dp", task_cfg, policy_cfg)
    ours = policy.state_dict()
    mapped = {}
    for k, v in sd.items():
        nk = _map_key(k)
        if nk is not None:
            assert nk in ours and ours[nk].shape == v.shape, (k, nk, tuple(v.shape))
            mapped[nk] = v
    missing = [k for k in ours if k not in mapped and not k.endswith("pos_grid")]
    assert not missing, missing
    for cam in rgb_keys:  # robomimic pos_x/pos_y와 우리 pos_grid가 같은 격자인지
        grid = ours[f"encoders.{cam}.pool.pos_grid"]
        pre = f"obs_encoder.obs_nets.{cam}.pool."
        assert torch.allclose(grid[:, 0], sd[pre + "pos_x"].flatten()) and torch.allclose(grid[:, 1], sd[pre + "pos_y"].flatten())
        assert float(sd[pre + "temperature"]) == 1.0
    policy.load_state_dict({**ours, **mapped})

    def minmax(key):
        scale = sd[f"normalizer.params_dict.{key}.scale"].double()
        offset = sd[f"normalizer.params_dict.{key}.offset"].double()
        return {"min": ((-1 - offset) / scale).tolist(), "max": ((1 - offset) / scale).tolist()}

    for cam in rgb_keys:  # 이미지 정규화가 [0,1]->[-1,1]이어야 인코더의 x*2-1과 맞다
        assert torch.allclose(sd[f"normalizer.params_dict.{cam}.scale"], torch.tensor(2.0))
        assert torch.allclose(sd[f"normalizer.params_dict.{cam}.offset"], torch.tensor(-1.0))
    stats = {"obs": {k: minmax(k) for k in lowdim_keys}, "action": minmax("action"),
             "dp_input_stats": {k: {s: sd[f"normalizer.params_dict.{k}.input_stats.{s}"].tolist()
                                    for s in ("min", "max", "mean", "std")} for k in lowdim_keys + ["action"]}}

    os.makedirs(out_dir, exist_ok=True)
    m = re.search(r"epoch=?(\d+)", os.path.basename(dp_ckpt_path))
    epoch = int(m.group(1)) if m else 0
    out = os.path.join(out_dir, f"policy_epoch{epoch}.pt")
    torch.save({"model": policy.state_dict(), "epoch": epoch, "dp_ckpt": os.path.abspath(dp_ckpt_path)}, out)
    with open(os.path.join(out_dir, "normalization_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    save_run_config(out_dir, task_cfg, policy_cfg, "diffusion_dp")
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("dp_ckpt")
    ap.add_argument("out_dir")
    ap.add_argument("--task", default="can")
    ap.add_argument("--env-hdf5", default=None)
    a = ap.parse_args()
    print(convert_dp_ckpt(a.dp_ckpt, a.out_dir, a.task, a.env_hdf5))
