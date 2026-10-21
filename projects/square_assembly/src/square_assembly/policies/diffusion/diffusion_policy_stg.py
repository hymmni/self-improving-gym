"""공유 백본 Diffusion Policy + STG 헤드 — 인코더 하나를 행동(BC) 손실과 steps-to-go(STG) 손실이 함께 학습한다.

배경(experiments/2026-10-06_si-dppo-r0v10.md §4~§9): 얼린 VIP 특징 위 STG 예측기(①)는 빈손으로 기둥 위에 간 장면을
"거의 끝"으로 읽어 DPPO가 그 장면으로 보상을 해킹했다. 입력만 바꾼 변형(이미지만·손목만)도 같았다. 그래서 SI-EFM
원형(한 네트워크 + 두 헤드)과 NoMaD·SADP 패턴대로 정책 조건 벡터(global_cond)에 작은 STG 헤드를 붙이고 함께 학습한다.

부모(DiffusionPolicyImage)와 다른 점:
- 인코더: ResNet18+SpatialSoftmax 대신 사전학습 VIP ResNet50(카메라 두 대가 공유). 앞단(vip_frozen_stages)은 얼리고,
  BatchNorm은 늘 eval(작은 배치 미세조정 관례, 롤아웃·RL과 통계가 같다). 입력 전처리는 ① 캐시(cache_vip_feats)와 같다
  (crop 76 -> resize 224 -> ImageNet 정규화). 출력은 ① 체크포인트의 특징 평균·표준편차로 표준화한다.
- stg_head: global_cond -> num_stg_bins 선형층 하나(bias 없음). DstgPredictor(head_hidden=())와 키가 같아
  export_dstg_ckpt가 내보낸 체크포인트를 DstgReward/train_si가 그대로 읽는다.
- compute_loss: bc_mask 행은 디퓨전 MSE, stg_mask 행은 CE. 합 = BC + stg_weight * STG(stg_weight는 학습기가 warmup).

RL(train_si) 중엔 인코더가 얼려져 unet만 바뀌므로 STG 헤드가 읽는 특징 = 보상이 움직이지 않는다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.checkpoint import checkpoint

from square_assembly.networks.conditional_unet1d import ConditionalUnet1d
from square_assembly.policies.diffusion.diffusion_policy_image import DiffusionPolicyImage
from square_assembly.scripts.cache_vip_feats import EMBED_DIM, build_encoder


class VipCameraEncoder(nn.Module):
    """(B,3,H,W) [0,1] -> (B,1024) 표준화된 VIP 특징. convnet은 카메라끼리 같은 객체를 공유한다."""

    def __init__(self, convnet, feat_mean, feat_std, crop_hw, size):
        super().__init__()
        self.convnet = convnet
        self.random_crop = torchvision.transforms.RandomCrop(tuple(crop_hw))
        self.center_crop = torchvision.transforms.CenterCrop(tuple(crop_hw))
        self.size = size
        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.register_buffer("feat_mean", feat_mean.float())
        self.register_buffer("feat_std", feat_std.float())
        self.out_dim = EMBED_DIM
        self.grad_checkpoint = False

    def _features(self, x):
        if not (self.grad_checkpoint and self.training and torch.is_grad_enabled()):
            return self.convnet(x)
        # From: torchvision ResNet._forward_impl — 층(layer1~4) 경계 활성값만 남기고 안쪽은 역전파 때 다시 계산한다.
        # 224px 전체 미세조정이 24GB에서 OOM이라서(2026-10-19 Crong). BN이 늘 eval이라 재계산 결과가 같다.
        net = self.convnet
        x = net.maxpool(net.relu(net.bn1(net.conv1(x))))
        for layer in (net.layer1, net.layer2, net.layer3, net.layer4):
            x = checkpoint(layer, x, use_reentrant=False)
        return net.fc(torch.flatten(net.avgpool(x), 1))

    def forward(self, x):
        x = self.random_crop(x) if self.training else self.center_crop(x)
        x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        # ResNet50 @224가 학습 시간의 대부분이라 인코더만 bf16(UNet은 기본 정책처럼 fp32).
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            f = self._features((x - self.img_mean) / self.img_std)
        return (f.float() - self.feat_mean) / self.feat_std


def _camera_stats(stats_ckpt, rgb_keys):
    """① 체크포인트(train_dstg_vip)의 frame_mean/std에서 카메라별 1024차원 구간을 꺼낸다. None이면 항등.

    중심은 차원별 평균, 크기는 카메라당 스칼라 하나(차원별 표준편차의 RMS)다. VIP 특징은 1024차원 중 대부분이 거의
    상수라(① split0: 2048 중 1707차원이 std < 0.01, 중앙값 0.0005) 차원별로 나누면 미세조정 중 작은 변화가 수천 배로
    커져 STG 손실이 2 epoch 안에 6.5 -> 985로 폭주했다(2026-10-07 스모크). 얼린 ①에선 문제가 없던 표준화다.
    """
    if stats_ckpt is None:
        return {k: (torch.zeros(EMBED_DIM), torch.ones(EMBED_DIM)) for k in rgb_keys}
    ckpt = torch.load(stats_ckpt, map_location="cpu", weights_only=False)
    cams = list(ckpt["cache_meta"]["rgb_keys"])
    mean, std = torch.tensor(ckpt["frame_mean"]), torch.tensor(ckpt["frame_std"])
    out = {}
    for k in rgb_keys:
        s = slice(cams.index(k) * EMBED_DIM, (cams.index(k) + 1) * EMBED_DIM)
        out[k] = (mean[s], std[s].pow(2).mean().sqrt().expand(EMBED_DIM).clone())
    return out


class DiffusionPolicyImageStg(DiffusionPolicyImage):
    def __init__(self, num_stg_bins, vip_pretrained=True, vip_stats_ckpt=None, vip_size=224,
                 vip_frozen_stages=("conv1", "bn1", "layer1", "layer2"), vip_grad_checkpoint=False, **kwargs):
        # ponytail: 부모가 ResNet18 인코더와 작은 조건의 UNet을 만든 뒤 아래서 갈아 끼운다(생성 1~2초 낭비) —
        # 부모 원본을 안 건드리기 위함(ADR-005).
        super().__init__(**kwargs)
        convnet = build_encoder(pretrained=vip_pretrained)
        convnet = convnet.module.convnet if vip_pretrained else convnet   # DataParallel(VIP).convnet = resnet50(fc->1024)
        for name in vip_frozen_stages:
            getattr(convnet, name).requires_grad_(False)
        stats = _camera_stats(vip_stats_ckpt, self.rgb_keys)
        crop_hw = kwargs.get("crop_hw") or kwargs.get("image_hw", (84, 84))
        self.encoders = nn.ModuleDict(
            {k: VipCameraEncoder(convnet, *stats[k], crop_hw, vip_size) for k in self.rgb_keys})
        for enc in self.encoders.values():
            enc.grad_checkpoint = vip_grad_checkpoint

        stage_dim = self.stage_embed.embedding_dim if self.stage_embed is not None else 0
        per_step = len(self.rgb_keys) * EMBED_DIM + sum(kwargs["obs_dims"][k] for k in self.lowdim_keys) + stage_dim
        global_cond_dim = self.obs_horizon * per_step
        self.unet = ConditionalUnet1d(
            input_dim=self.action_dim, global_cond_dim=global_cond_dim,
            down_dims=kwargs.get("down_dims", (256, 512)), kernel_size=kwargs.get("kernel_size", 5),
            n_groups=kwargs.get("n_groups", 8), diffusion_step_embed_dim=kwargs.get("diffusion_step_embed_dim", 256),
        )
        self.stg_head = nn.Sequential(nn.Linear(global_cond_dim, num_stg_bins, bias=False))
        self.stg_weight = 0.0
        self.last_metrics = {}

    def train(self, mode=True):
        super().train(mode)
        for m in self.encoders.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self

    def compute_loss(self, batch, reduction="mean"):
        if "stg_label" not in batch:
            return super().compute_loss(batch, reduction)
        global_cond = self.get_global_cond(batch["obs"])
        bc, stg = batch["bc_mask"], batch["stg_mask"]
        zero = global_cond.sum() * 0.0
        bc_loss = self._bc_loss(global_cond[bc], batch["action"][bc], batch["action_mask"][bc]) if bc.any() else zero
        stg_loss = (F.cross_entropy(self.stg_head(global_cond[stg]), batch["stg_label"][stg].long())
                    if stg.any() else zero)
        self.last_metrics = {"bc_loss": bc_loss.detach(), "stg_loss": stg_loss.detach(), "stg_weight": self.stg_weight}
        return bc_loss + self.stg_weight * stg_loss

    def _bc_loss(self, global_cond, action, action_mask):
        # From: DiffusionPolicyImage.compute_loss(reduction="mean") — global_cond를 STG와 한 번만 계산하려고 분리.
        noise = torch.randn_like(action)
        timesteps = torch.randint(
            0, self.train_scheduler.config.num_train_timesteps, (action.shape[0],), device=action.device).long()
        pred = self.unet(self.train_scheduler.add_noise(action, noise, timesteps), timesteps, global_cond)
        mask = action_mask.unsqueeze(-1).float()
        return (F.mse_loss(pred, noise, reduction="none") * mask).sum() / mask.sum().clamp(min=1.0) / self.action_dim


def export_dstg_ckpt(policy_ckpt_path, out_path):
    """학습된 정책 체크포인트의 stg_head를 DstgReward가 읽는 predictor.pt 형식으로 내보낸다(정책 경로를 그대로 가리킴)."""
    from square_assembly.utils.checkpoints import load_run_config
    from square_assembly.utils.task_utils import task_obs_keys

    model = torch.load(policy_ckpt_path, map_location="cpu", weights_only=False)["model"]
    head = {k[len("stg_head."):]: v for k, v in model.items() if k.startswith("stg_head.")}
    saved = load_run_config(policy_ckpt_path)
    torch.save({"model": head, "num_bins": int(head["0.weight"].shape[0]), "policy_ckpt": policy_ckpt_path,
                "obs_keys": task_obs_keys(saved.task), "head_hidden": []}, out_path)
    return out_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        sys.exit("usage: python -m square_assembly.policies.diffusion.diffusion_policy_stg <policy_epochN.pt> <out predictor.pt>")
    print(export_dstg_ckpt(sys.argv[1], sys.argv[2]))
