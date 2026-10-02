"""VIP 특징 캐시로 학습한 STG 예측기(train_dstg_vip.py)를 롤아웃 중 실시간 프레임에 붙이는 보상 래퍼.

DstgReward(dstg_reward.py)는 얼린 정책 인코더(ResNet) 예측기만 감싼다. 2026-10-01 비교에서 ResNet 예측기는
held-out 정확도는 VIP와 같지만 너트를 떨어뜨리는 순간에 거의 반응하지 않아(experiments/2026-10-01_stg-resnet-vs-vip.md)
RL 보상에는 VIP 예측기를 쓰기로 했다. 원본은 건드리지 않고 상속으로 예측기만 바꾼다(ADR-005) —
d/success/threshold/CVaR/calibrate_threshold는 DstgReward 것을 그대로 쓴다.

입력은 DstgReward.d와 같다: 정책이 받는 obs 배치(rgb (B,To,C,H,W) [0,1], lowdim은 min-max 정규화).
학습 때와 같은 값이 헤드에 들어가도록 캐시 스크립트의 encode_frames를 그대로 부르고, lowdim은 정규화를 풀어
원래 값으로 되돌린 뒤 체크포인트의 frame_mean/frame_std로 표준화한다.

vip 패키지와 가중치(~/.vip)가 필요하다 — DOCKER.md §6-1(HOME을 쓰기 가능한 곳으로 돌려야 받는다).
"""

import torch
import torch.nn as nn

from square_assembly.policies.diffusion.dino_stg_predictor import load_checkpoint
from square_assembly.policies.diffusion.dstg_reward import DstgReward
from square_assembly.scripts.cache_vip_feats import build_encoder, encode_frames


class VipStgPredictor(nn.Module):
    """정책용 obs 배치 -> STG logits. cache_vip_feats(특징) + DinoFeatureWindows(창·표준화) + DinoStgHead와 같은 계산."""

    def __init__(self, head, ckpt, normalizer, encoder):
        super().__init__()
        if ckpt.get("obs_parts", "all") != "all" or ckpt.get("fail_bin") is not None:
            raise ValueError("obs_parts=all, fail_bin 없는 VIP 예측기만 지원한다")
        meta = ckpt["cache_meta"]
        self.head, self.encoder, self.normalizer = head, encoder, normalizer
        self.rgb_keys, self.lowdim_keys = list(meta["rgb_keys"]), list(meta["lowdim_keys"])
        self.crop, self.size = meta["crop"], meta["size"]
        self.register_buffer("frame_mean", torch.tensor(ckpt["frame_mean"], dtype=torch.float32))
        self.register_buffer("frame_std", torch.tensor(ckpt["frame_std"], dtype=torch.float32))

    def forward(self, obs):
        device = self.frame_mean.device
        b, to = obs[self.rgb_keys[0]].shape[:2]
        parts = []
        for k in self.rgb_keys:
            # [0,1] float -> 원래 uint8 (T,H,W,3). encode_frames가 캐시와 같은 전처리·float16 반올림을 한다.
            u8 = (obs[k].reshape(b * to, *obs[k].shape[2:]) * 255).round().clamp(0, 255).byte()
            feat = encode_frames(self.encoder, u8.permute(0, 2, 3, 1).cpu().numpy(), self.crop, self.size, device)
            parts.append(torch.from_numpy(feat).float().to(device))
        low = self.normalizer.unnormalize_obs({k: obs[k].cpu() for k in self.lowdim_keys})
        parts += [low[k].reshape(b * to, -1).to(device) for k in self.lowdim_keys]
        frames = (torch.cat(parts, dim=-1) - self.frame_mean) / self.frame_std
        return self.head(frames.reshape(b, -1))


class VipStgReward(DstgReward):
    """VIP 예측기 체크포인트(train_dstg_vip.py의 predictor.pt/best_predictor.pt)를 DstgReward 인터페이스로 감싼다.

    normalizer: 정책의 MinMaxNormalizer — obs 배치의 lowdim 정규화를 풀 때 쓴다.
    encoder: 테스트용 주입구(None이면 사전학습 VIP를 불러온다).
    """

    def __init__(self, ckpt_path, normalizer, statistic="mean", cvar_alpha=0.8, threshold=None,
                 device="cpu", encoder=None):
        device = torch.device(device)
        head, ckpt = load_checkpoint(ckpt_path, device)
        encoder = encoder if encoder is not None else build_encoder(pretrained=True, device=device)
        predictor = VipStgPredictor(head, ckpt, normalizer, encoder)
        self.policy_ckpt_path = None
        self.rgb_keys = predictor.rgb_keys
        self.obs_keys = predictor.rgb_keys + predictor.lowdim_keys
        self.obs_horizon = int(ckpt["obs_horizon"])
        self._init_common(predictor, ckpt["num_bins"], statistic, cvar_alpha, threshold, device)


def load_stg_reward(ckpt_path, normalizer, **kwargs):
    """체크포인트 종류에 맞는 보상 래퍼를 돌려준다 — VIP 캐시 예측기엔 cache_meta가, ResNet 예측기엔 policy_ckpt가 있다."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return VipStgReward(ckpt_path, normalizer, **kwargs) if "cache_meta" in ckpt else DstgReward(ckpt_path, **kwargs)
