"""diffusion_policy.py / diffusion_policy_image.py 를 factory registry에 등록.

정책 클래스 자체(생성자 시그니처)는 손대지 않는다 — 여기선 task_cfg/policy_cfg를 그
클래스가 원하는 kwargs로 펼쳐주는 빌더 함수만 등록한다.
"""

from square_assembly.factory import registry
from square_assembly.policies.diffusion.diffusion_policy import DiffusionPolicyLowDim
from square_assembly.policies.diffusion.diffusion_policy_image import DiffusionPolicyImage


@registry.register_policy("diffusion_lowdim")
def _build_diffusion_lowdim(task_cfg, policy_cfg):
    return DiffusionPolicyLowDim(
        obs_keys=task_cfg.obs_keys,
        obs_dims=task_cfg.obs_dims,
        obs_horizon=policy_cfg.obs_horizon,
        action_dim=task_cfg.action_dim,
        pred_horizon=policy_cfg.pred_horizon,
        down_dims=policy_cfg.down_dims,
        kernel_size=policy_cfg.kernel_size,
        n_groups=policy_cfg.n_groups,
        diffusion_step_embed_dim=policy_cfg.diffusion_step_embed_dim,
        num_train_timesteps=policy_cfg.num_train_timesteps,
        beta_schedule=policy_cfg.beta_schedule,
        num_inference_steps=policy_cfg.num_inference_steps,
    )


@registry.register_policy("diffusion")
def _build_diffusion(task_cfg, policy_cfg):
    return DiffusionPolicyImage(
        rgb_keys=task_cfg.rgb_keys,
        lowdim_keys=task_cfg.lowdim_keys,
        obs_dims=task_cfg.obs_dims,
        obs_horizon=policy_cfg.obs_horizon,
        action_dim=task_cfg.action_dim,
        pred_horizon=policy_cfg.pred_horizon,
        num_kp=policy_cfg.get("num_kp", 32),
        image_hw=policy_cfg.get("image_hw", [task_cfg.image_size, task_cfg.image_size]),
        crop_hw=policy_cfg.get("crop_hw", None),
        down_dims=policy_cfg.down_dims,
        kernel_size=policy_cfg.kernel_size,
        n_groups=policy_cfg.n_groups,
        diffusion_step_embed_dim=policy_cfg.diffusion_step_embed_dim,
        num_train_timesteps=policy_cfg.num_train_timesteps,
        beta_schedule=policy_cfg.beta_schedule,
        num_inference_steps=policy_cfg.num_inference_steps,
        # stage-conditioning(2026-08-04, robocasa 실험) — task.num_stages가 없으면 둘 다
        # None/무시되고 기존 동작 그대로(하위호환). 근거: robocasa/research_design.md.
        num_stages=task_cfg.get("num_stages", None),
        stage_embed_dim=policy_cfg.get("stage_embed_dim", 32),
    )


@registry.register_policy("diffusion_stg")
def _build_diffusion_stg(task_cfg, policy_cfg):
    # 공유 백본(VIP) + STG 헤드(diffusion_policy_stg.py). 무거운 import(vip)는 이 정책을 만들 때만.
    from square_assembly.policies.diffusion.diffusion_policy_stg import DiffusionPolicyImageStg

    return DiffusionPolicyImageStg(
        rgb_keys=task_cfg.rgb_keys,
        lowdim_keys=task_cfg.lowdim_keys,
        obs_dims=task_cfg.obs_dims,
        obs_horizon=policy_cfg.obs_horizon,
        action_dim=task_cfg.action_dim,
        pred_horizon=policy_cfg.pred_horizon,
        image_hw=policy_cfg.get("image_hw", [task_cfg.image_size, task_cfg.image_size]),
        crop_hw=policy_cfg.get("crop_hw", None),
        down_dims=policy_cfg.down_dims,
        kernel_size=policy_cfg.kernel_size,
        n_groups=policy_cfg.n_groups,
        diffusion_step_embed_dim=policy_cfg.diffusion_step_embed_dim,
        num_train_timesteps=policy_cfg.num_train_timesteps,
        beta_schedule=policy_cfg.beta_schedule,
        num_inference_steps=policy_cfg.num_inference_steps,
        num_stg_bins=policy_cfg.stg.num_bins,
        vip_pretrained=policy_cfg.get("vip_pretrained", True),
        vip_stats_ckpt=policy_cfg.get("vip_stats_ckpt", None),
        vip_size=policy_cfg.get("vip_size", 224),
        vip_frozen_stages=tuple(policy_cfg.get("vip_frozen_stages", ("conv1", "bn1", "layer1", "layer2"))),
        vip_grad_checkpoint=policy_cfg.get("vip_grad_checkpoint", False),
    )


@registry.register_policy("diffusion_dp")
def _build_diffusion_dp(task_cfg, policy_cfg):
    # DP 원저자 공개 체크포인트 호환(diffusion_policy_dp.py) — 인코더·청크 시작점만 다르다.
    from square_assembly.policies.diffusion.diffusion_policy_dp import DiffusionPolicyImageDP

    return DiffusionPolicyImageDP(
        rgb_keys=task_cfg.rgb_keys,
        lowdim_keys=task_cfg.lowdim_keys,
        obs_dims=task_cfg.obs_dims,
        obs_horizon=policy_cfg.obs_horizon,
        action_dim=task_cfg.action_dim,
        pred_horizon=policy_cfg.pred_horizon,
        num_kp=policy_cfg.get("num_kp", 32),
        image_hw=[task_cfg.image_size, task_cfg.image_size],
        crop_hw=list(policy_cfg.crop_hw),
        down_dims=policy_cfg.down_dims,
        kernel_size=policy_cfg.kernel_size,
        n_groups=policy_cfg.n_groups,
        diffusion_step_embed_dim=policy_cfg.diffusion_step_embed_dim,
        num_train_timesteps=policy_cfg.num_train_timesteps,
        beta_schedule=policy_cfg.beta_schedule,
        num_inference_steps=policy_cfg.num_inference_steps,
    )
