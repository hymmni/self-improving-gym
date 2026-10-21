"""DDPO(PyTorch/diffusers) 핵심 로그확률 함수 검증.

phases/5-mani-sim-ddpo/step0.md의 인터페이스(`sample_with_trace`/`step_logp`/
`chain_logp`)를 대상으로, 아주 작은 랜덤 초기화 `DiffusionPolicyImage`로 빠르게
돌게 구성한다 — 체크포인트·robomimic 데이터 불필요.
"""

import math

import pytest
import torch

from square_assembly.policies.diffusion.ddpo import _ddpm_posterior, chain_logp, sample_with_trace, step_logp
from square_assembly.policies.diffusion.diffusion_policy_image import DiffusionPolicyImage

DEVICE = "cpu"
IMAGE_HW = (32, 32)


def _make_policy(num_train_timesteps=10, num_inference_steps=10, init_seed=0):
    torch.manual_seed(init_seed)
    policy = DiffusionPolicyImage(
        rgb_keys=["cam"],
        lowdim_keys=["state"],
        obs_dims={"state": 4},
        obs_horizon=2,
        action_dim=3,
        pred_horizon=4,
        num_kp=4,
        image_hw=IMAGE_HW,
        down_dims=(32, 64),
        num_train_timesteps=num_train_timesteps,
        num_inference_steps=num_inference_steps,
    )
    policy.eval()
    return policy.to(DEVICE)


def _make_obs(batch_size=2, obs_horizon=2, device=DEVICE):
    return {
        "cam": torch.rand(batch_size, obs_horizon, 3, *IMAGE_HW, device=device),
        "state": torch.randn(batch_size, obs_horizon, 4, device=device),
    }


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_trace_matches_predict_action_chunk(seed):
    policy = _make_policy()
    obs = _make_obs()

    torch.manual_seed(seed)
    action_ref = policy.predict_action_chunk(obs)

    with torch.no_grad():
        global_cond = policy.get_global_cond(obs)
    shape = (obs["state"].shape[0], policy.pred_horizon, policy.action_dim)
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    action_trace, xs = sample_with_trace(
        policy.unet, policy.inference_scheduler, global_cond, shape, DEVICE, generator=generator
    )

    assert torch.allclose(action_trace, action_ref, atol=1e-5)
    assert xs.shape == (policy.num_inference_steps + 1, *shape)
    assert torch.allclose(xs[-1], action_ref, atol=1e-5)


def test_step_logp_matches_empirical_distribution():
    policy = _make_policy()
    obs = _make_obs(batch_size=1)
    with torch.no_grad():
        global_cond = policy.get_global_cond(obs)

    policy.inference_scheduler.set_timesteps(policy.num_inference_steps, device=DEVICE)
    t = policy.inference_scheduler.timesteps[3]
    assert t.item() > 0, "테스트는 t=0(결정론적 스텝)이 아닌 중간 스텝을 대상으로 해야 한다"

    torch.manual_seed(42)
    x_in = torch.randn(1, policy.pred_horizon, policy.action_dim, device=DEVICE)
    with torch.no_grad():
        eps = policy.unet(x_in, t, global_cond)

    n_samples = 8000
    generator = torch.Generator(device=DEVICE).manual_seed(123)
    with torch.no_grad():
        outs = torch.cat(
            [policy.inference_scheduler.step(eps, t, x_in, generator=generator).prev_sample for _ in range(n_samples)],
            dim=0,
        )
    empirical_mean = outs.mean(dim=0)
    empirical_std = outs.std(dim=0)

    analytic_mean, analytic_std = _ddpm_posterior(policy.inference_scheduler, x_in, eps, t.expand(1))
    analytic_mean = analytic_mean[0]
    analytic_std = analytic_std.item()

    assert torch.allclose(empirical_mean, analytic_mean, atol=0.08)
    rel_err = (empirical_std - analytic_std).abs() / analytic_std
    assert (rel_err < 0.1).all(), f"relative std error too large: {rel_err}"

    # step_logp가 내부적으로 같은 (mean, std)를 쓰는지도 함께 확인한다: 표본 평균
    # 위치에서의 로그확률이 그 Gaussian의 최댓값(정규화 상수) 근방과 일치해야 한다.
    logp_at_mean = step_logp(policy.unet, policy.inference_scheduler, global_cond, x_in, analytic_mean[None], t.expand(1))
    d = analytic_mean.numel()
    expected_peak = -d * math.log(analytic_std) - 0.5 * d * math.log(2.0 * math.pi)
    assert torch.allclose(logp_at_mean, torch.tensor([expected_peak]), atol=1e-3)


def test_chain_logp_gradient_flows_to_unet_only():
    policy = _make_policy()
    obs = _make_obs()

    # 설계 규약: global_cond는 no_grad(또는 .detach())로 만들어 DDPO 업데이트가
    # unet만 겨냥하게 한다 — 인코더는 뒤 step에서 얼려 재사용해야 한다.
    with torch.no_grad():
        global_cond = policy.get_global_cond(obs)

    policy.inference_scheduler.set_timesteps(policy.num_inference_steps, device=DEVICE)
    shape = (obs["state"].shape[0], policy.pred_horizon, policy.action_dim)
    generator = torch.Generator(device=DEVICE).manual_seed(7)
    _, xs = sample_with_trace(policy.unet, policy.inference_scheduler, global_cond, shape, DEVICE, generator=generator)

    timesteps_all = policy.inference_scheduler.timesteps[:-1]  # t=0 제외
    logp = chain_logp(policy.unet, policy.inference_scheduler, global_cond, xs, timesteps_all)
    logp.sum().backward()

    assert any(p.grad is not None and torch.any(p.grad != 0) for p in policy.unet.parameters())
    for p in policy.encoders.parameters():
        assert p.grad is None


# ---------------------------------------------------------------- DDIM(eta>0) 전이·노이즈 하한

def _use_ddim(policy, steps=5, eta=1.0):
    from square_assembly.runners.rollout import maybe_speed_up_inference
    maybe_speed_up_inference(policy, steps, eta=eta)
    return policy


@pytest.mark.parametrize("seed", [0, 1])
def test_trace_matches_predict_action_chunk_with_ddim_eta1(seed):
    policy = _use_ddim(_make_policy())
    obs = _make_obs()

    torch.manual_seed(seed)
    action_ref = policy.predict_action_chunk(obs)

    with torch.no_grad():
        global_cond = policy.get_global_cond(obs)
    shape = (obs["state"].shape[0], policy.pred_horizon, policy.action_dim)
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    action_trace, xs = sample_with_trace(
        policy.unet, policy.inference_scheduler, global_cond, shape, DEVICE, generator=generator
    )

    assert torch.allclose(action_trace, action_ref, atol=1e-5)
    assert xs.shape == (policy.num_inference_steps + 1, *shape)


def test_ddim_transition_matches_empirical_distribution():
    from square_assembly.policies.diffusion.ddpo import _transition
    policy = _use_ddim(_make_policy())
    scheduler = policy.inference_scheduler
    scheduler.set_timesteps(policy.num_inference_steps, device=DEVICE)
    t = scheduler.timesteps[1]

    torch.manual_seed(42)
    x_in = 3.0 * torch.randn(1, policy.pred_horizon, policy.action_dim)
    eps = torch.randn_like(x_in)
    a_t = scheduler.alphas_cumprod[t]
    x0 = (x_in - (1 - a_t).sqrt() * eps) / a_t.sqrt()
    assert (x0.abs() > 1).any(), "x0 clip이 걸려야 DDPM posterior 공식과 구별된다"

    generator = torch.Generator(device=DEVICE).manual_seed(123)
    outs = torch.cat([scheduler.step(eps, t, x_in, eta=1.0, generator=generator).prev_sample for _ in range(8000)])

    mean, std = _transition(scheduler, x_in, eps, t.expand(1))
    assert torch.allclose(outs.mean(dim=0), mean[0], atol=0.08)
    assert ((outs.std(dim=0) - std.item()).abs() / std.item() < 0.1).all()


def test_min_std_floors_every_step_and_gives_the_last_step_a_logp():
    from square_assembly.policies.diffusion.ddpo import _transition
    policy = _use_ddim(_make_policy())
    scheduler = policy.inference_scheduler
    scheduler.set_timesteps(policy.num_inference_steps, device=DEVICE)
    x = torch.randn(1, policy.pred_horizon, policy.action_dim)
    eps = torch.randn_like(x)

    raw = [_transition(scheduler, x, eps, t.expand(1))[1].item() for t in scheduler.timesteps]
    floored = [_transition(scheduler, x, eps, t.expand(1), min_std=0.1)[1].item() for t in scheduler.timesteps]
    assert raw[-1] == 0.0                                    # 마지막 단계는 원래 결정적 — 로그확률이 없다
    assert floored == pytest.approx([max(s, 0.1) for s in raw])

    with torch.no_grad():
        global_cond = policy.get_global_cond(_make_obs(batch_size=1))
        _, xs = sample_with_trace(policy.unet, scheduler, global_cond, x.shape, DEVICE, min_std=0.1)
        lp = step_logp(policy.unet, scheduler, global_cond, xs[-2], xs[-1], scheduler.timesteps[-1].expand(1), min_std=0.1)
    assert torch.isfinite(lp).all()
    assert not torch.equal(xs[-1], _transition(scheduler, xs[-2], policy.unet(xs[-2], scheduler.timesteps[-1], global_cond),
                                               scheduler.timesteps[-1].expand(1))[0])   # 마지막 단계에도 노이즈가 들어갔다


def test_min_std_keeps_the_final_action_inside_the_normalized_range():
    policy = _use_ddim(_make_policy())
    scheduler = policy.inference_scheduler
    scheduler.set_timesteps(policy.num_inference_steps, device=DEVICE)
    with torch.no_grad():
        global_cond = policy.get_global_cond(_make_obs(batch_size=16))
    generator = torch.Generator(device=DEVICE).manual_seed(0)
    action, xs = sample_with_trace(policy.unet, scheduler, global_cond, (16, policy.pred_horizon, policy.action_dim),
                                   DEVICE, generator=generator, min_std=0.1)
    assert action.abs().max() <= scheduler.config.clip_sample_range   # 마지막 단계 노이즈가 범위 밖으로 밀어내면 안 된다
    assert torch.equal(xs[-1], action)                                # 로그확률은 실제 실행한(자른) 행동으로 잰다
