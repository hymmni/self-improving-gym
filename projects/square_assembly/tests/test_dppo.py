"""DPPO 부품(디노이징 할인·클립 일정, 로그확률 축약, 업데이트 한 번) — env·체크포인트 없이 돈다."""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from square_assembly.policies.diffusion import ddpo, dppo
from square_assembly.policies.diffusion.diffusion_policy_image import DiffusionPolicyImage
from square_assembly.runners.rollout import maybe_speed_up_inference


def test_denoising_weights_follow_the_dppo_schedule():
    discount, clip = dppo.denoising_weights(np.arange(5), 5, gamma_denoising=0.99, clip=0.01, clip_base=0.001, clip_rate=3)
    assert discount[-1] == pytest.approx(1.0) and discount[0] == pytest.approx(0.99 ** 4)   # 마지막(가장 깨끗한) 단계가 1
    assert clip[0] == pytest.approx(0.001) and clip[-1] == pytest.approx(0.01)
    assert (np.diff(clip) > 0).all() and (np.diff(discount) > 0).all()

    discount, clip = dppo.denoising_weights(np.arange(1), 1, gamma_denoising=0.99, clip=0.01, clip_base=0.001, clip_rate=3)
    assert discount[0] == pytest.approx(1.0) and clip[0] == pytest.approx(0.01)


def test_reduce_logp_clamps_then_averages_only_the_executed_steps():
    lp = torch.tensor([[[1.0, -9.0], [0.0, 3.0], [100.0, 100.0]]])     # (1, 3스텝, 2차원), 실행은 앞 2스텝
    assert dppo.reduce_logp(lp, horizon=2).tolist() == pytest.approx([(1.0 - 5.0 + 0.0 + 2.0) / 4])


def test_dppo_update_weights_each_denoising_step_by_its_discount():
    from square_assembly.scripts.train_si import _bind_sampler, _dppo_update
    torch.manual_seed(0)
    policy = DiffusionPolicyImage(rgb_keys=["cam"], lowdim_keys=["state"], obs_dims={"state": 4}, obs_horizon=2,
                                  action_dim=3, pred_horizon=4, num_kp=4, image_hw=(32, 32), down_dims=(32, 64),
                                  num_train_timesteps=10, num_inference_steps=10).eval()
    maybe_speed_up_inference(policy, 5, eta=1.0)
    fns = _bind_sampler(eta=1.0, min_std=0.1)
    scheduler = policy.inference_scheduler
    scheduler.set_timesteps(5)

    n = 6                                                                # 결정 6개 = 에피소드 2개 x 3
    obs = {"cam": torch.rand(n, 2, 3, 32, 32), "state": torch.randn(n, 2, 4)}
    with torch.no_grad():
        cond = policy.get_global_cond(obs)
        _, xs = fns.sample_with_trace(policy.unet, scheduler, cond, (n, 4, 3), "cpu")
    xs = xs.transpose(0, 1).contiguous()                                 # (N, n_steps+1, Tp, Da)
    rewards = np.array([1.0, -2.0, 3.0, 0.5, 0.0, -1.0], dtype=np.float32)

    cfg = OmegaConf.create(dict(
        ft_denoising_steps=None, logp_batch=1000, gamma=0.9, gae_lambda=0.95, advantage_norm=False, seed0=0,
        update_epochs=1, clip_ratio=0.01, clip_ratio_base=0.001, clip_ratio_rate=3, gamma_denoising=0.5,
        max_grad_norm=1.0, value_coef=0.5, policy=dict(action_horizon=2)))
    critic = dppo.Critic(cond.shape[-1], hidden=(8,))
    with torch.no_grad():
        values = critic(cond).numpy()
    adv = np.concatenate([dppo.gae(rewards[s], values[s], 0.9, 0.95)[0] for s in (slice(0, 3), slice(3, 6))])
    before = [p.clone() for p in policy.unet.parameters()]

    losses, info = _dppo_update(policy, fns, torch.optim.Adam(policy.unet.parameters(), lr=1e-3), critic,
                                torch.optim.Adam(critic.parameters(), lr=1e-3), cond, xs, rewards, [3, 3],
                                scheduler.timesteps, 5, cfg, torch.device("cpu"), it=1)

    # 배치가 하나라 첫 업데이트의 비율은 정확히 1 — 손실은 -mean(어드밴티지 x 디노이징 할인)이다.
    discount = 0.5 ** (4 - np.arange(5))
    assert losses[0] == pytest.approx(-(adv[:, None] * discount[None, :]).mean(), rel=1e-4)
    assert info["ratio_mean"] == pytest.approx(1.0, abs=1e-5) and info["clip_frac"] == 0.0
    assert any(not torch.equal(a, b) for a, b in zip(before, policy.unet.parameters()))


def _tiny_update_setup():
    from square_assembly.scripts.train_si import _bind_sampler
    torch.manual_seed(0)
    policy = DiffusionPolicyImage(rgb_keys=["cam"], lowdim_keys=["state"], obs_dims={"state": 4}, obs_horizon=2,
                                  action_dim=3, pred_horizon=4, num_kp=4, image_hw=(32, 32), down_dims=(32, 64),
                                  num_train_timesteps=10, num_inference_steps=10).eval()
    maybe_speed_up_inference(policy, 5, eta=1.0)
    fns = _bind_sampler(eta=1.0, min_std=0.1)
    policy.inference_scheduler.set_timesteps(5)
    obs = {"cam": torch.rand(6, 2, 3, 32, 32), "state": torch.randn(6, 2, 4)}
    with torch.no_grad():
        cond = policy.get_global_cond(obs)
        _, xs = fns.sample_with_trace(policy.unet, policy.inference_scheduler, cond, (6, 4, 3), "cpu")
    return policy, fns, cond, xs.transpose(0, 1).contiguous()


def _run_update(policy, fns, cond, xs, critic, critic_opt, **over):
    from square_assembly.scripts.train_si import _dppo_update
    base = dict(
        ft_denoising_steps=None, logp_batch=1000, gamma=0.9, gae_lambda=0.95, advantage_norm=False, seed0=0,
        update_epochs=1, clip_ratio=0.01, clip_ratio_base=0.001, clip_ratio_rate=3, gamma_denoising=0.5,
        max_grad_norm=None, value_coef=0.5, policy=dict(action_horizon=2))
    cfg = OmegaConf.create({**base, **over})
    rewards = np.array([1.0, -2.0, 3.0, 0.5, 0.0, -1.0], dtype=np.float32)
    return _dppo_update(policy, fns, torch.optim.SGD(policy.unet.parameters(), lr=1e-2), critic, critic_opt,
                        cond, xs, rewards, [3, 3], policy.inference_scheduler.timesteps, 5, cfg, torch.device("cpu"), it=1)


def test_ppo_batch_accumulates_memory_chunks_into_one_optimizer_step():
    """ppo_batch(옵티마이저 한 스텝의 쌍 수)는 logp_batch(GPU 메모리 청크)와 별개 — 청크 그래디언트를 모아 한 번만 step.
    전체 30쌍을 7쌍 청크로 나눠도 한 덩어리로 한 번 밟은 것과 같아야 한다(DPPO 원본: 큰 배치 + 그래디언트 누적)."""
    import copy
    policy, fns, cond, xs = _tiny_update_setup()
    critic = dppo.Critic(cond.shape[-1], hidden=(8,))
    p_one, p_chunk = copy.deepcopy(policy), copy.deepcopy(policy)
    losses_one, _ = _run_update(p_one, fns, cond, xs, copy.deepcopy(critic), torch.optim.SGD(critic.parameters(), lr=0.0))
    losses_chunk, _ = _run_update(p_chunk, fns, cond, xs, copy.deepcopy(critic), torch.optim.SGD(critic.parameters(), lr=0.0),
                                  logp_batch=7, ppo_batch=30)
    assert len(losses_one) == len(losses_chunk) == 1
    assert losses_chunk[0] == pytest.approx(losses_one[0], rel=1e-4)
    for a, b in zip(p_one.unet.parameters(), p_chunk.unet.parameters()):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)


def test_critic_can_step_with_every_ppo_minibatch():
    """critic_per_minibatch=true면 크리틱도 PPO 미니배치마다 한 스텝(DPPO 원본 Algorithm 1, 21행) — 기본은 epoch마다 한 스텝."""
    policy, fns, cond, xs = _tiny_update_setup()
    counts = {}
    for flag in (False, True):
        critic = dppo.Critic(cond.shape[-1], hidden=(8,))
        opt = torch.optim.SGD(critic.parameters(), lr=1e-3)
        n = [0]
        step = opt.step
        opt.step = lambda *a, **k: (n.__setitem__(0, n[0] + 1), step(*a, **k))[1]
        _run_update(policy, fns, cond, xs, critic, opt, logp_batch=10, update_epochs=2, critic_per_minibatch=flag)
        counts[flag] = n[0]
    assert counts == {False: 2, True: 6}   # 30쌍 / 10 = 미니배치 3개 x 2 epoch
