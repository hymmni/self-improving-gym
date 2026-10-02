"""DDPO(Black et al. 2023) 핵심 부품: PyTorch/diffusers DDPMScheduler 디퓨전 액션
정책의 단계별 로그확률.

왜 필요한가: SI-EFM Stage-2는 REINFORCE로 정책을 업데이트하는데, 그러려면
`log p(a|o)`가 필요하다. `DiffusionPolicyImage.predict_action_chunk()`의 역확산
100단계(DDPM ancestral sampling)를 적분해야 닫힌 형태의 `log p(a|o)`가 나온다 —
계산이 불가능하다.

DDPO의 우회는 `src/ddpo.py`(JAX 버전, phase 4 step 0)와 동일하다: 역확산의 각
단계는 파라미터에 의존하는 평균과 파라미터 독립적인 표준편차를 갖는 등방 가우시안
전이이므로, "체인 전체"가 아니라 "단계 하나하나"를 정책 그래디언트 대상으로 삼으면
각 단계의 로그확률은 정확히 계산된다. 체인 로그확률은 단계 로그확률의 합. 마지막
단계(diffusers 타임스텝 t=0)는 노이즈를 더하지 않는 결정론적 변환이라 확률밀도가
디랙 델타이므로 합에서 제외한다.

평균·분산 공식은 짐작이 아니라 설치된 `diffusers==0.30.3`의 실제 소스를 직접 읽고
재유도했다 (`DDPMScheduler.step()`, `DDPMScheduler._get_variance()`,
`DDPMScheduler.previous_timestep()` — `python -c "import diffusers, inspect;
print(inspect.getsource(...))"`로 확인). 기본 설정(`variance_type='fixed_small'`,
`clip_sample=True`, `timestep_spacing='leading'`, `prediction_type='epsilon'`)
기준:

  prev_t = t - (num_train_timesteps // num_inference_steps)
  alpha_prod_t = alphas_cumprod[t]
  alpha_prod_t_prev = alphas_cumprod[prev_t]  (prev_t < 0 이면 1.0)
  beta_prod_t = 1 - alpha_prod_t;  beta_prod_t_prev = 1 - alpha_prod_t_prev
  current_alpha_t = alpha_prod_t / alpha_prod_t_prev;  current_beta_t = 1 - current_alpha_t

  x0 = (x - sqrt(beta_prod_t) * eps) / sqrt(alpha_prod_t)
  x0 = clip(x0, -clip_sample_range, clip_sample_range)      # clip_sample=True

  mean = [sqrt(alpha_prod_t_prev) * current_beta_t / beta_prod_t] * x0
       + [sqrt(current_alpha_t) * beta_prod_t_prev / beta_prod_t] * x

  variance = clip((1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t, min=1e-20)
  std = sqrt(variance)     # variance_type='fixed_small' 경로 (diffusers 기본값)
  # t == 0 인 스텝만 noise를 더하지 않음 (diffusers step()의 `if t > 0:` 분기)

DDIM(`maybe_speed_up_inference`로 추론 스케줄러를 바꾼 경우)도 eta>0이면 단계마다
노이즈가 들어가 같은 방식이 성립한다 — `_ddim_transition`이 `DDIMScheduler.step(eta=...)`의
평균·표준편차를 그대로 재현한다. eta=0은 결정론적이라 확률이 없다(노이즈 하한 없이는 범위 밖).

`min_std`(노이즈 하한)는 DPPO(Ren et al. 2024)의 탐색 장치다: 단계별 표준편차가 이 값 밑으로
못 내려가게 하고, 원래 결정론적인 마지막 단계에도 노이즈를 넣는다. 샘플링과 로그확률이 같은
하한을 써야 하므로 `sample_with_trace`/`step_logp*`에 같은 값을 넘길 것.

`diffusion_policy_image.py`/`conditional_unet1d.py`는 수정하지 않는다 — 정책
인스턴스에서 `unet`/`inference_scheduler`/`get_global_cond(obs)` 결과를 읽기만
한다(ADR-004/005 준용).
"""

import math

import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler


def _ddpm_posterior(scheduler, x, eps, t):
    """`DDPMScheduler.step()`이 계산하는 posterior 평균과 표준편차를, 배치 내에서
    서로 다른 timestep을 섞어서도 계산할 수 있도록 벡터화한 버전.

    From: diffusers.schedulers.scheduling_ddpm.DDPMScheduler.step /
    DDPMScheduler._get_variance / DDPMScheduler.previous_timestep
    (설치된 diffusers==0.30.3 실제 소스를 읽고 재구현 — 모듈 docstring 참고)

    Args:
      x: (B, ...) 현재 latent (posterior의 조건 x_t).
      eps: (B, ...) `unet(x, t, global_cond)`의 노이즈 예측.
      t: (B,) long tensor. diffusers 타임스텝 값 그대로(0..num_train_timesteps-1).

    Returns:
      mean: x와 같은 shape.
      std: (B,) — 파라미터 독립 상수(등방 가우시안이라 전 차원에 공통).
    """
    device, dtype = x.device, x.dtype
    num_train_timesteps = scheduler.config.num_train_timesteps
    step_size = num_train_timesteps // scheduler.num_inference_steps

    alphas_cumprod = scheduler.alphas_cumprod.to(device=device, dtype=dtype)
    one = alphas_cumprod.new_tensor(1.0)

    t = t.to(device=device, dtype=torch.long)
    prev_t = t - step_size

    alpha_prod_t = alphas_cumprod[t]
    alpha_prod_t_prev = torch.where(prev_t >= 0, alphas_cumprod[prev_t.clamp(min=0)], one)
    beta_prod_t = 1.0 - alpha_prod_t
    beta_prod_t_prev = 1.0 - alpha_prod_t_prev
    current_alpha_t = alpha_prod_t / alpha_prod_t_prev
    current_beta_t = 1.0 - current_alpha_t

    variance = ((1.0 - alpha_prod_t_prev) / (1.0 - alpha_prod_t) * current_beta_t).clamp(min=1e-20)
    std = variance.sqrt()  # variance_type == "fixed_small" (diffusers 기본값)

    # broadcast (B,) -> x.shape for the elementwise arithmetic below
    extra = (1,) * (x.dim() - 1)
    ap_t, ap_tp, bp_t, bp_tp, ca_t, cb_t = (
        v.view(-1, *extra) for v in (alpha_prod_t, alpha_prod_t_prev, beta_prod_t, beta_prod_t_prev,
                                      current_alpha_t, current_beta_t)
    )

    x0 = (x - bp_t.sqrt() * eps) / ap_t.sqrt()
    if scheduler.config.clip_sample:
        x0 = x0.clamp(-scheduler.config.clip_sample_range, scheduler.config.clip_sample_range)

    pred_original_sample_coeff = ap_tp.sqrt() * cb_t / bp_t
    current_sample_coeff = ca_t.sqrt() * bp_tp / bp_t
    mean = pred_original_sample_coeff * x0 + current_sample_coeff * x

    return mean, std


def _ddim_transition(scheduler, x, eps, t, eta):
    """`DDIMScheduler.step(eta=...)`의 평균과 표준편차(배치 안에서 timestep이 섞여도 된다).

    From: diffusers.schedulers.scheduling_ddim.DDIMScheduler.step / DDIMScheduler._get_variance
    (설치된 diffusers==0.30.3 소스 재구현. use_clipped_model_output=False 기본값 —
    `predict_action_chunk`·수집기와 같은 경로라 clip된 x0에서 eps를 다시 구하지 않는다)

    Returns: mean (x와 같은 shape), std (B,) — 마지막 단계(prev_t<0)는 정확히 0.
    """
    alphas_cumprod = scheduler.alphas_cumprod.to(device=x.device, dtype=x.dtype)
    t = t.to(device=x.device, dtype=torch.long)
    prev_t = t - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    extra = (1,) * (x.dim() - 1)
    a_t = alphas_cumprod[t].view(-1, *extra)
    a_prev = torch.where(prev_t >= 0, alphas_cumprod[prev_t.clamp(min=0)],
                         scheduler.final_alpha_cumprod.to(alphas_cumprod)).view(-1, *extra)

    x0 = (x - (1.0 - a_t).sqrt() * eps) / a_t.sqrt()
    if scheduler.config.clip_sample:
        x0 = x0.clamp(-scheduler.config.clip_sample_range, scheduler.config.clip_sample_range)
    std = eta * ((1.0 - a_prev) / (1.0 - a_t) * (1.0 - a_t / a_prev)).sqrt()
    mean = a_prev.sqrt() * x0 + (1.0 - a_prev - std ** 2).sqrt() * eps
    return mean, std.flatten()


def _transition(scheduler, x, eps, t, eta=1.0, min_std=None):
    """추론 스케줄러 한 단계의 가우시안 전이 (mean, std). std는 (B,)이고 결정론적 단계는 0.

    DDPM은 t=0에서, DDIM은 마지막 단계에서 노이즈를 더하지 않는다 — `min_std`를 주면 그 단계까지
    포함해 전부 그 값 이상으로 올린다(DPPO의 min_sampling/logprob_denoising_std).
    """
    if isinstance(scheduler, DDIMScheduler):
        mean, std = _ddim_transition(scheduler, x, eps, t, eta)
    else:
        mean, std = _ddpm_posterior(scheduler, x, eps, t)
        std = torch.where(t.to(std.device) > 0, std, torch.zeros_like(std))
    if min_std:
        std = std.clamp(min=min_std)
    return mean, std


def _expand_t(t, batch_size):
    """0-d 텐서(배치 전체가 같은 t, `scheduler.timesteps`를 순회할 때)를 (B,)로 편다."""
    return t.expand(batch_size) if t.dim() == 0 else t


@torch.no_grad()
def sample_with_trace(unet, scheduler, global_cond, shape, device, generator=None, eta=1.0, min_std=None):
    """`policy.predict_action_chunk()`와 **수치적으로 동일한** 샘플을 내되, 중간
    latent 전체를 함께 반환한다.

    `scheduler`는 호출 전에 `set_timesteps(...)`가 이미 적용된 상태여야 한다
    (`policy.inference_scheduler`를 그대로 넘기면, `predict_action_chunk` 호출이
    설정해둔 `.timesteps`/`.num_inference_steps`를 그대로 재사용한다).

    난수 소비 순서를 `predict_action_chunk`와 정확히 맞춘다: 초기 노이즈 1회 +
    매 역확산 스텝마다(표준편차가 0인 마지막 단계 제외) 노이즈 1회 — diffusers
    `DDPMScheduler.step()`의 `if t > 0:` 분기와 동일하다. `min_std`를 주면 마지막 단계에도
    노이즈가 들어가므로 더는 `predict_action_chunk`와 같지 않다(DPPO 탐색 노이즈).

    Args:
      shape: (B, pred_horizon, action_dim).
      generator: torch.Generator(device=device) 또는 None.

    Returns:
      sample_final: (B, Tp, Da)
      xs: (n_steps+1, B, Tp, Da) — xs[0]=초기 노이즈, xs[n_steps]=sample_final,
          xs[i]는 i번째 역확산 스텝 적용 직전 latent.
    """
    b = shape[0]
    x = torch.randn(shape, generator=generator, device=device)
    xs = [x]
    for t in scheduler.timesteps:
        eps = unet(x, t, global_cond)
        mean, std = _transition(scheduler, x, eps, _expand_t(t, b), eta, min_std)
        if bool((std > 0).all()):
            noise = torch.randn(x.shape, generator=generator, device=device)
            x = mean + std.view(-1, *([1] * (x.dim() - 1))) * noise
        else:
            x = mean
        xs.append(x)
    if min_std:
        # From: irom-princeton/dppo model/diffusion/diffusion_vpg.py (forward, final_action_clip_value)
        # 마지막 단계의 탐색 노이즈가 행동을 정규화 범위 밖으로 밀어내지 않게 자른다. 로그확률은
        # 원본처럼 자른 값에 가우시안을 그대로 적용한다(잘림 보정 없음).
        r = scheduler.config.clip_sample_range
        x = xs[-1] = x.clamp(-r, r)
    xs = torch.stack(xs, dim=0)
    return x, xs


def step_logp_elem(unet, scheduler, global_cond, x_in, x_out, timesteps, eta=1.0, min_std=None):
    """(env-step, 역확산-단계) 쌍의 미니배치에 대한 한 단계 로그확률 — 원소별 (B, Tp, Da).

    x_in, x_out: (B, Tp, Da). timesteps: (B,) long — diffusers 타임스텝 값 그대로.
    표준편차가 0인 단계(`min_std` 없이 마지막 단계)는 결정론적 전이라 로그확률이 정의되지
    않으므로 호출자가 걸러서 절대 넣지 말 것.

    unet 파라미터에 대해 미분 가능하다(grad 필요) — `torch.no_grad()`로 감싸지 않는다.
    """
    eps = unet(x_in, timesteps, global_cond)
    mean, std = _transition(scheduler, x_in, eps, timesteps, eta, min_std)
    std = std.view(-1, *([1] * (x_in.dim() - 1)))
    return -0.5 * ((x_out - mean) / std) ** 2 - torch.log(std) - 0.5 * math.log(2.0 * math.pi)


def step_logp(unet, scheduler, global_cond, x_in, x_out, timesteps, eta=1.0, min_std=None):
    """`step_logp_elem`을 청크 전체 차원에 대해 더한 (B,) — 등방 가우시안의 결합 로그확률."""
    return step_logp_elem(unet, scheduler, global_cond, x_in, x_out, timesteps, eta, min_std).flatten(1).sum(dim=-1)


def chain_logp(unet, scheduler, global_cond, xs, timesteps_all):
    """한 샘플의 체인 전체 로그확률 합(t=0 제외 전부). 진단·테스트용.

    xs: (n_steps+1, B, Tp, Da) — `sample_with_trace`가 반환한 것과 같은 규약.
    timesteps_all: `scheduler.timesteps`에서 t=0을 제외한 전체 목록(길이 n_steps-1,
    `xs[i]`--(kk번째 역확산 단계)-->`xs[i+1]` 순서와 일치).
    """
    b = xs.shape[1]
    total = xs.new_zeros(b)
    for i, t in enumerate(timesteps_all):
        t_batch = _expand_t(t, b)
        total = total + step_logp(unet, scheduler, global_cond, xs[i], xs[i + 1], t_batch)
    return total
