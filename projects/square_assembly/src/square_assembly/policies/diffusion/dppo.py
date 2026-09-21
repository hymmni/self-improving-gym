r"""DPPO(Ren et al. 2024, *Diffusion Policy Policy Optimization*) 부품 — 크리틱, GAE,
클립된 대리 손실.

DDPO-SF(`train_si.py`의 기본 경로)와의 관계: 로그확률 계산은 완전히 공유한다
(`ddpo.py`의 `sample_with_trace`/`step_logp`). 바뀌는 건 그 로그확률을 무엇에 곱하고
배치를 몇 번 쓰느냐다.

  DDPO-SF : loss = -c * mean(R_j * logp_new)                    배치 1회 사용
  DPPO    : loss = -mean(min(rho*A_j, clip(rho)*A_j)) + v*MSE   배치 여러 epoch 사용
            rho = exp(logp_new - logp_old)

왜 이 확장을 하는가(`phases/4-diffusion-si/step4.md`는 DDPO-SF를 고른 이유로 SI-EFM
Algorithm 1이 deadly triad의 두 꼭짓점을 배제한다는 점을 든다 — DPPO는 그 둘을 다시
들여온다): square task는 에피소드 하나가 실물 시뮬레이션 50초라 배치를 한 번 쓰고
버리는 비용이 지배적이다. 논문 충실성과 표본 효율의 맞교환이고, 그래서 알고리즘을
`algo` 설정으로 나란히 두고 같은 지표로 비교할 수 있게 한다.

DPPO 논문의 이중 MDP 중 어드밴티지는 **환경 MDP 쪽**에서만 만든다(결정=청크 단위 GAE).
한 결정의 디노이징 스텝들은 그 결정의 어드밴티지를 그대로 나눠 갖는다 — 디노이징
스텝 사이에는 환경 보상이 없으므로 그 안에서 시간차를 매길 근거가 없다.
"""

import numpy as np
import torch
import torch.nn as nn


class Critic(nn.Module):
    """V(o) — 정책이 이미 만든 global_cond 위의 MLP.

    비전 인코더를 새로 두지 않는 이유: 정책의 인코더는 얼려져 있고 매 결정마다
    global_cond가 이미 계산되므로, 그걸 재사용하면 크리틱이 공짜에 가깝다. 인코더를
    따로 학습시키면 보상 경로(얼려진 STG 예측기)와 다른 표현이 섞여 해석이 어려워진다.
    """

    def __init__(self, cond_dim, hidden=(256, 256)):
        super().__init__()
        layers, prev = [], cond_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, cond):
        return self.net(cond).squeeze(-1)


def gae(rewards, values, gamma, lam):
    """한 에피소드의 GAE(Schulman et al. 2016). 마지막 결정 뒤는 부트스트랩하지 않는다.

    에피소드가 max_steps로 잘렸든 성공으로 끝났든 여기서는 똑같이 V=0으로 끊는다 —
    잘린 에피소드를 부트스트랩하려면 "그 뒤에 무엇이 있었는가"를 크리틱이 알아야 하는데,
    학습 초반 크리틱은 그걸 모르고 그 오차가 어드밴티지에 그대로 실린다.

    Args:
        rewards: (T,) 결정별 보상.
        values: (T,) 결정별 V(o).
    Returns:
        (advantages, returns): 각각 (T,) float32.
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last = 0.0
    for t in reversed(range(T)):
        next_v = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * next_v - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + values


def clipped_surrogate(logp_new, logp_old, adv, clip_ratio):
    """PPO 클립 손실과 진단값(클립된 비율, 근사 KL).

    Returns:
        (loss, info): loss는 스칼라 텐서, info는 float dict.
    """
    ratio = torch.exp(logp_new - logp_old)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
    loss = -torch.min(unclipped, clipped).mean()
    with torch.no_grad():
        info = {
            "ratio_mean": float(ratio.mean()),
            "clip_frac": float(((ratio - 1.0).abs() > clip_ratio).float().mean()),
            # Schulman의 저분산 근사 KL — 음수가 나오면 뭔가 잘못된 것이므로 보기용으로 남긴다.
            "approx_kl": float(((ratio - 1.0) - (logp_new - logp_old)).mean()),
        }
    return loss, info


def denoising_step_filter(n_pairs_per_decision, ft_denoising_steps):
    """그래디언트를 받을 디노이징 단계 인덱스.

    DPPO 논문은 마지막 몇 단계만 파인튜닝해도 성능이 유지되고 계산이 크게 준다고 보고한다
    (`p`가 클수록 최종 샘플에 가깝다 — `_tile_pair_indices`의 인덱싱 규칙). None이면 전부.
    """
    if not ft_denoising_steps or ft_denoising_steps >= n_pairs_per_decision:
        return np.arange(n_pairs_per_decision)
    return np.arange(n_pairs_per_decision - ft_denoising_steps, n_pairs_per_decision)
