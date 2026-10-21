"""DDIM 추론 전환 — η=0은 같은 시작 노이즈에서 결정적, η=1은 단계마다 노이즈가 들어가 DDPO가 확률을 잴 수 있다."""

import pytest
import torch

pytest.importorskip("diffusers")
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from square_assembly.runners.rollout import maybe_speed_up_inference


class _Policy:
    """predict 루프만 흉내 — 실제 정책(diffusion_policy_image.py)과 같은 step 호출."""

    def __init__(self):
        self.train_scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2")
        self.inference_scheduler = self.train_scheduler
        self.num_inference_steps = 100

    def sample(self, seed):
        torch.manual_seed(seed)
        x = torch.randn(1, 16, 7)
        self.inference_scheduler.set_timesteps(self.num_inference_steps)
        for t in self.inference_scheduler.timesteps:
            x = self.inference_scheduler.step(0.1 * x, t, x, **getattr(self, "inference_step_kwargs", {})).prev_sample
        return x


def test_ddim_eta0_is_deterministic_and_eta1_is_stochastic():
    det, sto = _Policy(), _Policy()
    maybe_speed_up_inference(det, 10, eta=0.0)
    maybe_speed_up_inference(sto, 10, eta=1.0)
    assert det.num_inference_steps == sto.num_inference_steps == 10
    torch.manual_seed(0); a = det.sample(0)
    b = det.sample(0)
    assert torch.equal(a, b)
    assert not torch.equal(sto.sample(0), a)                 # 같은 시작 노이즈라도 η=1은 단계마다 노이즈를 더한다
    assert sto.inference_step_kwargs == {"eta": 1.0}


def test_no_steps_leaves_the_policy_untouched():
    p = _Policy()
    maybe_speed_up_inference(p, None, eta=1.0)
    assert p.num_inference_steps == 100 and not hasattr(p, "inference_step_kwargs")


def test_sampler_label_names_what_the_collector_ran():
    from square_assembly.scripts.collect_square_scripted_intervention import sampler_label
    p = _Policy()
    assert sampler_label(p) == "DDPM100"
    maybe_speed_up_inference(p, 10, eta=1.0)
    assert sampler_label(p) == "DDIM10 eta=1"
