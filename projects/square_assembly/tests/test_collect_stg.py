"""수집기의 STG 실시간 예측(--stg-ckpt) — 다음 스텝에 정책이 볼 obs 창(저장된 앞 프레임 + 지금)을 예측기에 넣고,
그 분포를 창에 그 스텝 번호로 넘긴다. 창·robosuite·실제 예측기 없이 돈다."""

import numpy as np
import torch

from square_assembly.scripts.collect_square_scripted_intervention import make_noise_tracker, make_stg_fn

_CAM = "agentview_image"


class _Reward:
    obs_horizon, obs_keys, rgb_keys = 2, [_CAM, "robot0_eef_pos"], [_CAM]

    def __init__(self):
        self.seen = []

    def predictor(self, batch):
        self.seen.append({k: v.cpu().numpy() for k, v in batch.items()})
        return torch.log(torch.tensor([[0.25, 0.75]]))


class _Norm:
    def normalize_obs(self, d):
        return {k: v * 10 for k, v in d.items()}


def _raw(v):
    """env가 주는 obs — rgb는 CHW float[0,1]."""
    return {_CAM: np.full((3, 4, 4), v, np.float32), "robot0_eef_pos": np.full(3, v, np.float32)}


def _stored(v):
    """수집기가 쌓는 obs — rgb는 HWC uint8."""
    return {_CAM: np.full((4, 4, 3), round(v * 255), np.uint8), "robot0_eef_pos": np.full(3, v, np.float32)}


def test_stg_sees_the_window_the_policy_will_see():
    r = _Reward()
    fn = make_stg_fn(r, _Norm(), "cpu")
    np.testing.assert_allclose(fn([], _raw(0.2)), [0.25, 0.75], atol=1e-6)
    first = r.seen[-1][_CAM]
    assert first.shape == (1, 2, 3, 4, 4)                  # 앞 프레임이 없으면 지금 프레임으로 채운다
    np.testing.assert_allclose(first[0, 0], first[0, 1])
    fn([_stored(0.0), _stored(0.4)], _raw(0.8))
    b = r.seen[-1]
    np.testing.assert_allclose(b[_CAM][0, :, 0, 0, 0], [0.4, 0.8], atol=1e-2)       # 저장된 마지막 + 지금
    np.testing.assert_allclose(b["robot0_eef_pos"][0, :, 0], [4.0, 8.0], atol=1e-5)  # lowdim은 정규화해서


def test_tracker_hands_the_prediction_to_the_window_at_the_next_step():
    calls = []

    class _Interv:
        def set_stg(self, step, probs):
            calls.append((step, probs))

        def render(self, frames):
            return True

    class _Env:
        def render(self, **kw):
            return np.zeros((8, 8, 3), np.uint8)

    obs_ep = [_stored(0.1)] * 3
    track, _ = make_noise_tracker(_Env(), _Interv(), _CAM, ["agentview"], 8, obs_ep, lambda: None,
                                  stg=lambda ep, raw: ("probs", len(ep)))
    assert track(_raw(0.5))
    assert calls == [(3, ("probs", 3))]                    # 이 obs가 저장될 자리 = 다음 스텝
