"""collect_episode의 스텝 정렬과 되감기(pre_step_fn) 검증 — 가짜 env로, 렌더/정책 없이.

가짜 env의 상태는 정수 하나(지금까지 밟은 스텝 수)이고 obs도 그 값이라, "obs[i]가 액션 i를
넣기 직전 관측인가"와 "되감은 뒤 스텝 번호·버퍼가 t부터 다시 이어지는가"를 값으로 바로 본다.
"""

import numpy as np
import pytest

from square_assembly.runners import intervention_rollout
from square_assembly.runners.intervention_rollout import collect_episode


class _CounterEnv:
    def __init__(self):
        self.t = 0

    def _obs(self):
        return {"x": np.array([float(self.t)])}

    def reset(self):
        self.t = 0
        return self._obs()

    def reset_to(self, t):
        self.t = t
        return self._obs()

    def step(self, action):
        self.t += 1
        return self._obs(), 0.0, False, {}

    def is_success(self):
        return {"task": False}


def _run(env, **kwargs):
    return collect_episode(
        env, policy=None, normalizer=None, obs_keys=["x"], obs_horizon=2, action_horizon=4,
        device=None, intervention_fn=lambda step, obs: None,
        predict_fn=lambda history: np.zeros((4, 7)), print_diagnostics=False, **kwargs,
    )


@pytest.fixture(autouse=True)
def _raw_storage(monkeypatch):
    # ObsUtils는 전역 초기화가 필요해서, 저장 변환은 값 그대로 통과시킨다.
    monkeypatch.setattr(intervention_rollout, "_to_storage_obs", lambda key, value: value)


def test_obs_is_the_one_before_its_action():
    rendered = []
    result = _run(_CounterEnv(), max_steps=5, render_fn=lambda obs: rendered.append(obs["x"][0]) or True)
    assert [o["x"][0] for o in result["obs"]] == [0, 1, 2, 3, 4]  # obs[i] = 액션 i 직전
    assert rendered == [1, 2, 3, 4, 5]                           # render_fn은 스텝 뒤 — 저장용이 아니다
    assert len(result["actions"]) == 5


def test_rewind_truncates_and_continues_from_t():
    env, calls, rendered = _CounterEnv(), [], []

    def pre_step(step, obs):
        calls.append((step, obs["x"][0]))
        if step == 6 and len(calls) == 7:  # 한 번만 6 -> 2로
            return 2, env.reset_to(2)
        return None

    result = _run(env, max_steps=8, pre_step_fn=pre_step,
                  render_fn=lambda obs: rendered.append(obs["x"][0]) or True)
    assert [o["x"][0] for o in result["obs"]] == list(range(8))
    assert len(result["actions"]) == len(result["action_modes"]) == 8
    assert calls[6:8] == [(6, 6.0), (2, 2.0)]  # 되감은 직후 같은 스텝 번호로 다시 불린다
    assert rendered[5:7] == [6, 2]             # 되감은 장면을 한 번 다시 그린다(일시정지 유지용)


def test_rewind_stops_when_the_redraw_says_stop():
    env, rewound = _CounterEnv(), []

    def pre_step(step, obs):
        if step == 3:
            rewound.append(step)
            return 1, env.reset_to(1)
        return None

    result = _run(env, max_steps=10, pre_step_fn=pre_step, render_fn=lambda obs: not rewound)
    assert len(result["actions"]) == 1  # 되감은 뒤 다시 그리다 창이 닫히면(q) 거기서 끝
