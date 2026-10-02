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


def _run(env, predict_fn=lambda history: np.zeros((4, 7)), **kwargs):
    return collect_episode(
        env, policy=None, normalizer=None, obs_keys=["x"], obs_horizon=2, action_horizon=4,
        device=None, intervention_fn=lambda step, obs: None,
        predict_fn=predict_fn, print_diagnostics=False, **kwargs,
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
            env.reset_to(2)
            return 2
        return None

    result = _run(env, max_steps=8, pre_step_fn=pre_step,
                  render_fn=lambda obs: rendered.append(obs["x"][0]) or True)
    assert [o["x"][0] for o in result["obs"]] == list(range(8))
    assert len(result["actions"]) == len(result["action_modes"]) == 8
    assert calls[6:8] == [(6, 6.0), (2, 2.0)]  # 되감은 직후 같은 스텝 번호로 다시 불린다
    assert rendered[5:7] == [6, 2]             # 되감은 장면을 한 번 다시 그린다(일시정지 유지용)


def test_rewind_resumes_the_chunk_that_was_running_then():
    """6 -> 2로 되감으면 2·3은 원래 청크 0의 남은 액션, 4부터 새 추론(원래 obs 히스토리로)."""
    env, seen, done = _CounterEnv(), [], []

    def predict(history):
        seen.append([o["x"][0] for o in history])
        return np.full((4, 7), float(len(seen) - 1))

    def pre_step(step, obs):
        if step == 6 and not done:
            done.append(step)
            env.reset_to(2)
            return 2
        return None

    result = _run(env, predict_fn=predict, max_steps=8, pre_step_fn=pre_step)
    assert list(result["actions"][:, 0]) == [0, 0, 0, 0, 2, 2, 2, 2]  # 청크 1(4~5)은 되감기로 버려짐
    assert seen == [[0, 0], [3, 4], [3, 4]]  # 되감은 뒤 첫 추론도 원래와 같은 히스토리


def test_rewind_stops_when_the_redraw_says_stop():
    env, rewound = _CounterEnv(), []

    def pre_step(step, obs):
        if step == 3:
            rewound.append(step)
            env.reset_to(1)
            return 1
        return None

    result = _run(env, max_steps=10, pre_step_fn=pre_step, render_fn=lambda obs: not rewound)
    assert len(result["actions"]) == 1  # 되감은 뒤 다시 그리다 창이 닫히면(q) 거기서 끝


class _HardResetEnv:
    """robosuite hard_reset처럼 reset마다 sim 객체를 새로 만든다 — 수집기가 옛 sim을 붙잡으면
    기록되는 상태가 전부 그 옛 sim의 멈춘 값이 된다(2026-09-22 r0v2 수집에서 실제로 났던 사고)."""

    class _Sim:
        def __init__(self, t):
            self.t = t

        def get_state(self):
            return np.array([float(self.t)])

    def __init__(self):
        self.env = type("Raw", (), {})()
        self.env.sim = self._Sim(-99)

    def _obs(self):
        return {"x": np.array([float(self.env.sim.t)])}

    def reset(self):
        self.env.sim = self._Sim(0)
        return self._obs()

    def reset_to(self, state):
        self.env.sim.t = int(state["states"][0])
        return self._obs()

    def step(self, action):
        self.env.sim.t += 1
        return self._obs(), 0.0, False, {}

    def is_success(self):
        return {"task": False}


class _RewindAt:
    def __init__(self, at, to):
        self.at, self.to = at, to

    def pop_rewind(self, step):
        if step == self.at:
            self.at = None
            return self.to
        return None


def test_recorder_follows_the_sim_across_hard_resets_and_rewinds_to_it():
    from square_assembly.scripts.collect_square_scripted_intervention import make_recorder

    env = _HardResetEnv()
    pre_step, obs_ep, states_ep = make_recorder(env, _RewindAt(at=6, to=2), ["x"], [])
    result = _run(env, max_steps=8, pre_step_fn=pre_step)
    assert [s[0] for s in states_ep] == list(range(8))  # 멈춘 옛 sim(-99)이 아니라 지금 sim
    assert [o["x"][0] for o in obs_ep] == list(range(8))
    assert len(result["actions"]) == 8


def test_resume_continues_a_stored_prefix_and_can_rewind_into_it():
    """저장된 에피소드의 앞 3스텝을 이어받아 step 3에서 시작하고, 그 앞부분의 step 1로도 되감을 수 있다."""
    env, seen, histories = _CounterEnv(), [], []
    rewinds = iter([None, None, 1])                      # 두 스텝 진행한 뒤(step 5 직전) step 1로

    def pre_step(step, obs):
        seen.append((step, obs["x"][0]))
        t = next(rewinds, None)
        if t is not None:
            env.reset_to(t)
        return t

    def predict(history):
        histories.append([o["x"][0] for o in history])
        return np.full((4, 7), 5.0)

    result = _run(env, predict_fn=predict, max_steps=6, reset_fn=lambda: env.reset_to(3), pre_step_fn=pre_step,
                  resume={"obs": [{"x": np.array([float(t)])} for t in range(3)],
                          "actions": np.ones((3, 7)), "modes": [0, 0, 0]})

    assert seen == [(3, 3), (4, 4), (5, 5), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]
    assert histories == [[2, 3], [0, 1], [4, 5]]          # 히스토리도 저장된 앞부분에서 이어진다
    assert result["actions"][:, 0].tolist() == [1, 5, 5, 5, 5, 5]   # step 0만 저장된 액션, 되감은 뒤는 새 액션
    assert [o["x"][0] for o in result["obs"]] == [0, 1, 2, 3, 4, 5]


def test_rewind_also_restores_the_gripper_command_ramp():
    """robosuite 그리퍼는 명령을 내부에서 적분한다(current_action) — sim 상태에 없어서 따로 되돌려야 한다."""
    from square_assembly.scripts.collect_square_scripted_intervention import make_recorder

    env = _HardResetEnv()
    grip = type("Grip", (), {"current_action": np.zeros(2)})()
    env.env.robots = [type("Robot", (), {"arms": ["right"], "gripper": {"right": grip}})()]
    real_step = env.step

    def step(action):
        grip.current_action = grip.current_action + 1.0      # 스텝마다 명령이 한 칸씩 쌓인다
        return real_step(action)

    env.step = step

    pre_step, obs_ep, states_ep = make_recorder(env, _RewindAt(at=6, to=2), ["x"], [])
    _run(env, max_steps=8, pre_step_fn=pre_step)
    assert grip.current_action[0] == 8.0      # step 2로 되감으며 그때 값(2)으로 돌아가 2 -> 8. 안 되돌리면 6 -> 12


def test_gripper_ramp_follows_the_binary_command_at_the_gripper_speed():
    from square_assembly.scripts.collect_square_scripted_intervention import gripper_ramp
    ramp = gripper_ramp([1, 1, 1, 1, 1, 1, -1], speed=0.2)
    assert len(ramp) == 8                                     # ramp[t] = 액션 t를 넣기 직전 값
    np.testing.assert_allclose([r[1] for r in ramp], [0, .2, .4, .6, .8, 1.0, 1.0, .8], atol=1e-9)
    np.testing.assert_allclose([r[0] for r in ramp], [0, -.2, -.4, -.6, -.8, -1.0, -1.0, -.8], atol=1e-9)
