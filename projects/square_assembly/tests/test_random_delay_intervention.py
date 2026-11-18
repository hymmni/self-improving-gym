"""랜덤 지연 개입(수집기 --random-delay): Tab → d ~ U{0..max_steps−t_press−1} 뒤 넘겨받기, 다시 Tab = 취소,
넘겨받은 뒤엔 돌려주지 않고 max_steps도 안 걸린다. 창(Tk)/robosuite 없이 가짜 env로 돈다."""

import json

import numpy as np
import pytest

from square_assembly.datasets.labels import LABEL_INTV, LABEL_PREINTV, LABEL_ROLLOUT
from square_assembly.runners import intervention_rollout
from square_assembly.runners.intervention_rollout import collect_episode
from square_assembly.runners.mouse_teleop import MouseTeleopIntervention

_DOWN = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
_OBS = {"robot0_gripper_qpos": np.array([0.04, -0.04])}


def _state():
    return {"grip": np.array([0.0, 0.0, 1.0]), "R": _DOWN.copy(), "nut": np.array([0.1, 0.0, 0.83]),
            "handle": np.array([0.154, 0.0, 0.83]), "peg": np.array([0.23, 0.1, 0.85])}


def _viewer(**kw):
    return MouseTeleopIntervention(env=object(), state_fn=_state, **kw)


@pytest.fixture(autouse=True)
def _raw_storage(monkeypatch):
    monkeypatch.setattr(intervention_rollout, "_to_storage_obs", lambda key, value: value)


class _Env:
    def __init__(self, success_at=None):
        self.t, self.success_at = 0, success_at

    def reset(self):
        self.t = 0
        return {"x": np.array([0.0]), **_OBS}

    def step(self, action):
        self.t += 1
        return {"x": np.array([float(self.t)]), **_OBS}, 0.0, False, {}

    def is_success(self):
        return {"task": self.success_at is not None and self.t >= self.success_at}


def _collect(viewer, env, max_steps, keys_at):
    """keys_at = {스텝: 키} — 그 스텝을 밟은 직후(render 시점) 키를 누른다."""
    def render(obs):
        k = keys_at.get(env.t)
        if k:
            viewer._handle_key(k)
        return True

    return collect_episode(
        env, policy=None, normalizer=None, obs_keys=["x"], obs_horizon=1, action_horizon=4, device=None,
        intervention_fn=viewer, max_steps=max_steps, should_end_fn=viewer.should_end, render_fn=render,
        predict_fn=lambda h: np.zeros((4, 7)), print_diagnostics=False, limit_policy_only=viewer.random_delay)


def test_without_random_delay_tab_takes_over_on_the_next_step():
    v = _viewer()
    assert v(0, _OBS) is None
    v._handle_key("Tab")
    assert v(1, _OBS) is not None and v.num_triggers == 1 and v.delay_events == []


def test_delay_is_uniform_over_the_remaining_steps():
    v = _viewer(random_delay=True, max_steps=700)
    v.delay_seed, v._rng = 0, np.random.default_rng(0)
    ds = []
    for _ in range(4000):
        v._next_step = 100
        v._handle_key("Tab")
        ds.append(v._delay_req["d"])
        v._handle_key("Tab")                      # 취소하고 다시
    ds = np.array(ds)
    assert ds.min() >= 0 and ds.max() <= 599      # t0 ≤ max_steps − 1
    assert abs(ds.mean() - 299.5) < 10 and ds.min() < 10 and ds.max() > 590
    assert np.histogram(ds, bins=6, range=(0, 600))[0].min() > 550   # 각 칸 기대 667
    v._next_step = 699                             # 마지막 스텝에 누르면 d = 0
    v._handle_key("Tab")
    assert v._delay_req["d"] == 0 and v._delay_req["t0"] == 699


def test_policy_keeps_driving_until_t0_then_human_keeps_control_past_max_steps():
    v = _viewer(random_delay=True, max_steps=50)
    v.reset()
    r = _collect(v, _Env(), 50, {10: "Tab", 45: "Tab", 80: "q"})   # 45의 Tab은 넘겨받은 뒤라 안 돌려준다
    rec = v.delay_record()
    t_press, t0 = rec["t_press"], rec["t0"]
    assert t_press == 10 and 0 <= rec["delay_d"] <= 39 and t0 == 10 + rec["delay_d"] and rec["max_delay"] == 39
    modes = r["action_modes"]
    assert len(modes) == 80                        # max_steps(50)를 넘어 사람이 q를 누를 때까지
    assert (modes[t0:] == LABEL_INTV).all() and not (modes[:t0] == LABEL_INTV).any()
    assert (modes[max(0, t0 - 15):t0] == LABEL_PREINTV).all()   # PREINTV도 t0 기준
    assert not rec["ended_during_delay"] and v.num_triggers == 1


def test_episode_that_ends_during_the_delay_is_saved_without_intervention():
    v = _viewer(random_delay=True, max_steps=700)
    v.reset()
    v.delay_seed, v._rng = 1, np.random.default_rng(1)   # 이 seed의 첫 d(=t0−5)가 30보다 크다
    r = _collect(v, _Env(success_at=30), 700, {5: "Tab"})
    rec = v.delay_record()
    assert r["success"] and len(r["action_modes"]) == 30 and not (r["action_modes"] == LABEL_INTV).any()
    assert rec["t_press"] == 5 and rec["t0"] == -1 and rec["ended_during_delay"] and rec["delay_d"] > 25
    assert json.loads(rec["delay_events"])[0]["status"] == "pending"


def test_tab_during_the_delay_cancels_the_request():
    v = _viewer(random_delay=True, max_steps=40)
    v.reset()
    r = _collect(v, _Env(), 40, {3: "Tab", 4: "Tab"})
    rec = v.delay_record()
    assert (r["action_modes"] == LABEL_ROLLOUT).all() and len(r["action_modes"]) == 40   # 정책이 끝까지, 시간 제한으로 끝
    assert rec["t_press"] == -1 and rec["t0"] == -1 and not rec["ended_during_delay"]
    assert [e["status"] for e in json.loads(rec["delay_events"])] == ["cancelled"]


def test_rewind_before_the_press_forgets_it_and_into_the_delay_marks_it_rewound():
    v = _viewer(random_delay=True, max_steps=700)
    v._next_step = 20
    v._handle_key("Tab")
    v._handle_key("b")                             # 20스텝 뒤로 → 0, 누르기 전
    assert v.pop_rewind(20) == 0 and v.delay_events == [] and v._delay_req is None
    v._next_step = 30
    v._handle_key("Tab")
    v._delay_req["t0"] = 100                       # t0를 고정해 지연 한가운데로 되감는다
    v.rewind_to(31)
    assert v.pop_rewind(60) == 31 and v._delay_req is None and v.delay_events[0]["status"] == "rewound"
    assert v(31, _OBS) is None                     # 정책이 계속 몬다


def test_no_handback_keeps_human_control_without_random_delay():
    v = _viewer(handback=False)
    v._handle_key("Tab")
    v._handle_key("Tab")
    assert v(1, _OBS) is not None and v.num_triggers == 1
