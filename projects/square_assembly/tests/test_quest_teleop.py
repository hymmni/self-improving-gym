"""Quest 컨트롤러 제어 법칙 검증 — 가짜 reader로 돈다(헤드셋·SDK 불필요)."""

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from square_assembly.runners.quest_teleop import QuestTeleopController
from square_assembly.runners.square_oracle import _POS_SCALE, _ROT_SCALE

M = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)  # XR(x右,y上,-z前) -> 월드
_DOWN = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
_ID = [0, 0, 0, 1.0]


class FakeReader:
    def __init__(self):
        self.pose, self.grip, self.trigger = np.array([0, 0, 0, *_ID], float), 0.0, 0.0

    def set(self, pos=None, quat=None, grip=None, trigger=None):
        if pos is not None:
            self.pose[:3] = pos
        if quat is not None:
            self.pose[3:] = quat
        if grip is not None:
            self.grip = grip
        if trigger is not None:
            self.trigger = trigger

    def __call__(self):
        return self.pose.copy(), self.grip, self.trigger


def _state(pos=(0.0, 0.0, 1.0), R=_DOWN):
    return {"grip": np.array(pos, float), "R": np.array(R, float)}


def _make(**kw):
    r = FakeReader()
    kw.setdefault("euro_min_cutoff", 1e6)  # 필터를 사실상 끈다
    return r, QuestTeleopController(reader=r, **kw)


def _engage(r, c, state=None):
    state = state or _state()
    c.take_over(state)
    r.set(grip=0.0)
    c.action(state)  # 그립 뗀 상태를 한 번 봐서 무장
    r.set(grip=1.0)
    return c.action(state)


def test_clutch_engage_does_not_jump():
    r, c = _make()
    r.set(pos=[0.3, -0.2, 0.5])
    a = _engage(r, c)
    np.testing.assert_allclose(a[:6], 0, atol=1e-6)


def test_one_to_one_position():
    r, c = _make(kp=1.0, kd=0.0)
    _engage(r, c)
    r.set(pos=[0.01, 0, 0])
    a = c.action(_state())
    np.testing.assert_allclose(c.target_xy, [0, 0.01], atol=1e-7)
    np.testing.assert_allclose(a[:3], [0, 0.01 / _POS_SCALE, 0], atol=1e-6)


def test_axes_forward_is_minus_world_x():
    r, c = _make(kp=1.0, kd=0.0)
    _engage(r, c)
    r.set(pos=[0, 0, -0.01])
    a = c.action(_state())
    assert a[0] < 0 and abs(a[1]) < 1e-6 and abs(a[2]) < 1e-6


def test_pos_scale_halves_motion():
    r, c = _make(kp=1.0, kd=0.0, pos_scale=0.5)
    _engage(r, c)
    r.set(pos=[0.01, 0, 0])
    c.action(_state())
    np.testing.assert_allclose(c.target_xy, [0, 0.005], atol=1e-7)


def test_rotation_follows_controller():
    r, c = _make(kp=5.0)
    _engage(r, c)
    r.set(quat=Rot.from_rotvec([0, 0.1, 0]).as_quat())
    a = c.action(_state())
    np.testing.assert_allclose(a[3:6], M @ [0, 0.1, 0] / _ROT_SCALE, atol=1e-5)


def test_release_holds_actual_pose():
    r, c = _make(kp=1.0, kd=0.0)
    _engage(r, c)
    r.set(pos=[0.05, 0, 0])
    c.action(_state())
    r.set(grip=0.0)
    c.action(_state(pos=(0.0, 0.02, 1.0)))  # 놓는 스텝: EE는 0.02m만 감
    np.testing.assert_allclose(c.target_p, [0.0, 0.02, 1.0], atol=1e-9)
    a = c.action(_state(pos=(0.0, 0.02, 0.99)))  # 처짐: 목표는 고정이라 밀어 올려야 한다
    np.testing.assert_allclose(c.target_p, [0.0, 0.02, 1.0], atol=1e-9)
    assert a[2] > 0 and abs(a[0]) < 1e-6 and abs(a[1]) < 1e-6


def test_take_over_holds_pose():
    r, c = _make(kp=1.0, kd=0.0)
    c.take_over(_state())
    a = c.action(_state(pos=(0.0, 0.0, 0.99)))
    assert a[2] > 0
    np.testing.assert_allclose(c.target_p, [0, 0, 1.0], atol=1e-9)


def test_invalid_pose_while_held_requires_regrip():
    r, c = _make(kp=1.0, kd=0.0)
    _engage(r, c)
    r.set(pos=[0.01, 0, 0])
    c.action(_state())
    r.set(pos=[5, 5, 5], quat=[0, 0, 0, 0])
    c.action(_state())
    held = c.target_p.copy()  # 직전 목표 유지
    r.set(quat=_ID)  # 포즈 복귀 — 그립은 그대로 눌림: 재개해도 목표가 튀지 않는다
    c.action(_state())
    np.testing.assert_array_equal(c.target_p, held)


def test_action_is_clipped():
    r, c = _make()
    _engage(r, c)
    r.set(pos=[0.5, 0, 0], quat=Rot.from_rotvec([0, 1.5, 0]).as_quat())
    a = c.action(_state())
    assert np.all(np.abs(a[:3]) <= 1.0 + 1e-6) and np.all(np.abs(a[3:6]) <= 0.4 + 1e-6)
    assert a[1] == 1.0


def test_trigger_binary_and_independent_of_grip():
    r, c = _make()
    r.set(grip=0.0, trigger=0.6)
    assert c.action(_state())[6] == 1.0
    r.set(trigger=0.4)
    assert c.action(_state())[6] == -1.0


def test_take_over_requires_regrip():
    r, c = _make(kp=1.0, kd=0.0)
    r.set(grip=0.95)
    c.take_over(_state())
    r.set(pos=[0.02, 0, 0])
    np.testing.assert_array_equal(c.action(_state())[:6], 0)
    r.set(pos=[0.04, 0, 0])
    np.testing.assert_array_equal(c.action(_state())[:6], 0)
    r.set(grip=0.0)
    c.action(_state())
    r.set(grip=0.95)
    c.action(_state())  # 새로 잡음
    r.set(pos=[0.05, 0, 0])
    assert c.action(_state())[1] > 0


def test_invalid_pose_holds():
    r, c = _make()
    _engage(r, c)
    r.set(pos=[0.1, 0, 0], quat=[0, 0, 0, 0])
    np.testing.assert_array_equal(c.action(_state())[:6], 0)


def test_rotation_near_pi_is_continuous():
    r, c = _make()
    _engage(r, c)
    signs = []
    for ang in np.arange(3.0, 3.2001, 0.05):
        r.set(quat=Rot.from_rotvec([0, 0, ang]).as_quat())
        signs.append(np.sign(c.action(_state())[3:6]))
    assert all(np.array_equal(signs[0], s) for s in signs)
    assert np.any(signs[0] != 0)


def test_set_cursor_and_wheel_are_noops():
    r, c = _make()
    c.take_over(_state())
    before = c.target_xy.copy()
    c.set_cursor((1.0, 1.0))
    assert c.wheel(3) is False
    np.testing.assert_array_equal(c.target_xy, before)
