"""Quest 컨트롤러 제어 법칙 검증 — 가짜 reader로 돈다(헤드셋·SDK 불필요)."""

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from square_assembly.runners.quest_teleop import QuestTeleopController, undo_origin_yaw
from square_assembly.runners.square_oracle import _POS_SCALE, _ROT_SCALE

M = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)  # XR(x右,y上,-z前) -> 월드
_DOWN = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
_ID = [0, 0, 0, 1.0]


class FakeReader:
    def __init__(self):
        self.pose, self.grip, self.trigger = np.array([0, 0, 0, *_ID], float), 0.0, 0.0
        self.stick = np.zeros(2)

    def set(self, pos=None, quat=None, grip=None, trigger=None, stick=None):
        if stick is not None:
            self.stick = np.asarray(stick, float)
        if pos is not None:
            self.pose[:3] = pos
        if quat is not None:
            self.pose[3:] = quat
        if grip is not None:
            self.grip = grip
        if trigger is not None:
            self.trigger = trigger

    def __call__(self):
        return self.pose.copy(), self.grip, self.trigger, self.stick.copy()


def _state(pos=(0.0, 0.0, 1.0), R=_DOWN):
    return {"grip": np.array(pos, float), "R": np.array(R, float)}


def _make(**kw):
    r = FakeReader()
    kw.setdefault("euro_min_cutoff", 1e6)  # 필터를 사실상 끈다
    kw.setdefault("rot_source", "pose")  # 기존 테스트는 컨트롤러 회전 추종 기준
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
    r, c = _make()
    _engage(r, c)
    r.set(pos=[0.01, 0, 0])
    a = c.action(_state())
    np.testing.assert_allclose(c.target_xy, [0, 0.01], atol=1e-7)
    np.testing.assert_allclose(a[:3], [0, 0.01 / _POS_SCALE, 0], atol=1e-6)


def test_axes_forward_is_minus_world_x():
    r, c = _make()
    _engage(r, c)
    r.set(pos=[0, 0, -0.01])
    a = c.action(_state())
    assert a[0] < 0 and abs(a[1]) < 1e-6 and abs(a[2]) < 1e-6


def test_pos_scale_halves_motion():
    r, c = _make(pos_scale=0.5)
    _engage(r, c)
    r.set(pos=[0.01, 0, 0])
    c.action(_state())
    np.testing.assert_allclose(c.target_xy, [0, 0.005], atol=1e-7)


def test_rotation_follows_controller():
    r, c = _make()
    _engage(r, c)
    r.set(quat=Rot.from_rotvec([0, 0.1, 0]).as_quat())
    a = c.action(_state())
    np.testing.assert_allclose(a[3:6], M @ [0, 0.1, 0] / _ROT_SCALE, atol=1e-5)


def test_release_holds_actual_pose():
    r, c = _make()
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
    r, c = _make()
    c.take_over(_state())
    a = c.action(_state(pos=(0.0, 0.0, 0.99)))
    assert a[2] > 0
    np.testing.assert_allclose(c.target_p, [0, 0, 1.0], atol=1e-9)


def test_invalid_pose_while_held_requires_regrip():
    r, c = _make()
    _engage(r, c)
    r.set(pos=[0.01, 0, 0])
    c.action(_state())
    r.set(pos=[5, 5, 5], quat=[0, 0, 0, 0])
    c.action(_state())
    np.testing.assert_allclose(c.target_p, [0, 0, 1.0], atol=1e-9)  # 실제 EE에 고정(놓을 때와 같다)
    held = c.target_p.copy()
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
    r, c = _make()
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


def test_mouse_input_disabled_ignores_motion_and_wheel():
    from square_assembly.runners.mouse_teleop import MouseTeleopIntervention

    _, c = _make()
    interv = MouseTeleopIntervention(env=object(), controller=c, mouse_input=False, state_fn=_state)
    calls = []
    c.set_cursor = lambda *a, **k: calls.append("set_cursor")   # 가드가 없으면 여기로 들어온다
    c.wheel = lambda *a, **k: calls.append("wheel")
    interv._cur = np.zeros(2)   # _apply_target이 set_cursor까지 가도록
    xy = None if c.target_xy is None else np.array(c.target_xy, float)
    yaw = c.target_yaw
    interv._on_motion(100, 100)
    interv._on_wheel(3)
    assert calls == []
    assert (c.target_xy is None) if xy is None else np.testing.assert_array_equal(c.target_xy, xy) is None
    assert c.target_yaw == yaw
    assert not interv._magnet_on()
    assert "Quest" in interv._HELP[1]


def test_human_only_task_cfg_selects_can():
    from square_assembly.scripts.collect_square_scripted_intervention import load_human_only_task_cfg

    assert load_human_only_task_cfg("can").env_name == "PickPlaceCan"
    assert load_human_only_task_cfg("square").env_name == "NutAssemblySquare"


def test_pause_resume_requires_regrip():
    from square_assembly.runners.mouse_teleop import MouseTeleopIntervention

    r, c = _make()
    interv = MouseTeleopIntervention(env=object(), controller=c, mouse_input=False, state_fn=_state)
    interv._active = True
    _engage(r, c)
    interv._handle_key("s")       # 멈춤 — 그립은 계속 쥔 채
    r.set(pos=[0.0, 0.5, 0.0])    # 멈춘 동안 손이 멀리 움직인다
    interv._handle_key("s")       # 재개
    a = interv(0, {})
    np.testing.assert_allclose(a[:6], 0, atol=1e-6)
    r.set(grip=0.0)
    interv(1, {})
    r.set(grip=1.0)
    np.testing.assert_allclose(interv(2, {})[:6], 0, atol=1e-6)


def test_pause_resume_mouse_mode_does_not_resync():
    from square_assembly.runners.mouse_teleop import MouseTeleopIntervention

    _, c = _make()
    interv = MouseTeleopIntervention(env=object(), controller=c, mouse_input=True, state_fn=_state)
    interv._handle_key("s")
    interv._handle_key("s")
    assert not interv._resync


def test_grip_hysteresis():
    r, c = _make()
    _engage(r, c)
    for g in (0.85, 0.95, 0.85, 0.95):
        r.set(grip=g)
        c.action(_state())
        assert c.clutched
    r.set(grip=0.6)
    c.action(_state())
    assert not c.clutched


def test_floor_clamps_target_z():
    r, c = _make()
    _engage(r, c, _state(pos=(0, 0, 0.9)))
    r.set(pos=[0, -0.5, 0])   # 손을 크게 내린다(XR y 아래)
    c.action(_state(pos=(0, 0, 0.9)))
    assert c.target_p[2] >= c.z_min - 1e-12
    assert c.target_p[2] == c.z_min


def test_stick_mode_ignores_controller_pitch_and_levels_wrist():
    tilted = Rot.from_rotvec([0.2, 0, 0]).as_matrix() @ _DOWN
    r, c = _make(rot_source="stick")
    _engage(r, c, _state(R=tilted))
    r.set(quat=Rot.from_rotvec([0.5, 0, 0]).as_quat())  # XR x축(피치) — 야우가 아니므로 무시
    a = c.action(_state(R=tilted))
    assert a[3] < -1e-3 and abs(a[4]) < 1e-6 and abs(a[5]) < 1e-6  # 기운 x축만 되돌린다


def test_stick_mode_position_still_follows():
    r, c = _make(rot_source="stick")
    _engage(r, c)
    r.set(pos=[0.01, 0, 0])
    a = c.action(_state())
    np.testing.assert_allclose(a[:3], [0, 0.01 / _POS_SCALE, 0], atol=1e-6)


def test_stick_mode_yaw_follows_controller_yaw():
    r, c = _make(rot_source="stick")
    _engage(r, c)
    y0 = c.target_yaw
    r.set(quat=Rot.from_rotvec([0, 0.1, 0]).as_quat())  # XR y(위)축 = 월드 z축 +0.1rad
    a = c.action(_state())
    np.testing.assert_allclose(c.target_yaw, y0 + 0.1, atol=1e-6)
    assert a[5] > 0 and abs(a[3]) < 1e-6 and abs(a[4]) < 1e-6


def test_stick_tilts_like_a_helicopter_and_returns_level():
    r, c = _make(rot_source="stick", max_tilt=np.deg2rad(30))
    _engage(r, c)
    r.set(stick=[0, 1.0])  # 앞 — 위가 앞으로, 끝은 사람 쪽(+x)
    c.action(_state())
    np.testing.assert_allclose(c.target_R.as_matrix()[:, 2], [0.5, 0, -np.cos(np.deg2rad(30))], atol=1e-6)
    r.set(stick=[1.0, 0])  # 오른쪽 — 끝은 왼쪽(−y)
    c.action(_state())
    np.testing.assert_allclose(c.target_R.as_matrix()[:, 2], [0, -0.5, -np.cos(np.deg2rad(30))], atol=1e-6)
    r.set(stick=[0.1, 0])  # deadzone → 수직
    c.action(_state())
    np.testing.assert_allclose(c.target_R.as_matrix()[:, 2], [0, 0, -1], atol=1e-6)


def test_undo_origin_yaw_cancels_app_turn_about_headset():
    head = np.array([0.03, -0.5, 0.69, *Rot.from_rotvec([0, 0.3, 0]).as_quat()])
    ctrl = np.array([0.2, -0.56, 0.4, *Rot.from_rotvec([0.1, -0.2, 0.05]).as_quat()])
    turn = Rot.from_rotvec([0, 1.2, 0])  # 앱이 기준점을 헤드셋 둘레로 1.2rad 돌림
    h2, c2 = head.copy(), ctrl.copy()
    h2[3:] = (turn * Rot.from_quat(head[3:])).as_quat()
    c2[:3] = turn.apply(ctrl[:3] - head[:3]) + head[:3]
    c2[3:] = (turn * Rot.from_quat(ctrl[3:])).as_quat()
    out = undo_origin_yaw(c2, h2, yaw0=0.3)
    np.testing.assert_allclose(out[:3], ctrl[:3], atol=1e-9)
    assert (Rot.from_quat(out[3:]) * Rot.from_quat(ctrl[3:]).inv()).magnitude() < 1e-9
