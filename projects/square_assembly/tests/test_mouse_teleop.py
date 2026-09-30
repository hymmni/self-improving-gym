"""마우스 텔레옵 제어 법칙 + 키 상태 머신 검증. 창(Tk)/robosuite 없이 돈다.

실제 조작감(게인·속도)은 서버에서 창을 띄워 손으로 맞추는 것이라 여기서는 "입력이 액션의
어느 성분을 어느 방향으로 움직이는가"와 상한/클램프/초기화만 본다.
"""

import numpy as np
import pytest

from square_assembly.runners.mouse_teleop import (
    MouseTeleopController, MouseTeleopIntervention, TopDownMap,
)
from square_assembly.runners.square_oracle import _POS_SCALE

# 수직 아래를 보는 그리퍼, 손가락 축 = 월드 +x
_DOWN = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])


def _state(x=0.0, y=0.0, z=1.0, R=_DOWN):
    return {"grip": np.array([x, y, z]), "R": R.copy(),
            "nut": np.array([0.1, 0.0, 0.83]), "handle": np.array([0.154, 0.0, 0.83]),
            "peg": np.array([0.23, 0.1, 0.85])}


def test_xy_is_proportional_and_capped():
    c = MouseTeleopController(kp=1.0, kd=0.0, pos_cap=0.3)
    c.take_over(_state())
    c.set_cursor((0.005, 0.0))
    a = c.action(_state())
    np.testing.assert_allclose(a[:2], [0.005 / _POS_SCALE, 0.0])
    c.set_cursor((-0.2, 0.05))
    a = c.action(_state())
    np.testing.assert_allclose(a[:2], [-0.3, 0.3])  # 상한에 걸림


def test_kd_damps_when_already_moving_toward_target():
    p = MouseTeleopController(kp=1.0, kd=0.0)
    d = MouseTeleopController(kp=1.0, kd=0.5)
    for c in (p, d):
        c.take_over(_state(x=0.0))
        c.set_cursor((0.01, 0.0))
        c.action(_state(x=0.0))
    a_p = p.action(_state(x=0.004))  # 한 스텝에 4mm 전진한 상태
    a_d = d.action(_state(x=0.004))
    assert 0 < a_d[0] < a_p[0]


def test_take_over_holds_position_until_cursor_moves():
    c = MouseTeleopController()
    c.set_cursor((0.3, 0.3))  # 잡기 전에 쌓인 커서 위치는 무시돼야 한다
    c.take_over(_state())
    np.testing.assert_array_equal(c.action(_state())[:2], [0.0, 0.0])


def test_z_keys_and_limits():
    c = MouseTeleopController(z_speed=0.2, z_min=0.83, z_max=1.10)
    c.take_over(_state())
    c.z_up = True
    assert c.action(_state(z=1.0))[2] == pytest.approx(0.2)
    assert c.action(_state(z=1.10))[2] == 0.0  # 상한에서 멈춤
    c.z_up, c.z_down = False, True
    assert c.action(_state(z=1.0))[2] == pytest.approx(-0.2)
    assert c.action(_state(z=0.83))[2] == 0.0
    c.z_up = True  # 둘 다 누르면 정지
    assert c.action(_state(z=1.0))[2] == 0.0


def test_wheel_turns_yaw_target_about_world_z():
    c = MouseTeleopController(yaw_step=np.deg2rad(5.0), rot_cap=0.4)
    c.take_over(_state())
    np.testing.assert_array_equal(c.action(_state())[3:6], [0.0, 0.0, 0.0])  # 목표 = 현재
    c.wheel(+1)
    a = c.action(_state())
    assert a[5] == pytest.approx(np.deg2rad(5.0) / 0.5)  # z축 회전, _ROT_SCALE=0.5
    assert abs(a[3]) < 1e-9 and abs(a[4]) < 1e-9
    c.wheel(-3)
    assert c.action(_state())[5] == pytest.approx(-np.deg2rad(10.0) / 0.5)


def test_gripper_follows_the_button_immediately():
    """robosuite가 부호만 보고 스스로 램프한다 — 여기서 램프하면 부호가 바뀔 때까지 0.5초 늦기만 한다."""
    c = MouseTeleopController()
    c.take_over(_state())
    assert c.grip_cmd == -1.0
    c.grip_pressed = True
    assert c.action(_state())[6] == 1.0
    c.grip_pressed = False
    assert c.action(_state())[6] == -1.0


def test_take_over_follows_the_button_not_the_fingers():
    """넘겨받는 순간에도 그리퍼는 버튼만 따른다 — 빈손으로 닫힌 채(헛집기) 넘겨받으면 클릭 없이 바로 연다.
    쥔 너트를 유지하려면 버튼을 누른 채 Tab을 누른다(2026-09-30 사용자 결정)."""
    c = MouseTeleopController()
    c.take_over(_state())
    assert c.grip_cmd == -1.0 and c.action(_state())[6] == -1.0
    c.grip_pressed = True  # 누른 채 Tab
    c.take_over(_state())
    assert c.grip_cmd == 1.0
    for _ in range(30):
        assert c.action(_state())[6] == 1.0
    c.grip_pressed = False
    assert c.action(_state())[6] == -1.0


def test_map_roundtrip_and_orientation():
    m = TopDownMap((0.0, 0.0), 0.8, 480)
    assert m.to_px((0.0, 0.0)) == (240, 240)
    px, py = m.to_px((0.1, 0.2))
    assert px > 240 and py > 240  # +y는 오른쪽, +x는 아래(로봇 쪽이 위)
    np.testing.assert_allclose(m.to_world(*m.to_px((0.1, 0.2))), [0.1, 0.2], atol=1e-3)


def _make():
    interv = MouseTeleopIntervention(env=object(), state_fn=_state)
    return interv


def test_keys_drive_the_state_machine():
    interv = _make()
    obs = {"robot0_gripper_qpos": np.array([0.04, -0.04])}
    assert interv(0, obs) is None
    interv._handle_key("Tab")
    assert interv.num_triggers == 1
    a = interv(1, obs)
    assert a.shape == (7,)
    interv._handle_key("Tab")  # 한 번 더 누르면 정책에 돌려준다
    assert interv(2, obs) is None
    interv._handle_key("Tab")
    assert interv.num_triggers == 2
    assert not interv._paused
    interv._handle_key("s")
    assert interv._paused
    interv._handle_key("s")
    assert not interv._paused
    assert not interv.should_end()
    interv._handle_key("q")
    assert interv.should_end()


def test_reset_clears_everything():
    interv = _make()
    interv._handle_key("Tab")
    interv._handle_key("s")
    interv._handle_key("q")
    interv.reset()
    assert interv(0, {}) is None and not interv._paused and not interv.should_end()
    assert interv.num_triggers == 0


def test_space_and_shift_move_z_for_as_long_as_they_are_held():
    """예전(pynput + 0.6초 유지)엔 꾹 눌러도 처음 0.6초만 움직였다 — 이제 누름~뗌 사이 내내 움직인다."""
    interv = _make()
    interv._on_key("Tab", True)
    interv._on_key("space", True)
    assert all(interv(t, {})[2] > 0 for t in range(40))    # 2초(40스텝) 누르고 있어도 계속
    interv._on_key("space", False)
    interv._on_key("space", True)                         # X11 키 반복 = 뗌+누름 쌍 — 끊기지 않는다
    assert interv(40, {})[2] > 0
    interv._on_key("space", False)
    assert interv(41, {})[2] == 0.0
    interv._on_key("Shift_L", True)
    assert all(interv(t, {})[2] < 0 for t in range(42, 82))  # 수식키는 반복이 없어도 계속
    interv._on_key("B", True)                             # Shift를 누른 채 b(대문자로 온다)도 먹는다
    assert interv.pop_rewind(82) == 62
    interv._on_key("Shift_L", False)
    assert interv(62, {})[2] == 0.0


def test_ctrl_lifts_because_rustdesk_sends_space_as_taps():
    """RustDesk는 꾹 누른 Space를 누름+뗌 쌍으로만 보낸다 — Ctrl은 누름·뗌이 제대로 와서 누르는 동안 올라간다."""
    interv = _make()
    interv._on_key("Tab", True)
    interv._on_key("Control_L", True)
    assert all(interv(t, {})[2] > 0 for t in range(20))
    interv._on_key("Control_L", False)
    assert interv(20, {})[2] == 0.0
    interv._on_key("space", True); interv._on_key("space", False)  # 같은 update()에서 처리되는 탭 쌍
    assert interv(21, {})[2] == 0.0


def test_shift_tab_still_toggles_and_yaw_is_wheel_only():
    interv = _make()
    interv._on_key("Tab", True)
    interv._on_key("a", True); interv._on_key("d", True)  # a/d는 없앴다(2026-09-22) — 휠로만 돈다
    assert interv(0, {})[5] == 0.0
    interv.controller.wheel(+1)
    assert interv(1, {})[5] > 0            # 휠 위 한 칸: +5도 목표 -> z축 양의 회전
    assert interv._on_key("ISO_Left_Tab", True) == "break"  # Shift+Tab도 전환, Tk 포커스 이동은 막는다
    assert interv(2, {}) is None


def test_back_accumulates_pauses_and_clamps_at_zero():
    interv = MouseTeleopIntervention(env=object(), state_fn=_state, back_steps=40)
    assert interv.pop_rewind(100) is None
    interv._handle_key("b")
    interv._handle_key("b")
    assert interv._paused                  # 되감으면 멈춰서 어디로 왔는지 보여준다
    assert interv.pop_rewind(100) == 20    # 두 번 = 80스텝 뒤로
    assert interv._rewound == (100, 20)    # 멈춘 화면에 "어디서 어디로" 띄울 정보
    interv._handle_key("s")
    assert interv._rewound is None         # 재개하면 지운다
    assert interv.pop_rewind(100) is None  # 꺼내면 비워진다
    interv._handle_key("b")
    assert interv.pop_rewind(15) == 0      # 시작보다 앞으로는 못 간다


def test_restart_goes_to_zero_and_beats_back():
    interv = _make()
    interv._handle_key("b")
    interv._handle_key("r")
    interv._handle_key("b")
    assert interv.pop_rewind(300) == 0


def test_rewind_resyncs_the_controller_to_the_restored_pose():
    """되감기 전 목표(커서·야우·그리퍼)를 그대로 들고 가면 재개하자마자 팔이 튄다."""
    interv = _make()
    interv._handle_key("Tab")
    interv.controller.set_cursor((0.2, 0.2))
    assert np.abs(interv(0, {})[:2]).max() > 0
    interv._handle_key("b")
    interv.pop_rewind(50)
    np.testing.assert_allclose(interv(10, {})[:2], 0.0)  # 커서가 다시 움직일 때까지 제자리


def test_back_is_one_second_by_default():
    assert _make().back_steps == 20  # 20Hz 제어


def test_reset_clears_pending_rewind():
    interv = _make()
    interv._handle_key("r")
    interv.reset()
    assert interv.pop_rewind(100) is None


def _modes(interv, pattern):
    """pattern(문자열, h=사람 p=정책)대로 스텝을 밟는다 — 스텝마다 모드가 기록된다."""
    for t, m in enumerate(pattern):
        if (m == "h") != interv._active:
            interv._handle_key("Tab")
        interv(t, {})
    return len(pattern)


def test_left_goes_back_to_where_the_current_mode_started():
    interv = _make()
    step = _modes(interv, "p" * 30 + "h" * 12)       # 30에서 개입, 지금 42
    interv._handle_key("Left")
    assert interv._paused
    assert interv.pop_rewind(step) == 30 and interv._active  # 개입 시작점, 사람 모드 그대로
    interv(30, {})
    interv._handle_key("Left")                         # 전환점 위에서 또 누르면 그 앞 전환점
    assert interv.pop_rewind(31) == 30
    interv._handle_key("Left")
    assert interv.pop_rewind(30) == 0 and not interv._active  # 정책 구간 시작 — 정책 모드로


def test_left_presses_accumulate_across_switches():
    interv = _make()
    step = _modes(interv, "p" * 10 + "h" * 5 + "p" * 5 + "h" * 5)
    interv._handle_key("Left"); interv._handle_key("Left")
    assert interv.pop_rewind(step) == 15 and not interv._active


def test_back_past_the_takeover_hands_control_back_to_the_policy():
    """개입 시작점 앞으로 되감으면 정책이 다시 몬다 — 개입 시점이 되감기 계산으로 앞당겨지지 않는다."""
    interv = _make()
    step = _modes(interv, "p" * 100 + "h" * 10)
    interv._handle_key("b")
    assert interv.pop_rewind(step) == 90
    assert interv(90, {}) is None                      # 90은 원래 정책 스텝
    interv._handle_key("r")
    assert interv.pop_rewind(91) == 0 and not interv._active


def test_rewind_to_jumps_to_that_step_pauses_and_restores_its_mode():
    """렌더 노이즈 복구용 — 수집기가 노이즈 직전 정상 스텝을 직접 지정한다. 키 되감기보다 우선."""
    interv = _make()
    step = _modes(interv, "p" * 20 + "h" * 10)
    interv._handle_key("b")
    interv.rewind_to(12)
    assert interv._paused
    assert interv.pop_rewind(step) == 12 and not interv._active  # 12는 정책 스텝
    assert interv.pop_rewind(12) is None                        # 요청은 한 번만


def test_reset_clears_a_pending_rewind_to():
    interv = _make()
    _modes(interv, "p" * 5)
    interv.rewind_to(2)
    interv.reset()
    assert interv.pop_rewind(5) is None


def _map_state():
    box = [np.array([0.0, 0.0]), np.array([0.02, 0.0]), np.array([0.02, 0.02]), np.array([0.0, 0.02])]
    return {**_state(), "peg_boxes": [box], "nut_boxes": [box, box]}


def test_cameras_stack_on_the_left_and_the_map_takes_the_right():
    pytest.importorskip("cv2")
    interv = MouseTeleopIntervention(env=object(), state_fn=_map_state, map_size=960)
    frames = [np.zeros((480, 480, 3), np.uint8), np.full((480, 480, 3), 255, np.uint8)]
    canvas = interv._compose(frames)
    assert canvas.shape == (960, 480 + 960, 3)
    assert canvas[300, 200].max() == 0 and canvas[700, 200].min() == 255  # agentview 위, wrist 아래


def test_camera_tiles_get_a_border_and_their_name():
    pytest.importorskip("cv2")
    interv = MouseTeleopIntervention(env=object(), state_fn=_map_state, map_size=960)
    gray = np.full((480, 480, 3), 128, np.uint8)
    plain = interv._compose([gray, gray])
    boxed = interv._compose({"agentview": gray, "robot0_eye_in_hand": gray})
    assert (plain[480, 240] == 128).all() and (plain[5:25, 5:60] == 128).all()  # 이름 없으면 안 그림
    assert (boxed[480, 240] != 128).any() and (boxed[240, 240] == 128).all()  # 경계선만, 가운데는 그대로
    assert (boxed[5:25, 5:60] != 128).any()  # 왼쪽 위 이름표


def test_cursor_only_follows_the_mouse_over_the_map():
    pytest.importorskip("cv2")
    interv = MouseTeleopIntervention(env=object(), state_fn=_map_state, map_size=960)
    interv._compose([np.zeros((480, 480, 3), np.uint8)] * 2)
    interv.controller.target_xy = None
    interv._on_motion(100, 480)  # 카메라 열 위
    assert interv.controller.target_xy is None
    interv._on_motion(480 + 480, 480)  # 맵 한가운데 = 테이블 중심
    np.testing.assert_allclose(interv.controller.target_xy, interv.map.center, atol=1e-3)
