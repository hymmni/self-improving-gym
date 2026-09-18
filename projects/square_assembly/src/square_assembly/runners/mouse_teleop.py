"""마우스+키보드로 사람이 개입하는 뷰어/컨트롤러 — 2D 탑뷰 맵 위에서 EE를 몬다.

정책이 돌다가 `h`를 누르면 사람이 잡고, `p`로 정책에 돌려준다(그 사이 프레임은 INTV, 직전
구간은 PREINTV — SIRIUS 스킴, intervention_rollout.collect_episode가 라벨링한다. 돌려준
뒤 다시 잡으면 개입이 여러 번 기록된다). `s`는 일시정지(sim·기록 모두 멈추고 창만 살아
있음), `q`는 에피소드 포기.

사람 제어 중 매 스텝(20Hz)의 7-dim OSC_POSE delta 액션:
- xy: 맵 위 커서 위치를 목표로 PD  (kp·err − kd·v, pos_cap으로 클립)
- z: Space/Shift 누른 동안 일정 속도로 ↑/↓ (z_min~z_max에서 멈춤)
- 회전: 휠 한 칸 = 야우 목표 ±yaw_step. 손목은 항상 수직 아래(오라클과 같은 자세 제어)
- 그리퍼: 좌클릭 누른 동안 +1(닫힘) 쪽으로, 뗀 동안 −1(열림) 쪽으로 grip_rate씩 램프

맵은 카메라 영상이 아니라 sim 좌표(특권 정보, 표시 전용)로 직접 그린다 — 픽셀↔월드가
선형이라 캘리브레이션 없이 커서를 곧바로 목표 좌표로 쓴다. 화면 방향은 agentview와
맞췄다: 화면 오른쪽 = 월드 +y, 화면 위 = 월드 −x(로봇 쪽이 위).

키 입력 경로가 둘인 이유: cv2.waitKey는 창에 포커스가 있을 때만 오고 Shift 단독 입력과
키 떼기를 못 본다. 그래서 단발 키(h/p/s/q)는 cv2, "누르고 있는 동안"(Space/Shift)과 휠은
pynput 리스너로 받는다(pynput은 전역 리스너라 다른 창에 타이핑해도 잡힌다 — 사람 제어
중에만 z에 반영되므로 실사용에선 문제가 안 됐다).
"""

import os

import numpy as np

from square_assembly.runners.square_oracle import (
    _POS_SCALE, read_privileged_state, rot_delta_toward, yaw_of,
)

_GAP_CLOSED = 0.05  # 손가락 간격(m)이 이보다 좁으면 "쥔 상태"로 보고 닫힘 명령에서 시작


class MouseTeleopController:
    """입력 상태(커서·키·버튼·휠)를 7-dim 액션으로 바꾸는 제어 법칙. cv2/pynput 무관.

    Args:
        kp, kd: xy PD 게인. err(m)·v(m/step)에 곱해 delta(m)를 만든 뒤 _POS_SCALE로 나눈다.
        pos_cap: xy delta 액션 상한(1.0 == 5cm/step). 오라클(0.5)보다 낮춰야 사람이 따라간다.
        z_speed: Space/Shift 누른 동안의 z 액션(0.2 == 1cm/step).
        z_min, z_max: 그리퍼 site z 허용 범위(m). 테이블 윗면 0.82, peg 윗면 0.95.
        yaw_step: 휠 한 칸당 야우 목표 변화(rad).
        rot_cap: 회전 delta 액션 상한.
        grip_rate: 스텝당 그리퍼 명령 변화량(0.1 == 20스텝=1초에 완전 개폐).
    """

    def __init__(self, kp=1.0, kd=0.0, pos_cap=0.3, z_speed=0.2, z_min=0.83, z_max=1.10,
                 yaw_step=np.deg2rad(5.0), rot_cap=0.4, grip_rate=0.1):
        self.kp, self.kd, self.pos_cap = kp, kd, pos_cap
        self.z_speed, self.z_min, self.z_max = z_speed, z_min, z_max
        self.yaw_step, self.rot_cap, self.grip_rate = yaw_step, rot_cap, grip_rate
        self.reset()

    def reset(self):
        self.target_xy = None      # None이면 xy는 제자리 유지(잡은 뒤 커서가 아직 안 움직임)
        self.target_yaw = None     # None이면 현재 야우 유지
        self.grip_cmd = -1.0
        self.z_up = self.z_down = False
        self.grip_pressed = False
        self._prev_xy = None

    def take_over(self, state, finger_gap=None):
        """사람이 잡는 순간 목표를 현재 자세로 초기화 — 그 전에 쌓인 커서/휠로 튀지 않게."""
        self.target_xy = None
        self.target_yaw = yaw_of(state["R"])
        self._prev_xy = state["grip"][:2].copy()
        self.grip_cmd = 1.0 if (finger_gap is not None and finger_gap < _GAP_CLOSED) else -1.0

    def set_cursor(self, world_xy):
        self.target_xy = np.asarray(world_xy, dtype=float)[:2].copy()

    def wheel(self, notches):
        if self.target_yaw is not None:
            self.target_yaw += notches * self.yaw_step

    def action(self, state):
        grip, R = state["grip"], state["R"]
        a = np.zeros(7, dtype=np.float32)

        xy = grip[:2]
        v = xy - self._prev_xy if self._prev_xy is not None else np.zeros(2)
        self._prev_xy = xy.copy()
        if self.target_xy is not None:
            delta = self.kp * (self.target_xy - xy) - self.kd * v
            a[:2] = np.clip(delta / _POS_SCALE, -self.pos_cap, self.pos_cap)

        dz = float(self.z_up) - float(self.z_down)
        if (dz > 0 and grip[2] >= self.z_max) or (dz < 0 and grip[2] <= self.z_min):
            dz = 0.0
        a[2] = self.z_speed * dz

        yaw = self.target_yaw if self.target_yaw is not None else yaw_of(R)
        a[3:6], _ = rot_delta_toward(R, yaw, self.rot_cap)

        self.grip_cmd = float(np.clip(
            self.grip_cmd + self.grip_rate * (1.0 if self.grip_pressed else -1.0), -1.0, 1.0))
        a[6] = self.grip_cmd
        return a


class TopDownMap:
    """월드 xy ↔ 맵 픽셀. 화면 오른쪽 = +y, 화면 아래 = +x(agentview에서 로봇이 위쪽에 보이는 방향)."""

    def __init__(self, center_xy, extent_m, size_px):
        self.center = np.asarray(center_xy, dtype=float)[:2]
        self.size = int(size_px)
        self.scale = self.size / float(extent_m)  # px per m

    def to_px(self, xy):
        x, y = float(xy[0]), float(xy[1])
        px = self.size / 2 + (y - self.center[1]) * self.scale
        py = self.size / 2 + (x - self.center[0]) * self.scale
        return int(round(px)), int(round(py))

    def to_world(self, px, py):
        y = self.center[1] + (px - self.size / 2) / self.scale
        x = self.center[0] + (py - self.size / 2) / self.scale
        return np.array([x, y])


class MouseTeleopIntervention:
    """intervention_fn + render_fn 계약을 함께 구현하는 사람 개입 장치.

    Args:
        env: robomimic EnvRobosuite(또는 raw robosuite env). 맵과 제어에 sim 상태를 읽는다.
        controller: MouseTeleopController(None이면 기본 게인).
        map_size: 맵 한 변 픽셀. 오른쪽 agentview 프레임도 이 높이로 맞춘다.
        map_extent: 맵이 덮는 월드 폭(m). 테이블 한 변이 0.8m.
        state_fn: 테스트용 — sim 상태 dict를 돌려주는 함수(None이면 read_privileged_state).
    """

    _HELP = "[h]=human  [p]=policy  [s]=pause  [q]=give up   Space/Shift=z up/down  wheel=yaw  LMB=grip"

    def __init__(self, env, controller=None, map_size=480, map_extent=0.8, window_name="rollout",
                 takeover_key="h", handback_key="p", pause_key="s", quit_key="q", state_fn=None):
        self.raw = getattr(env, "env", env)
        self.controller = controller or MouseTeleopController()
        table = getattr(self.raw, "table_offset", None)
        self.map = TopDownMap(table[:2] if table is not None else (0.0, 0.0), map_extent, map_size)
        self.window_name = window_name
        self.keys = {ord(takeover_key): "takeover", ord(handback_key): "handback",
                     ord(pause_key): "pause", ord(quit_key): "quit"}
        self._state_fn = state_fn or (lambda: read_privileged_state(self.raw))
        self._listener = None      # pynput, render()에서 지연 생성(테스트는 창 없이 돈다)
        self._mouse_bound = False
        self._last_obs = None
        self.reset()

    # ---- intervention_fn 계약 -------------------------------------------------
    def reset(self):
        self._active = False
        self._paused = False
        self._quit_requested = False
        self.num_triggers = 0
        self.controller.reset()

    def should_end(self):
        return self._quit_requested

    def __call__(self, step, obs_raw):
        self._last_obs = obs_raw
        if not self._active:
            return None
        return self.controller.action(self._state_fn())

    # ---- 입력 -> 상태 -------------------------------------------------------------
    def _finger_gap(self):
        q = None if self._last_obs is None else self._last_obs.get("robot0_gripper_qpos")
        return None if q is None else float(q[0] - q[1])

    def _handle_key(self, key):
        """cv2 키 코드 -> 상태 갱신. cv2 없이 테스트 가능하게 render()에서 분리."""
        what = self.keys.get(key)
        if what == "takeover" and not self._active:
            self._active = True
            self.num_triggers += 1
            self.controller.take_over(self._state_fn(), self._finger_gap())
        elif what == "handback":
            self._active = False
        elif what == "pause":
            self._paused = not self._paused
        elif what == "quit":
            self._quit_requested = True

    def _on_mouse(self, event, x, y, flags, _param):
        import cv2

        if event == cv2.EVENT_MOUSEMOVE:
            if x < self.map.size:  # 오른쪽 agentview 패널 위에서는 목표를 안 바꾼다
                self.controller.set_cursor(self.map.to_world(x, y))
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.controller.grip_pressed = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.controller.grip_pressed = False
        # 휠은 여기로 안 온다 — cv2 Qt 창은 EVENT_MOUSEWHEEL을 콜백에 넘기지 않고 창 확대에 써버린다
        # (2026-09-18 rupy 주입 테스트: expanded/GUI_NORMAL 둘 다 콜백 0건). 휠은 pynput 마우스 리스너로 받는다.

    def _start_inputs(self):
        import cv2
        from pynput import keyboard as pynput_keyboard
        from pynput import mouse as pynput_mouse

        from square_assembly.runners.intervention_rollout import KeyboardIntervention

        cv2.setMouseCallback(self.window_name, self._on_mouse)
        self._mouse_bound = True
        up = KeyboardIntervention._resolve_keys("space", pynput_keyboard)
        down = KeyboardIntervention._resolve_keys("shift", pynput_keyboard)

        def is_up(key):
            # RustDesk 같은 원격 데스크톱은 공백을 "키"가 아니라 문자 ' '로 주입해서 Key.space가
            # 아니라 KeyCode(char=' ')로 들어온다(2026-09-18 pororo 실측: Shift는 되는데 Space만 무반응).
            return key in up or getattr(key, "char", None) == " "

        debug = bool(os.environ.get("MOUSE_TELEOP_DEBUG"))

        def on_press(key):
            if debug:
                print(f"[teleop] key press: {key!r} char={getattr(key, 'char', None)!r} vk={getattr(key, 'vk', None)!r}", flush=True)
            if is_up(key):
                self.controller.z_up = True
            elif key in down:
                self.controller.z_down = True

        def on_release(key):
            if is_up(key):
                self.controller.z_up = False
            elif key in down:
                self.controller.z_down = False

        self._listener = pynput_keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.start()
        # 전역 리스너라 어느 창 위에서 굴려도 야우가 돈다 — 사람 제어 중(_active)에만 반영된다.
        self._wheel_listener = pynput_mouse.Listener(on_scroll=lambda x, y, dx, dy: self.controller.wheel(dy))
        self._wheel_listener.start()

    # ---- render_fn 계약 -----------------------------------------------------------
    def render(self, frame):
        """frame(HWC uint8 RGB, agentview 고해상도)을 맵 옆에 붙여 띄우고 키를 처리한다.

        일시정지 중엔 여기서 창을 계속 갱신하며 머문다(호출부의 스텝 루프가 그동안 멈춘다).
        """
        import cv2

        if not self._mouse_bound:
            self._start_inputs()
        while True:
            cv2.imshow(self.window_name, self._compose(frame))
            self._handle_key(cv2.waitKey(50 if self._paused else 1) & 0xFF)
            if self._quit_requested or not self._paused:
                return not self._quit_requested

    def _compose(self, frame):
        import cv2

        rgb = np.asarray(frame)
        if rgb.shape[0] != self.map.size:
            rgb = cv2.resize(rgb, (int(rgb.shape[1] * self.map.size / rgb.shape[0]), self.map.size))
        right = np.ascontiguousarray(rgb[:, :, ::-1])
        canvas = np.concatenate([self._draw_map(), right], axis=1)
        mode = "PAUSED" if self._paused else ("HUMAN" if self._active else "policy")
        color = (0, 200, 255) if self._paused else ((0, 0, 255) if self._active else (0, 200, 0))
        cv2.putText(canvas, mode, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(canvas, self._HELP, (8, canvas.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1)
        return canvas

    def _draw_map(self):
        import cv2

        S, m = self.map.size, self.map
        img = np.full((S, S, 3), 40, dtype=np.uint8)
        s = self._state_fn()
        c = self.controller

        # 테이블(맵 전체 폭 = table 한 변)과 10cm 격자
        for k in np.arange(-0.4, 0.41, 0.1):
            px, _ = m.to_px((0.0, m.center[1] + k))
            _, py = m.to_px((m.center[0] + k, 0.0))
            cv2.line(img, (px, 0), (px, S), (60, 60, 60), 1)
            cv2.line(img, (0, py), (S, py), (60, 60, 60), 1)

        # peg, 너트(정사각형 + 핸들 막대)
        cv2.circle(img, m.to_px(s["peg"]), int(0.02 * m.scale), (200, 200, 60), -1)
        yaw_nut = np.arctan2(*(s["handle"] - s["nut"])[[1, 0]])
        half, cy_, sy_ = 0.025, np.cos(yaw_nut), np.sin(yaw_nut)
        corners = [s["nut"][:2] + half * np.array([cy_ * i - sy_ * j, sy_ * i + cy_ * j])
                   for i, j in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
        cv2.polylines(img, [np.array([m.to_px(p) for p in corners], dtype=np.int32)], True,
                      (80, 160, 255), 2)
        cv2.line(img, m.to_px(s["nut"]), m.to_px(s["handle"]), (80, 160, 255), 4)

        # 그리퍼: 중심 십자 + 손가락 두 개(손가락 축 = R의 x열, 간격 = 실제 열림)
        gap = self._finger_gap() or 0.08
        u = s["R"][:2, 0]
        u = u / (np.linalg.norm(u) + 1e-9)
        n = np.array([-u[1], u[0]])
        col = (0, 0, 255) if self._active else (0, 220, 0)
        gx, gy = m.to_px(s["grip"])
        cv2.drawMarker(img, (gx, gy), col, cv2.MARKER_CROSS, 14, 1)
        for sign in (1, -1):
            tip = s["grip"][:2] + sign * (gap / 2) * u
            cv2.line(img, m.to_px(tip - 0.01 * n), m.to_px(tip + 0.01 * n), col, 3)
        if c.target_xy is not None and self._active:
            cv2.circle(img, m.to_px(c.target_xy), 6, (255, 255, 255), 1)

        # z 바(오른쪽 가장자리): z_min~z_max, 눈금 = 테이블 윗면/peg 윗면, 마커 = 현재 z
        x0, top, bot = S - 18, 40, S - 40
        cv2.rectangle(img, (x0, top), (x0 + 8, bot), (120, 120, 120), 1)
        def z_to_py(z):
            return int(bot - (np.clip(z, c.z_min, c.z_max) - c.z_min) / (c.z_max - c.z_min) * (bot - top))
        for z_ref in (0.95,):
            cv2.line(img, (x0 - 4, z_to_py(z_ref)), (x0 + 12, z_to_py(z_ref)), (200, 200, 60), 1)
        cv2.rectangle(img, (x0 - 2, z_to_py(s["grip"][2]) - 3), (x0 + 10, z_to_py(s["grip"][2]) + 3), col, -1)
        cv2.putText(img, f"z {s['grip'][2]:.3f}  yaw {np.degrees(yaw_of(s['R'])):+.0f}  grip {c.grip_cmd:+.1f}",
                    (8, S - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        return img

    def close(self):
        import cv2

        for l in (self._listener, getattr(self, "_wheel_listener", None)):
            if l is not None:
                l.stop()
        cv2.destroyWindow(self.window_name)
