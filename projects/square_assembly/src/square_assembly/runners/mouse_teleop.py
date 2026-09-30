"""마우스+키보드로 사람이 개입하는 뷰어/컨트롤러 — 2D 탑뷰 맵 위에서 EE를 몬다.

정책이 돌다가 `Tab`을 누르면 사람이 잡고, 다시 `Tab`이면 정책에 돌려준다(그 사이 프레임은
INTV, 직전 구간은 PREINTV — SIRIUS 스킴, intervention_rollout.collect_episode가 라벨링한다.
돌려준 뒤 다시 잡으면 개입이 여러 번 기록된다). `s`는 일시정지(sim·기록 모두 멈추고 창만 살아
있음), `q`는 에피소드 포기.

되감기: `←`는 직전 모드 전환점(지금 모드가 시작된 스텝 — 개입 중이면 개입 시작점)으로, 이미
전환점에 서 있으면 그 앞 전환점으로 간다. `b`는 back_steps(기본 20 = 1초)만큼(연타하면 누적),
`r`은 에피소드 시작점으로 — 같은 너트 배치로 다시 한다. 되감으면 일시정지로 멈춰 어디로
돌아왔는지 보여주고, 모드도 그 스텝에서 쓰던 대로 돌아간다: 개입 시작점보다 앞으로 가면 다시
정책이 몬다. 그래서 기록되는 개입 시작은 늘 사람이 실제로 Tab을 누른 스텝이다(되감기 계산이
개입 시점을 앞당기지 않는다). 되감기 자체(sim·정책 청크 복원, 기록 자르기)는 여기서 안 한다 —
pop_rewind()로 돌아갈 스텝만 꺼내 주고, 수집기가 collect_episode의 pre_step_fn에서 처리한다.
버리지 않고 되돌리는 이유: robosuite는 reset마다 배치를 새로 뽑고 시드를 안 남겨서, q로 버린
어려운 배치는 다시 만날 수 없다.

사람 제어 중 매 스텝(20Hz)의 7-dim OSC_POSE delta 액션:
- xy: 맵 위 커서 위치를 목표로 PD  (kp·err − kd·v, pos_cap으로 클립)
- z: Ctrl(또는 Space)/Shift 누른 동안 일정 속도로 ↑/↓ (z_min~z_max에서 멈춤)
- 회전: 휠 한 칸 = 야우 목표 ±yaw_step. 손목은 항상 수직 아래(오라클과 같은 자세 제어)
- 그리퍼: 좌클릭 누른 동안 +1(닫힘), 뗀 동안 −1(열림). 일정 속도로 여닫는 램프는 robosuite
  PandaGripper가 이미 한다(부호만 보고 내부 명령을 옮긴다). 예전엔 여기서도 스텝당 0.1씩 램프를
  걸었는데, 부호가 뒤집히는 10스텝(0.5초) 동안 손가락이 전혀 안 움직여 반응만 늦었다(2026-09-22
  실측: 누른 뒤 움직이기 시작 0.55초 → 0.05초). PH 시연의 그리퍼 액션도 ±1뿐이다.
  넘겨받는 순간(Tab·되감기)에도 버튼만 따른다. 예전엔 손가락이 닫혀 있으면 클릭이 올 때까지 닫아뒀는데
  (쥔 너트를 Tab 순간 떨어뜨리지 않게), 헛집기로 빈손인 채 넘겨받아도 닫혀 있어 누르고 떼야 열렸다
  (2026-09-30 사용자 보고). 이제 쥔 너트를 유지하려면 버튼을 누른 채 Tab을 누른다.

맵은 카메라 영상이 아니라 sim 좌표(특권 정보, 표시 전용)로 직접 그린다 — 픽셀↔월드가
선형이라 캘리브레이션 없이 커서를 곧바로 목표 좌표로 쓴다. 화면 방향은 agentview와
맞췄다: 화면 오른쪽 = 월드 +y, 화면 위 = 월드 −x(로봇 쪽이 위).

창은 tkinter다(모든 입력을 창 하나의 이벤트로 받는다). 예전 cv2 창 + pynput 조합을 버린 이유
(2026-09-22):
- cv2(Qt) 창은 휠을 창 확대에 써버리고 막을 방법이 없었고, waitKey는 Shift 단독·키 떼기를 못 본다.
- 그래서 Space/Shift를 pynput 전역 리스너 + "누름 이벤트가 0.6초 안에 또 와야 유지" 방식으로
  받았는데, 이게 개입 중 z가 안 먹는 원인이었다. X11은 Shift 같은 수식키를 키 반복하지 않아서
  누름 이벤트가 한 번뿐이고, Space의 키 반복은 뗌+누름 쌍으로 와서 뗌이 유지를 끊는다. 결국
  둘 다 꾹 눌러도 처음 0.6초만 움직였다(pororo Xvfb에 XTest로 2초 누름 주입: Space 0.70초·
  Shift 0.60초만 이동).
Tk 창은 Shift 단독과 떼기를 그대로 받으므로 "누름~뗌 사이 = 누르고 있음"으로 충분하다. 키 반복의
뗌+누름 쌍은 한 번의 update()에서 연달아 처리돼 스텝 사이에 끊기지 않는다. 창 포커스를 잃으면
(FocusOut) 떼기가 안 올 수 있으니 z를 멈춘다.

입력기(XIM)는 끈다. pororo 세션엔 ibus-hangul이 XIM 서버로 떠 있고 한/영 전환 키가
`Hangul,Shift+space,Alt_R`이라, Tk가 기본값대로 XIM을 거치면 Shift(내리기)를 누른 채 Space(올리기)를
누르는 순간 입력기가 한/영 전환으로 가져가 Space가 안 먹었다(2026-09-22 사용자 보고, :1에서
`tk useinputmethods` = 1, XIM_SERVERS = @server=ibus 확인). 한글 모드가 되면 글자 키도 조합에 먹힌다.

올리기는 Ctrl이 기본이다. RustDesk(클라이언트 PC가 Wayland)는 Space를 "누르고 있음"으로 못 보낸다 —
꾹 누르면 누름+뗌이 같은 밀리초에 붙은 쌍을 키 반복 주기(33Hz)로 보낸다(2026-09-22 :1 XRecord 실측).
그래서 창에선 Space가 한 순간도 눌린 상태가 아니다. 수식키(Shift·Ctrl)는 누름·뗌이 제대로 온다.
Space도 남겨둔다(로컬 키보드나 X11 클라이언트에선 된다).
"""

import time

import numpy as np

from square_assembly.runners.square_oracle import (
    _POS_SCALE, read_privileged_state, rot_delta_toward, yaw_of,
)

class MouseTeleopController:
    """입력 상태(커서·키·버튼·휠)를 7-dim 액션으로 바꾸는 제어 법칙. 창(Tk) 무관.

    Args:
        kp, kd: xy PD 게인. err(m)·v(m/step)에 곱해 delta(m)를 만든 뒤 _POS_SCALE로 나눈다.
        pos_cap: xy delta 액션 상한(1.0 == 5cm/step). 오라클(0.5)보다 낮춰야 사람이 따라간다.
        z_speed: Space/Shift 누른 동안의 z 액션. 액션 1.0이 목표 5cm지만 OSC(kp=150, 임계 감쇠)는 한
            스텝(50ms)에 그 1/3쯤만 따라가서, 0.6이 실제 약 1cm/step이다. 0.2였을 땐 너트를 들고
            내려가던 정책을 넘겨받으면 Space를 0.5초 눌러도 −2~+3mm로 거의 안 올라갔다(2026-09-22
            pororo 실측, 0.6: 0.15초 안에 반등·0.5초에 4~6.5cm·뗀 뒤 5~9mm 더 감).
        z_min, z_max: 그리퍼 site z 허용 범위(m). 테이블 윗면 0.82, peg 윗면 0.95.
        yaw_step: 휠 한 칸당 야우 목표 변화(rad).
        rot_cap: 회전 delta 액션 상한.
    """

    def __init__(self, kp=1.0, kd=0.0, pos_cap=0.3, z_speed=0.6, z_min=0.83, z_max=1.10,
                 yaw_step=np.deg2rad(5.0), rot_cap=0.4):
        self.kp, self.kd, self.pos_cap = kp, kd, pos_cap
        self.z_speed, self.z_min, self.z_max = z_speed, z_min, z_max
        self.yaw_step, self.rot_cap = yaw_step, rot_cap
        self.reset()

    def reset(self):
        self.target_xy = None      # None이면 xy는 제자리 유지(잡은 뒤 커서가 아직 안 움직임)
        self.target_yaw = None     # None이면 현재 야우 유지
        self.grip_cmd = -1.0
        self.z_up = self.z_down = False  # 올리기/내리기 키를 누르고 있는 동안 True(창의 누름·뗌 이벤트)
        self.grip_pressed = False
        self._prev_xy = None

    def take_over(self, state):
        """사람이 잡는 순간 목표를 현재 자세로 초기화 — 그 전에 쌓인 커서/휠로 튀지 않게.
        그리퍼는 손가락 상태와 무관하게 지금 버튼 상태를 따른다(모듈 docstring)."""
        self.target_xy = None
        self.target_yaw = yaw_of(state["R"])
        self._prev_xy = state["grip"][:2].copy()
        self.grip_cmd = 1.0 if self.grip_pressed else -1.0

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

        self.grip_cmd = 1.0 if self.grip_pressed else -1.0
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
        map_size: 맵 한 변 픽셀. 왼쪽 카메라 열(위에서부터 세로로 쌓음)도 이 높이로 맞춘다.
        map_extent: 맵이 덮는 월드 폭(m). 테이블 한 변이 0.8m.
        state_fn: 테스트용 — sim 상태 dict를 돌려주는 함수(None이면 read_privileged_state).
        키 인자들은 Tk keysym이다("Tab", "Left", 소문자 한 글자).
    """

    _HELP = ("[Tab] human/policy   [<-] last switch   [b] back 1s   [r] restart",
             "[s] pause   [q] give up   Ctrl/Shift up/down   wheel yaw   LMB grip")

    def __init__(self, env, controller=None, map_size=480, map_extent=0.8, window_name="rollout",
                 toggle_key="Tab", switch_key="Left", pause_key="s", quit_key="q",
                 back_key="b", restart_key="r", back_steps=20, state_fn=None):
        self.raw = getattr(env, "env", env)
        self.controller = controller or MouseTeleopController()
        table = getattr(self.raw, "table_offset", None)
        self.map = TopDownMap(table[:2] if table is not None else (0.0, 0.0), map_extent, map_size)
        self.window_name = window_name
        self.back_steps = back_steps
        self.keys = {toggle_key: "toggle", switch_key: "last_switch",
                     pause_key: "pause", quit_key: "quit",
                     back_key: "back", restart_key: "restart"}
        self._state_fn = state_fn or (lambda: read_privileged_state(self.raw))
        self._root = None          # Tk 창, render()에서 지연 생성(테스트는 창 없이 돈다)
        self._last_obs = None
        self._map_x0 = 0          # 캔버스에서 맵이 시작하는 x — 왼쪽 카메라 열의 폭
        self.reset()

    # ---- intervention_fn 계약 -------------------------------------------------
    def reset(self):
        self._active = False
        self._paused = False
        self._quit_requested = False
        self._back, self._restart, self._switches, self._resync = 0, False, 0, False
        self._goto = None          # rewind_to로 지정한 스텝(렌더 노이즈 복구) — 키 되감기보다 우선
        self._modes = []           # 스텝별 사람 제어 여부 — 되감으면 그 스텝의 모드로 돌아간다
        self._rewound = None       # (되감기 전 스텝, 돌아간 스텝) — 멈춘 화면에 띄운다
        self.num_triggers = 0
        self.controller.reset()

    def should_end(self):
        return self._quit_requested

    def rewind_to(self, step):
        """수집기가 돌아갈 스텝을 직접 지정한다(렌더 노이즈 직전의 정상 스텝) — 되돌린 뒤 일시정지."""
        self._goto = step
        self._paused = True

    def idle(self, seconds):
        """창을 살려둔 채 기다린다(렌더 회복 대기) — 그냥 sleep하면 창이 응답 없음이 된다."""
        end = time.time() + seconds
        while time.time() < end:
            if self._root is not None:
                self._root.update()
            time.sleep(0.05)

    def pop_rewind(self, step):
        """되감기 요청이 있으면 돌아갈 스텝 번호를(없으면 None) 돌려주고 요청을 비운다.

        모드(사람/정책)도 돌아간 스텝에서 쓰던 대로 되돌린다.
        """
        if not (self._restart or self._back or self._switches or self._goto is not None):
            return None
        if self._goto is not None:
            t = self._goto
        elif self._restart:
            t = 0
        elif self._back:
            t = max(0, step - self._back)
        else:
            t = step
            for _ in range(self._switches):
                t = self._last_switch_before(t)
        self._back, self._restart, self._switches, self._goto = 0, False, 0, None
        if t < len(self._modes):
            self._active = self._modes[t]
            del self._modes[t:]
        self._resync = True
        self._rewound = (step, t)
        return t

    def _last_switch_before(self, step):
        """step보다 앞에서 가장 가까운 모드 전환점(새 모드의 첫 스텝). 없으면 0."""
        m = self._modes
        for s in range(min(step, len(m)) - 1, 0, -1):
            if m[s] != m[s - 1]:
                return s
        return 0

    def __call__(self, step, obs_raw):
        self._last_obs = obs_raw
        self._modes[step:] = [self._active]
        resync, self._resync = self._resync, False
        if not self._active:
            return None
        if resync:  # 되감기 전 목표(커서·야우·그리퍼)를 들고 가면 재개하자마자 팔이 튄다
            self.controller.take_over(self._state_fn())
        return self.controller.action(self._state_fn())

    # ---- 입력 -> 상태 -------------------------------------------------------------
    def _finger_gap(self):
        q = None if self._last_obs is None else self._last_obs.get("robot0_gripper_qpos")
        return None if q is None else float(q[0] - q[1])

    def _handle_key(self, key):
        """Tk keysym -> 상태 갱신. Tk 없이 테스트 가능하게 이벤트 처리와 분리."""
        what = self.keys.get(key)
        if what == "toggle":
            self._active = not self._active
            if self._active:
                self.num_triggers += 1
                self.controller.take_over(self._state_fn())
        elif what == "pause":
            self._paused = not self._paused
            if not self._paused:
                self._rewound = None
        elif what == "quit":
            self._quit_requested = True
        elif what == "last_switch":
            self._switches += 1
            self._paused = True
        elif what == "back":
            self._back += self.back_steps
            self._paused = True
        elif what == "restart":
            self._restart = True
            self._paused = True

    def _on_key(self, keysym, down):
        """Tk 누름/뗌 이벤트. 올리기/내리기 키는 누르고 있는 동안만 z를 움직이고, 나머지는 누를 때 한 번."""
        if keysym in ("Control_L", "Control_R", "space"):
            self.controller.z_up = down
        elif keysym in ("Shift_L", "Shift_R"):
            self.controller.z_down = down
        elif down:
            # Shift를 누른 채면 keysym이 대문자·ISO_Left_Tab으로 온다 — z를 내리면서도 키가 먹게.
            key = "Tab" if keysym == "ISO_Left_Tab" else (keysym.lower() if len(keysym) == 1 else keysym)
            self._handle_key(key)
        return "break"  # Tab이 Tk 기본 동작(포커스 이동)으로 새지 않게

    def _open_window(self):
        import tkinter as tk

        root = tk.Tk()
        root.tk.call("tk", "useinputmethods", False)  # ibus-hangul이 Shift+Space를 가져가지 않게(모듈 docstring)
        root.title(self.window_name)
        root.resizable(False, False)  # 창 = 캔버스 크기 고정 — 늘리면 빈 여백만 생긴다
        view = tk.Label(root, bd=0, highlightthickness=0)
        view.pack()
        c = self.controller

        view.bind("<Motion>", lambda e: self._on_motion(e.x, e.y))
        view.bind("<ButtonPress-1>", lambda e: setattr(c, "grip_pressed", True))
        view.bind("<ButtonRelease-1>", lambda e: setattr(c, "grip_pressed", False))
        view.bind("<Button-4>", lambda e: c.wheel(+1))   # X11 휠 위
        view.bind("<Button-5>", lambda e: c.wheel(-1))   # X11 휠 아래
        root.bind("<KeyPress>", lambda e: self._on_key(e.keysym, True))
        root.bind("<KeyRelease>", lambda e: self._on_key(e.keysym, False))
        root.bind("<FocusOut>", lambda e: setattr(c, "z_up", False) or setattr(c, "z_down", False))
        root.protocol("WM_DELETE_WINDOW", lambda: setattr(self, "_quit_requested", True))
        root.focus_force()
        self._root, self._view, self._photo = root, view, None

    def _on_motion(self, x, y):
        """왼쪽 카메라 열 위에서는 목표를 안 바꾼다 — 맵은 그 오른쪽에 붙어 있다."""
        if x >= self._map_x0:
            self.controller.set_cursor(self.map.to_world(x - self._map_x0, y))

    # ---- render_fn 계약 -----------------------------------------------------------
    def render(self, frame):
        """frame(HWC uint8 RGB 한 장, 여러 장의 리스트, 또는 {카메라 이름: 영상})을 맵 왼쪽에 세로로
        쌓아 띄우고 입력을 처리한다. dict면 칸마다 테두리와 카메라 이름을 붙인다.

        일시정지 중엔 여기서 창을 계속 갱신하며 머문다(호출부의 스텝 루프가 그동안 멈춘다).
        되감기 요청이 들어오면 일시정지여도 돌아간다 — 호출부가 되감은 장면으로 다시 부르면
        그 장면에서 다시 멈춘다(그래서 멈춘 채로 ←/b를 연타해 뒤로 훑을 수 있다).
        """
        from PIL import Image, ImageTk

        if self._root is None:
            self._open_window()
        frames = frame if isinstance(frame, (list, tuple, dict)) else [frame]
        while True:
            img = Image.fromarray(np.ascontiguousarray(self._compose(frames)[:, :, ::-1]))  # BGR -> RGB
            if self._photo is None:
                self._photo = ImageTk.PhotoImage(img)
                self._view.configure(image=self._photo)
            else:
                self._photo.paste(img)
            self._root.update()
            if self._quit_requested or not self._paused or self._back or self._restart or self._switches:
                return not self._quit_requested
            time.sleep(0.05)

    def _compose(self, frames):
        """[카메라들(위에서부터) | 맵] BGR 캔버스. 카메라 한 장의 높이 = 맵 높이 / 카메라 수.
        frames가 {이름: 영상}이면 칸 경계가 보이게 테두리와 이름표를 그린다."""
        import cv2

        names = list(frames) if isinstance(frames, dict) else [None] * len(frames)
        frames = list(frames.values()) if isinstance(frames, dict) else frames
        S = self.map.size
        h = S // len(frames)
        cams = []
        for f, name in zip(frames, names):
            f = np.asarray(f)
            f = f if f.shape[0] == h else cv2.resize(f, (int(f.shape[1] * h / f.shape[0]), h))
            f = np.ascontiguousarray(f[:, :, ::-1])  # RGB -> BGR
            if name is not None:
                label = name + (" (wrist)" if name == "robot0_eye_in_hand" else "")
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(f, (0, 0), (f.shape[1] - 1, h - 1), (0, 200, 255), 2)
                cv2.rectangle(f, (2, 2), (tw + 14, th + 14), (30, 30, 30), -1)
                cv2.putText(f, label, (8, th + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
            cams.append(f)
        left = np.concatenate(cams, axis=0)
        left = np.pad(left, ((0, S - left.shape[0]), (0, 0), (0, 0)))
        self._map_x0 = left.shape[1]
        canvas = np.ascontiguousarray(np.concatenate([left, self._draw_map()], axis=1))
        x0 = self._map_x0
        mode = "PAUSED" if self._paused else ("HUMAN" if self._active else "policy")
        color = (0, 200, 255) if self._paused else ((0, 0, 255) if self._active else (0, 200, 0))
        cv2.putText(canvas, mode, (x0 + 8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if self._paused and self._rewound is not None:
            src, dst = self._rewound
            cv2.putText(canvas, f"<< step {src} -> {dst}  [s]=resume",
                        (x0 + 120, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        # 도움말은 맵 왼쪽 아래에 두 줄로, 맵 크기에 비례한 글씨로(480 맵 기준 0.35 → 960 맵 0.7)
        scale, thick = 0.35 * self.map.size / 480, max(1, round(self.map.size / 480))
        line_h = int(32 * self.map.size / 960)
        for i, line in enumerate(self._HELP):
            y = canvas.shape[0] - 12 - (len(self._HELP) - 1 - i) * line_h
            cv2.putText(canvas, line, (x0 + 12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick)
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

        # peg, 너트: sim의 충돌 박스를 위에서 본 실제 크기로. 손잡이(마지막 박스)를 밝은 색으로 먼저 칠하고
        # 고리로 덮는다 — 손잡이는 고리 벽 속까지 들어가 있어서, 보이는 밝은 부분이 곧 쥘 수 있는 구간이다.
        def fill(pts, color):
            cv2.fillConvexPoly(img, cv2.convexHull(np.array([m.to_px(p) for p in pts], dtype=np.int32)), color)

        for pts in s["peg_boxes"]:
            fill(pts, (200, 200, 60))
        *ring, handle = s["nut_boxes"]
        fill(handle, (0, 210, 255))
        for pts in ring:
            fill(pts, (60, 120, 200))

        # 그리퍼: 실제(굵게, 간격 = 실제 열림)와 사람 제어 중엔 명령 목표(흰 선). 목표는 커서·휠·클릭을
        # 그 자리에서 바로 따라가고 실제 그리퍼는 PD·OSC로 뒤따라온다 — 둘을 겹쳐 봐야 조종감이 맞는다.
        col = (0, 0, 255) if self._active else (0, 220, 0)
        self._draw_gripper(img, s["grip"][:2], yaw_of(s["R"]), self._finger_gap() or 0.08, col, 3)
        if self._active:
            self._draw_gripper(img, c.target_xy if c.target_xy is not None else s["grip"][:2],
                               c.target_yaw if c.target_yaw is not None else yaw_of(s["R"]),
                               0.02 if c.grip_cmd > 0 else 0.08, (255, 255, 255), 2)

        # z 바(오른쪽 가장자리): z_min~z_max, 눈금 = peg 윗면, 마커 = 현재 z, 바닥~현재 z는 채운다.
        # 맵 크기에 비례해 굵게 — 480 기준 두께 그대로 960 맵에 그렸더니 안 보였다(2026-09-30).
        k = max(1, S // 480)
        x0, top, bot = S - 18 * k, 40, S - 40
        def z_to_py(z):
            return int(bot - (np.clip(z, c.z_min, c.z_max) - c.z_min) / (c.z_max - c.z_min) * (bot - top))
        zy = z_to_py(s["grip"][2])
        cv2.rectangle(img, (x0, zy), (x0 + 8 * k, bot), (100, 100, 100), -1)
        cv2.rectangle(img, (x0, top), (x0 + 8 * k, bot), (170, 170, 170), k)
        for z_ref in (0.95,):
            cv2.line(img, (x0 - 4 * k, z_to_py(z_ref)), (x0 + 12 * k, z_to_py(z_ref)), (200, 200, 60), 2 * k)
        cv2.rectangle(img, (x0 - 3 * k, zy - 3 * k), (x0 + 11 * k, zy + 3 * k), col, -1)
        dz = float(c.z_up) - float(c.z_down)
        if self._active and dz:  # z는 목표 위치가 아니라 속도 명령이라, 누르고 있는 방향을 화살표로
            cv2.arrowedLine(img, (x0 - 10 * k, zy), (x0 - 10 * k, zy - int(24 * k * dz)), (255, 255, 255),
                            2 * k, tipLength=0.4)
        cv2.putText(img, f"z {s['grip'][2]:.3f}  yaw {np.degrees(yaw_of(s['R'])):+.0f}  grip {c.grip_cmd:+.1f}",
                    (12, S - 12 - 2 * int(32 * S / 960) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35 * S / 480,
                    (200, 200, 200), max(1, round(S / 480)))
        return img

    def _draw_gripper(self, img, xy, yaw, gap, col, thickness):
        """중심 십자 + 손가락 두 개(손가락 축 = yaw 방향, 간격 gap m)."""
        import cv2

        m = self.map
        u = np.array([np.cos(yaw), np.sin(yaw)])
        n = np.array([-u[1], u[0]])
        cv2.drawMarker(img, m.to_px(xy), col, cv2.MARKER_CROSS, 14, 1)
        for sign in (1, -1):
            tip = np.asarray(xy)[:2] + sign * (gap / 2) * u
            cv2.line(img, m.to_px(tip - 0.008 * n), m.to_px(tip + 0.008 * n), col, thickness)  # 패드 폭 1.6cm

    def close(self):
        if self._root is not None:
            self._root.destroy()
            self._root = None
