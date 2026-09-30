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

붙잡기(자석, 2026-09-30): 사람 모드로 멈춰 있으면 그리퍼가 **실제 마우스 커서**를 자석처럼 끈다.
커서 위치(_cur)는 마우스 이벤트에서 손이 움직인 양(δ)만 받아 쌓고, 자석은 그 위에 얹은 뒤 OS 포인터를
그 자리로 옮긴다(warp, 이벤트마다·화면 갱신마다). 붙을 때와 떨어질 때 원리가 다르다:
- 끌림: field_radius(4.5cm) 안에선 가까울수록 세게 그리퍼 쪽으로 끌려가고, snap_radius(2.5cm) 안에서 붙는다.
  붙는 순간 빠른 스프링(snap_hz, ζ 0.65 — 넘침 약 7%)이 커서를 중심에 박는다(착). 목표 = 그리퍼 중심,
  야우 목표 = 실제 야우. (8cm/3cm·ζ 0.35는 너무 멀리서 끌리고 많이 넘친다는 사용자 보고로 줄였다.)
- 붙음: 되돌리는 스프링이 아니라 끈적임이다 — 손이 당긴 양(_stretch)의 hold_gain만 커서가 바로 따라가고,
  손을 멈추면(60ms) 당긴 양이 hold_tau로 풀려 착 스프링으로 중심에 돌아온다. 손이 움직이는 동안엔 풀지
  않아서 포인터가 늘 손 방향으로만(느리게) 움직인다 — 손과 스프링이 싸우며 포인터가 반대로 튀는(드드득)
  일이 없다(스프링으로 붙잡던 첫 버전에서 사용자 보고, 당기는 중에도 풀던 버전은 XTest로 최대 7px 역행).
- 떨어짐: 당긴 양이 break_dist(4.5cm)를 넘으면 떨어지고, 커서는 보이던 자리에서 당긴 방향으로 pop_dist(1.2cm)를
  약 0.1초에 걸쳐 미끄러져 나간다(착). 손이 간 자리(당긴 양 전체)로 순간이동하던 버전은 늘어나 보이던
  것(35%)보다 훨씬 멀리 튀어 보였고, 6mm 순간이동은 떨어지는 느낌이 너무 약했다(사용자 보고). 그 자리에 쉬고 있으면 다시 끌려오지 않고, 끌림 반경을 벗어나거나 0.3초 뒤 그리퍼 쪽으로
  다가가면 다시 붙을 수 있다(반경 밖까지 나가야만 되던 버전은 조금 떨어졌다 다시 붙이기가 안 됐다).
- 야우엔 따로 떨어지는 조건이 없다: 위치가 붙어 있는 동안 휠은 위치와 같은 끈적임으로 hold_gain만큼
  비틀고, 휠을 멈추면 착 스프링으로 원래 각도에 돌아온다 — 세게 돌려도 야우만 떨어져 나가지 않는다.
  붙는 순간엔 돌려둔 목표와 실제 야우의 차이에서 출발해 착 돌아 붙는다. 비틀림은 늘 [−π, π)로 접어서
  짧은 쪽으로 돈다(350도 돌려둔 목표가 −350도를 되감던 문제, 사용자 보고).
  제어 목표 야우는 붙어 있는 동안 실제 야우에 그대로이고, 위치가 떨어질 때 보이던 비틀림을 가지고 풀린다.
δ는 X 이벤트 순서로 잰다: 포인터를 옮기면(XWarpPointer) X 서버가 옮긴 자리에 이벤트를 하나 보내고, 그보다
먼저 도착하는 이벤트는 옮기기 전에 생긴 것이다. 그래서 옮긴 자리를 줄(_pending)에 넣고, 그 자리 이벤트가
올 때까지는 직전 이벤트 기준, 온 뒤로는 옮긴 자리 기준으로 잰다(_on_motion). "가장 가까운 기준점" 추정은
붙어 있을 때(포인터를 매번 되돌림) 손 움직임을 지워 버려서 버렸다.

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

def _wrap(a):
    """각도를 [−π, π)로 — 야우 비틀림이 늘 짧은 쪽으로 돌게."""
    return (a + np.pi) % (2 * np.pi) - np.pi


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
                 back_key="b", restart_key="r", back_steps=20, state_fn=None,
                 snap_radius=0.025, field_radius=0.045, pull_rate=16.0, snap_hz=7.0, snap_zeta=0.65,
                 hold_gain=0.35, hold_tau=0.2, break_dist=0.045, pop_dist=0.012):
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
        self.snap_radius, self.field_radius, self.pull_rate = snap_radius, field_radius, pull_rate
        self.snap_omega, self.snap_zeta = 2 * np.pi * snap_hz, snap_zeta
        self.hold_gain, self.hold_tau, self.break_dist, self.pop_dist = hold_gain, hold_tau, break_dist, pop_dist
        self._suppress_motion = False
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
        self._release_grab()
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
        self._pump(seconds)

    def _pump(self, seconds):
        """seconds 동안 2ms마다 입력을 처리하며 기다린다. 통째로 sleep하면 그동안 쌓인 마우스 이벤트를
        깨어나서 한꺼번에 처리해, 붙잡기가 포인터를 몰아서 되돌리며 역행이 보였다(XTest 실측 6px)."""
        end = time.time() + seconds
        while time.time() < end:
            if self._root is not None:
                self._root.update()
            time.sleep(0.002)

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
        self._release_grab()       # 되돌린 장면에선 그리퍼가 딴 데 있다
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
            self._release_grab()
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
        view.bind("<Button-4>", lambda e: self._on_wheel(+1))   # X11 휠 위
        view.bind("<Button-5>", lambda e: self._on_wheel(-1))   # X11 휠 아래
        root.bind("<KeyPress>", lambda e: self._on_key(e.keysym, True))
        root.bind("<KeyRelease>", lambda e: self._on_key(e.keysym, False))
        root.bind("<FocusOut>", lambda e: setattr(c, "z_up", False) or setattr(c, "z_down", False))
        root.protocol("WM_DELETE_WINDOW", lambda: setattr(self, "_quit_requested", True))
        root.focus_force()
        self._root, self._view, self._photo = root, view, None

    def _release_grab(self):
        self._cur = None            # 자석이 끄는 커서 위치(월드 m) — 멈춘 사람 모드에서만 의미
        self._snapped = self._free_until_exit = False
        self._stretch = np.zeros(2)  # 붙은 뒤 손이 당긴 양(월드 m) — 커서는 그중 hold_gain만 따라간다
        self._idle = 0.0            # 손이 마지막으로 움직인 뒤 흐른 물리 시간(s) — 멈춰야 당긴 양이 풀린다
        self._since_release = 0.0   # 떨어진 뒤 흐른 물리 시간(s) — 바로 다시 붙지 않게
        self._o, self._ov = np.zeros(2), np.zeros(2)  # 붙은 커서의 그리퍼 기준 위치와 속도(착 스프링)
        self._yo, self._yov = 0.0, 0.0  # 붙은 야우의 비틀림(rad, [−π, π))과 그 속도(착 스프링)
        self._pop_left = np.zeros(2)  # 떨어진 뒤 아직 미끄러져 나갈 거리(월드 m)
        self._wheel_idle = 0.0      # 휠이 마지막으로 돈 뒤 흐른 물리 시간(s)
        self._snap_t = self._release_t = 0.0
        self._prev_px = None        # 직전 마우스 이벤트 위치(캔버스 px)
        self._pending = []          # 옮겼지만 X 서버의 그 자리 이벤트가 아직 안 온 포인터 자리 [(px, 시각)]
        self._last_warp = None
        self._tick_t = None

    def _magnet_on(self):
        return self._paused and self._active

    def _on_motion(self, x, y):
        """왼쪽 카메라 열 위에서는 목표를 안 바꾼다 — 맵은 그 오른쪽에 붙어 있다.
        멈춘 사람 모드에선 손이 움직인 양만 받아 자석에 넘기고 포인터를 바로 옮긴다(모듈 docstring '붙잡기')."""
        if self._suppress_motion:  # 우리가 포인터를 옮기며 Tk가 바로 부르는 합성 이벤트
            return
        if x < self._map_x0:
            self._cur, self._prev_px = None, None
            return
        pos = np.array([x, y], dtype=float)
        if not self._magnet_on() or self._cur is None or self._prev_px is None:
            self._cur = self.map.to_world(x - self._map_x0, y)
            self._prev_px = pos
            if self._magnet_on():
                self._update_magnet(self._state_fn())
            self._apply_target()
            return
        now = time.time()
        self._pending = [(w, t) for w, t in self._pending if now - t < 0.2]  # 합쳐져 사라진 이벤트는 포기
        for j, (w, _) in enumerate(self._pending):
            if np.abs(pos - w).max() <= 0.5:  # 옮긴 자리 이벤트 — 여기서부터 새 기준
                del self._pending[:j + 1]
                self._prev_px = pos
                return
        base, self._prev_px = self._prev_px, pos
        delta = self.map.to_world(*(pos - (self._map_x0, 0))) - self.map.to_world(*(base - (self._map_x0, 0)))
        st = self._state_fn()
        if self._snapped:
            self._stretch = self._stretch + delta
            self._o = self._o + self.hold_gain * delta  # 손 입력은 스프링을 거치지 않고 바로
            self._idle = 0.0
        else:
            toward = np.dot(delta, np.asarray(st["grip"][:2], dtype=float) - self._cur) > 0
            if self._free_until_exit and toward and self._since_release > 0.3:
                self._free_until_exit = False  # 떨어졌다가 다시 그리퍼 쪽으로 — 다시 끌린다
            self._cur = self._cur + delta
        self._update_magnet(st)
        self._place_cursor(st)
        self._apply_target()

    def _update_magnet(self, st):
        g = np.asarray(st["grip"][:2], dtype=float)
        if self._snapped and np.linalg.norm(self._stretch) > self.break_dist:
            self._snapped = False
            pull = self._stretch / np.linalg.norm(self._stretch)
            self._cur = g + self._o                 # 보이던 자리에서
            self._pop_left = self.pop_dist * pull   # 당긴 방향으로 미끄러져 나간다(_tick)
            self.controller.target_yaw = yaw_of(st["R"]) + self._yo  # 보이던 비틀림을 가지고 풀린다
            self._free_until_exit = True
            self._since_release = 0.0
            self._release_t = time.time()
            return
        if self._snapped:
            return
        dist = np.linalg.norm(self._cur - g)
        if self._free_until_exit and dist > self.field_radius:
            self._free_until_exit = False
        elif not self._free_until_exit and dist < self.snap_radius:
            self._snapped = True
            self._o, self._ov, self._stretch = self._cur - g, np.zeros(2), np.zeros(2)
            self._pop_left = np.zeros(2)
            yaw, prev = yaw_of(st["R"]), self.controller.target_yaw
            self._yo, self._yov = (0.0 if prev is None else _wrap(prev - yaw)), 0.0  # 짧은 쪽에서 출발
            self._snap_t = time.time()
            self.controller.target_yaw = yaw

    def _place_cursor(self, st):
        """붙어 있으면 커서 = 그리퍼 + 착 스프링 위치. 그 자리로 OS 포인터를 옮긴다."""
        if self._snapped:
            self._cur = np.asarray(st["grip"][:2], dtype=float) + self._o
        px = np.array(self.map.to_px(self._cur), dtype=float) + (self._map_x0, 0)
        px = np.round(px)
        ref = self._pending[-1][0] if self._pending else self._prev_px
        if ref is None or np.abs(px - ref).max() >= 1.0:
            self._warp_pointer(px)

    def _apply_target(self):
        if self._cur is None:
            return
        self.controller.set_cursor(np.asarray(self._state_fn()["grip"][:2], dtype=float)
                                   if self._magnet_on() and self._snapped else self._cur)

    def _on_wheel(self, notches):
        """휠 = 야우 목표. 자석에 붙어 있는 동안엔 끈적하게 비틀기만 하고 원래 각도로 돌아온다(모듈 docstring)."""
        if not (self._magnet_on() and self._snapped):
            self.controller.wheel(notches)
            return
        self._yo = _wrap(self._yo + self.hold_gain * notches * self.controller.yaw_step)  # 끈적하게, 바로
        self._wheel_idle = 0.0

    def _animate(self, st):
        """화면 갱신마다 부른다 — 지난 호출 뒤 흐른 시간만큼 자석 물리를 진행한다."""
        now = time.time()
        dt = 0.0 if self._tick_t is None else min(now - self._tick_t, 0.05)
        self._tick_t = now
        self._tick(st, dt)

    def _tick(self, st, dt):
        """자석 물리 dt초: 붙음 = 당긴 양이 풀리고 착 스프링이 따라감, 근처 = 끌림. 그 뒤 포인터를 옮긴다."""
        if not self._magnet_on() or self._cur is None:
            return
        g = np.asarray(st["grip"][:2], dtype=float)
        w, z = self.snap_omega, self.snap_zeta
        n = max(1, int(np.ceil(dt / 0.005)))
        h = dt / n
        for _ in range(n):  # 반 암시적 오일러, 5ms 이하
            self._idle += h
            self._wheel_idle += h
            self._since_release += h
            if self._snapped:
                if self._idle > 0.06:
                    self._stretch = self._stretch * np.exp(-h / self.hold_tau)
                self._ov = self._ov + (w * w * (self.hold_gain * self._stretch - self._o) - 2 * z * w * self._ov) * h
                self._o = self._o + self._ov * h
                if self._wheel_idle > 0.06:  # 휠을 멈추면 원래 각도로 — 도는 동안엔 손과 싸우지 않는다
                    self._yov += (-w * w * self._yo - 2 * z * w * self._yov) * h
                    self._yo = _wrap(self._yo + self._yov * h)
            else:
                step = self._pop_left * (1 - np.exp(-h / 0.03))  # 떨어진 뒤 미끄러짐(약 0.1초)
                self._cur, self._pop_left = self._cur + step, self._pop_left - step
            if not self._snapped and not self._free_until_exit:
                d = np.linalg.norm(g - self._cur)
                pull = np.clip((self.field_radius - d) / (self.field_radius - self.snap_radius), 0.0, 1.0)
                self._cur = self._cur + (g - self._cur) * min(1.0, self.pull_rate * pull * h)
            self._update_magnet(st)
        self._place_cursor(st)
        self._apply_target()

    def _warp_pointer(self, px):
        self._pending.append((px, time.time()))
        self._last_warp = px
        if self._root is not None:
            self._suppress_motion = True
            try:
                self._view.event_generate("<Motion>", warp=True, x=int(px[0]), y=int(px[1]))
            finally:
                self._suppress_motion = False

    # ---- render_fn 계약 -----------------------------------------------------------
    def render(self, frame):
        """frame(HWC uint8 RGB 한 장, 여러 장의 리스트, 또는 {카메라 이름: 영상})을 맵 왼쪽에 세로로
        쌓아 띄우고 입력을 처리한다. dict면 칸 사이 구분선과 카메라 이름표를 붙인다.

        일시정지 중엔 여기서 창을 계속 갱신하며 머문다(호출부의 스텝 루프가 그동안 멈춘다).
        되감기 요청이 들어오면 일시정지여도 돌아간다 — 호출부가 되감은 장면으로 다시 부르면
        그 장면에서 다시 멈춘다(그래서 멈춘 채로 ←/b를 연타해 뒤로 훑을 수 있다).
        """
        from PIL import Image, ImageTk

        if self._root is None:
            self._open_window()
        frames = frame if isinstance(frame, (list, tuple, dict)) else [frame]
        while True:
            img = Image.fromarray(self._compose(frames))
            if self._photo is None:
                self._photo = ImageTk.PhotoImage(img)
                self._view.configure(image=self._photo)
            else:
                self._photo.paste(img)
            self._root.update()
            if self._quit_requested or not self._paused or self._back or self._restart or self._switches:
                return not self._quit_requested
            self._pump(0.02)

    def _compose(self, frames):
        """[카메라들(위에서부터) | 맵] RGB 캔버스. 카메라 한 장의 높이 = 맵 높이 / 카메라 수.
        frames가 {이름: 영상}이면 칸 사이 구분선과 이름표를 붙인다. 맵 위 글씨·게이지는 HUD로 그린다."""
        import cv2
        from PIL import Image, ImageDraw

        names = list(frames) if isinstance(frames, dict) else [None] * len(frames)
        frames = list(frames.values()) if isinstance(frames, dict) else frames
        S = self.map.size
        h = S // len(frames)
        cams = []
        for f in frames:
            f = np.asarray(f)
            cams.append(f if f.shape[0] == h else cv2.resize(f, (int(f.shape[1] * h / f.shape[0]), h)))
        left = np.concatenate(cams, axis=0)
        left = np.pad(left, ((0, S - left.shape[0]), (0, 0), (0, 0)))
        self._map_x0 = x0 = left.shape[1]
        s = self._state_fn()
        self._animate(s)
        img = Image.fromarray(np.concatenate([left, self._draw_map(s)[:, :, ::-1]], axis=1)).convert("RGBA")
        over = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(over)
        k = S / 960
        if names[0] is not None:
            self._draw_camera_labels(d, names, h, x0, k)
        self._draw_hud(d, s, x0, S, k)
        return np.asarray(Image.alpha_composite(img, over).convert("RGB"))

    def _draw_camera_labels(self, d, names, h, x0, k):
        from square_assembly.runners import teleop_hud as hud

        for i, name in enumerate(names):
            if i:
                d.line((0, i * h, x0, i * h), fill=(12, 13, 16, 255), width=max(2, round(4 * k)))
            title, sub = ("WRIST", name) if name == "robot0_eye_in_hand" else (name.upper(), "")
            x, y, r = round(10 * k), i * h + round(10 * k), round(5 * k)
            f1, f2 = hud.font(round(14 * k), True), hud.font(round(13 * k))
            w = d.textlength(title + ("  " if sub else ""), font=f1) + (d.textlength(sub, font=f2) if sub else 0)
            hh = round(26 * k)
            tx = x + 5 * r + 4 * k
            d.rounded_rectangle((x, y, tx + w + 12 * k, y + hh), radius=hh // 2, fill=hud.PANEL)
            d.ellipse((x + 2 * r, y + hh / 2 - r, x + 4 * r, y + hh / 2 + r), fill=(120, 200, 255, 255))
            d.text((tx, y + hh / 2), title, font=f1, fill=hud.TEXT, anchor="lm")
            if sub:
                d.text((tx + d.textlength(title + "  ", font=f1), y + hh / 2), sub, font=f2, fill=hud.DIM,
                       anchor="lm")
        d.line((x0 - 1, 0, x0 - 1, self.map.size), fill=(12, 13, 16, 255), width=max(2, round(4 * k)))

    def _draw_hud(self, d, s, x0, S, k):
        from square_assembly.runners import teleop_hud as hud

        c = self.controller
        mode_col = hud.HUMAN if self._active else hud.POLICY
        # 왼쪽 위: 모드 알약(+ 멈춤·되감기 안내)
        label = "HUMAN" if self._active else "POLICY"
        x = hud.pill(d, (x0 + 16 * k, 16 * k), label, mode_col + (255,),
                     color=(255, 255, 255) if self._active else (16, 24, 18), size=round(18 * k))
        if self._paused:
            x = hud.pill(d, (x + 8 * k, 16 * k), "PAUSED", hud.PAUSED + (255,), size=round(18 * k))
            msg = "S  resume"
            if self._rewound is not None:
                msg = f"step {self._rewound[0]} \u2192 {self._rewound[1]}     " + msg
            if self._active and not self._snapped:
                msg += "     hover the gripper to grab it"
            hud.pill(d, (x + 8 * k, 16 * k), msg, hud.PANEL, color=hud.TEXT, size=round(15 * k), bold=False,
                     pad=(12, 8))

        # 오른쪽: z 게이지(z_min~z_max), peg 윗면 눈금, 바닥~현재 z 채움
        top, bot = 80 * k, S - 150 * k
        gx0, gx1 = x0 + S - 40 * k, x0 + S - 26 * k
        def z_to_py(z):
            return bot - (np.clip(z, c.z_min, c.z_max) - c.z_min) / (c.z_max - c.z_min) * (bot - top)
        zy = z_to_py(s["grip"][2])
        d.rounded_rectangle((gx0, top, gx1, bot), radius=(gx1 - gx0) / 2, fill=(255, 255, 255, 28),
                            outline=(255, 255, 255, 70), width=1)
        d.rounded_rectangle((gx0, zy, gx1, bot), radius=(gx1 - gx0) / 2, fill=mode_col + (190,))
        py = z_to_py(0.95)
        d.line((gx0 - 6 * k, py, gx1 + 6 * k, py), fill=(64, 196, 196, 255), width=max(2, round(3 * k)))
        d.text((gx0 - 10 * k, py), "peg", font=hud.font(round(12 * k)), fill=(64, 196, 196), anchor="rm")
        d.rounded_rectangle((gx0 - 4 * k, zy - 4 * k, gx1 + 4 * k, zy + 4 * k), radius=4 * k, fill=(255, 255, 255, 255))
        d.text((gx0 - 10 * k, zy), f"{s['grip'][2]:.2f}", font=hud.font(round(13 * k), True), fill=hud.TEXT,
               anchor="rm")
        dz = float(c.z_up) - float(c.z_down)
        if self._active and dz:  # z는 위치가 아니라 속도 명령 — 누르고 있는 방향을 삼각형으로
            cx, tip = (gx0 + gx1) / 2, zy - 22 * k * dz
            d.polygon([(cx, tip), (cx - 7 * k, zy - 10 * k * dz), (cx + 7 * k, zy - 10 * k * dz)], fill=(255, 255, 255))

        # 붙잡기: 자석 반경, 붙는 순간 조여드는 링, 떨어지는 순간 퍼지는 파문, 늘어난 만큼 가늘어지는 줄
        gpx = np.array(self.map.to_px(s["grip"][:2]), dtype=float) + (x0, 0)
        if self._magnet_on():
            R, now = self.snap_radius * self.map.scale, time.time()
            if self._snapped:
                d.ellipse((*(gpx - R), *(gpx + R)), fill=mode_col + (50,))
            else:
                d.ellipse((*(gpx - R), *(gpx + R)), outline=(255, 255, 255, 70), width=max(1, round(2 * k)))
            age = now - self._snap_t
            if age < 0.35:
                rr = R * (1.6 - 1.0 * age / 0.35)
                d.ellipse((*(gpx - rr), *(gpx + rr)), outline=(255, 255, 255, int(230 * (1 - age / 0.35))),
                          width=max(2, round(3 * k)))
            age = now - self._release_t
            if age < 0.35:
                rr = R * (1.0 + 1.2 * age / 0.35)
                d.ellipse((*(gpx - rr), *(gpx + rr)), outline=mode_col + (int(200 * (1 - age / 0.35)),),
                          width=max(2, round(2 * k)))
            if self._snapped and self._cur is not None:
                cp = np.array(self.map.to_px(self._cur), dtype=float) + (x0, 0)
                pull = min(1.0, np.linalg.norm(self._stretch) / self.break_dist)
                if np.linalg.norm(cp - gpx) > 2:
                    d.line((*gpx, *cp), fill=(255, 255, 255, int(200 * (1 - 0.5 * pull))),
                           width=max(1, round((5 - 3.5 * pull) * k)))

        # 아래: 상태 줄 + 키 안내 패널
        px0, py0, px1, py1 = x0 + 16 * k, S - 120 * k, x0 + S - 60 * k, S - 16 * k
        d.rounded_rectangle((px0, py0, px1, py1), radius=14 * k, fill=hud.PANEL)
        sz = round(15 * k)
        lx, ly = px0 + 16 * k, py0 + 22 * k
        yaw = np.degrees(yaw_of(s["R"]))
        hud.row(d, (lx, ly), [("text", "z"), ("value", f"{s['grip'][2]:.3f} m"), ("text", "   yaw"),
                              ("value", f"{yaw:+.0f}\u00b0"), ("text", "   grip"),
                              ("value", "CLOSE" if c.grip_cmd > 0 else "OPEN")], size=sz)
        hud.row(d, (lx, ly + 32 * k), [("key", "Tab"), ("text", "human/policy"), ("key", "\u2190"),
                                       ("text", "last switch"), ("key", "B"), ("text", "back 1s"),
                                       ("key", "R"), ("text", "restart")], size=sz)
        hud.row(d, (lx, ly + 64 * k), [("key", "S"), ("text", "pause"), ("key", "Q"), ("text", "give up"),
                                       ("key", "Ctrl"), ("key", "Shift"), ("text", "up/down"),
                                       ("key", "Wheel"), ("text", "yaw"), ("key", "LMB"), ("text", "grip")], size=sz)

    def _draw_map(self, s):
        """맵 도형(격자·peg·너트·그리퍼)만 cv2로 BGR에 그린다 — 글씨·게이지는 _draw_hud."""
        import cv2

        S, m = self.map.size, self.map
        img = np.full((S, S, 3), (36, 33, 30), dtype=np.uint8)
        c = self.controller

        # 테이블(맵 전체 폭 = table 한 변)과 10cm 격자
        for k in np.arange(-0.4, 0.41, 0.1):
            px, _ = m.to_px((0.0, m.center[1] + k))
            _, py = m.to_px((m.center[0] + k, 0.0))
            cv2.line(img, (px, 0), (px, S), (54, 50, 46), 1, cv2.LINE_AA)
            cv2.line(img, (0, py), (S, py), (54, 50, 46), 1, cv2.LINE_AA)

        # peg, 너트: sim의 충돌 박스를 위에서 본 실제 크기로. 손잡이(마지막 박스)를 밝은 색으로 먼저 칠하고
        # 고리로 덮는다 — 손잡이는 고리 벽 속까지 들어가 있어서, 보이는 밝은 부분이 곧 쥘 수 있는 구간이다.
        def fill(pts, color):
            hull = cv2.convexHull(np.array([m.to_px(p) for p in pts], dtype=np.int32))
            cv2.fillConvexPoly(img, hull, color, cv2.LINE_AA)

        for pts in s["peg_boxes"]:
            fill(pts, (196, 196, 64))
        *ring, handle = s["nut_boxes"]
        fill(handle, (60, 205, 255))
        for pts in ring:
            fill(pts, (50, 127, 205))

        # 그리퍼: 실제(굵게, 간격 = 실제 열림)와 사람 제어 중엔 명령 목표(흰 선). 목표는 커서·휠·클릭을
        # 그 자리에서 바로 따라가고 실제 그리퍼는 PD·OSC로 뒤따라온다 — 둘을 겹쳐 봐야 조종감이 맞는다.
        col = (72, 99, 255) if self._active else (113, 204, 46)  # teleop_hud.HUMAN / POLICY (BGR)
        k = max(1, round(S / 480))
        self._draw_gripper(img, s["grip"][:2], yaw_of(s["R"]), self._finger_gap() or 0.08, col, 3 * k)
        if self._active:
            if self._magnet_on() and self._cur is not None:  # 멈춘 사람 모드: 자석이 끄는 커서 자세
                txy = self._cur
                tyaw = c.target_yaw if c.target_yaw is not None else yaw_of(s["R"])
                if self._snapped:  # 붙어 있으면 휠로 비튼 만큼 보인다(원래 각도로 돌아온다)
                    tyaw = yaw_of(s["R"]) + self._yo
            else:
                txy = c.target_xy if c.target_xy is not None else s["grip"][:2]
                tyaw = c.target_yaw if c.target_yaw is not None else yaw_of(s["R"])
            self._draw_gripper(img, txy, tyaw, 0.02 if c.grip_cmd > 0 else 0.08, (255, 255, 255), 2 * k)
        return img

    def _draw_gripper(self, img, xy, yaw, gap, col, thickness):
        """중심 십자 + 손가락 두 개(손가락 축 = yaw 방향, 간격 gap m)."""
        import cv2

        m = self.map
        u = np.array([np.cos(yaw), np.sin(yaw)])
        n = np.array([-u[1], u[0]])
        cv2.drawMarker(img, m.to_px(xy), col, cv2.MARKER_CROSS, 7 * thickness, max(1, thickness // 3), cv2.LINE_AA)
        for sign in (1, -1):
            tip = np.asarray(xy)[:2] + sign * (gap / 2) * u
            cv2.line(img, m.to_px(tip - 0.008 * n), m.to_px(tip + 0.008 * n), col, thickness, cv2.LINE_AA)  # 패드 폭 1.6cm

    def close(self):
        if self._root is not None:
            self._root.destroy()
            self._root = None
