"""Quest 3S 오른쪽 컨트롤러로 EE를 모는 제어 법칙 — MouseTeleopController를 대체한다.

- 그립(>= grip_threshold)을 누른 동안만 클러치가 물린다. 잡는 순간의 컨트롤러 포즈와 EE 포즈를 기준으로 삼아
  이후 컨트롤러가 움직인 양(1:1 × pos_scale/rot_scale)만큼 목표를 옮긴다. 그래서 잡기 전에 손이 어디 있든
  튀지 않는다. 놓으면 목표를 실제 EE에 붙여 그 자리에 선다.
- 트리거(>= trigger_threshold)는 그립과 별개로 그리퍼 닫힘(+1)/열림(−1)이다.
- XR 좌표(x 오른쪽, y 위, −z 앞)는 행렬 M으로 월드(x 로봇 앞쪽 반대, y 오른쪽, z 위)로 옮긴다.
- 위치와 (클러치 기준 대비) 상대 회전은 1-Euro 필터를 거친다. 회전은 rotvec이라 ±π 근처에서 부호가
  뒤집히므로, 필터 입력과 출력 액션 모두 직전 값에 가까운 쪽의 등가 rotvec을 고른다.
- 회전 입력은 rot_source로 고른다. "stick"(기본): 야우는 클러치 동안 컨트롤러가 수평으로 돈 각(앞 방향의 방위각)을
  따르고(손목 관절 범위에서 멈춘다), 롤·피치는 오른쪽 조이스틱이 헬기처럼 정한다 — 민 쪽으로 손목 위가 기울고(끝까지 = max_tilt)
  그리퍼 끝은 반대쪽을 향한다. 놓으면 수직 아래로 돌아온다. "pose": 컨트롤러 회전을 그대로 따른다.
"""

import math

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from square_assembly.runners.mouse_teleop import MouseTeleopController, _wrap
from square_assembly.runners.square_oracle import _POS_SCALE, _ROT_SCALE, yaw_of

# XR(x 오른쪽, y 위, −z 앞) -> 월드(+x 로봇에서 사람 쪽, +y 오른쪽, +z 위)
M = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)


# From: /home/hymm/Projects/manipulation_pipeline/src/flare/teleoperators/bi_piper_xr/bi_piper_xr.py
class OneEuroFilter:
    def __init__(self, dim, rate, min_cutoff, beta, d_cutoff):
        self.rate = rate
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = np.zeros(dim)

    def _alpha(self, cutoff):
        te = 1.0 / self.rate
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def filter(self, x):
        x = np.asarray(x, dtype=float)
        if self.x_prev is None:
            self.x_prev = x.copy()
            return x.copy()
        a_d = self._alpha(self.d_cutoff)
        dx = (x - self.x_prev) * self.rate
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        alpha = np.array([self._alpha(c) for c in cutoff])
        x_hat = alpha * x + (1 - alpha) * self.x_prev
        self.x_prev = x_hat.copy()
        self.dx_prev = dx_hat.copy()
        return x_hat


_xrt = None


def _head_yaw(q):
    """XR 위(+y)축 둘레 각 — 앱 기준점이 돈 양을 잰다."""
    return float(Rot.from_quat(q).as_euler("YXZ")[0])


class OriginTurnUndo:
    """XRoboToolkit 앱이 조이스틱 좌우로 기준점(XR Origin)을 돌린 것만 되돌린다.

    앱은 오른쪽 조이스틱 x를 밀면 기준점을 헤드셋 위치 둘레로 약 100도/s 돌린다(2026-11-10 실측: 헤드셋·
    컨트롤러 yaw가 함께 돌고 헤드셋 위치는 그대로, 조이스틱이 0일 때는 돌지 않음). 그래서 이번이나 직전 샘플에서
    |조이스틱 x| > stick_thr일 때의 헤드셋 yaw 변화만 앱 회전으로 보고 누적해 되돌린다. 조이스틱을 놓고 고개나
    몸을 돌린 것은 그대로 둔다. 조이스틱을 미는 동안 실제로 고개를 돌린 양은 앱 회전과 같이 되돌려진다.
    """

    def __init__(self, stick_thr=0.1):
        self.stick_thr = stick_thr
        self.R, self.t = Rot.identity(), np.zeros(3)  # 보정 변환: 보정 = R·raw + t
        self._yaw = self._sx = None

    def update(self, head, sx):
        yaw = _head_yaw(head[3:])
        if self._yaw is not None and max(abs(sx), abs(self._sx)) > self.stick_thr:
            d = Rot.from_rotvec([0.0, -_wrap(yaw - self._yaw), 0.0])  # raw에서 헤드셋 둘레로 되돌리는 회전
            self.t = self.R.apply(head[:3] - d.apply(head[:3])) + self.t
            self.R = self.R * d
        self._yaw, self._sx = yaw, sx

    def __call__(self, ctrl):
        out = np.array(ctrl, dtype=float)
        out[:3] = self.R.apply(out[:3]) + self.t
        out[3:] = (self.R * Rot.from_quat(out[3:])).as_quat()
        return out


def _pose_ok(x):
    return x.shape == (7,) and np.all(np.isfinite(x)) and np.linalg.norm(x[3:]) > 1e-6


def xrt_reader(headset_yaw_comp=True):
    """XRoboToolkit SDK로 오른쪽 컨트롤러를 읽는 reader를 돌려준다. SDK는 여기서 처음 import·init한다.
    headset_yaw_comp면 앱이 조이스틱으로 돌린 기준점을 OriginTurnUndo로 되돌린다(헤드셋을 못 읽은 스텝은 보정량 유지)."""
    global _xrt
    if _xrt is None:
        import xrobotoolkit_sdk as xrt
        xrt.init()
        _xrt = xrt
    xrt, undo = _xrt, OriginTurnUndo()

    def read():
        c = np.asarray(xrt.get_right_controller_pose(), dtype=float)
        stick = np.asarray(xrt.get_right_axis(), dtype=float)
        if headset_yaw_comp:
            h = np.asarray(xrt.get_headset_pose(), dtype=float)
            if _pose_ok(h):
                undo.update(h, float(stick[0]))
            if _pose_ok(c):
                c = undo(c)
        return c, float(xrt.get_right_grip()), float(xrt.get_right_trigger()), stick
    return read


def close_xrt():
    global _xrt
    if _xrt is not None:
        _xrt.close()
        _xrt = None


def _ctrl_yaw(cR):
    """월드로 옮긴 컨트롤러 회전의 앞 방향(XR −z = 월드 −x)이 월드 xy 평면에서 향하는 각."""
    f = cR.as_matrix() @ np.array([-1.0, 0.0, 0.0])
    return float(np.arctan2(f[1], f[0]))


def _near(rv, prev):
    """rv와 같은 회전인 rotvec(rv, rv−2π·축) 중 prev에 가까운 쪽 — ±π 근처 부호 뒤집힘 방지."""
    n = np.linalg.norm(rv)
    if n < 1e-9:
        return rv
    alt = rv * (1 - 2 * np.pi / n)
    return alt if np.linalg.norm(alt - prev) < np.linalg.norm(rv - prev) else rv


class QuestTeleopController(MouseTeleopController):
    """(포즈, 그립, 트리거)를 7-dim 액션으로 바꾼다. 창·SDK 무관(reader만 바꿔 끼우면 된다).

    Args:
        reader: () -> (pose7 [x,y,z,qx,qy,qz,qw], grip, trigger, stick [x,y]). None이면 xrt_reader().
        rot_source: "stick"(컨트롤러 야우 + 조이스틱 롤·피치) 또는 "pose"(컨트롤러 회전 추종).
        headset_yaw_comp: SDK reader에서 앱이 조이스틱으로 돌린 기준점을 헤드셋 yaw로 되돌린다(OriginTurnUndo).
        max_tilt: 조이스틱을 끝까지 밀었을 때 기울기(rad). stick_deadzone: 조이스틱 기울기가 이 미만이면 0으로 본다.
        pos_scale, rot_scale: 컨트롤러 이동/회전 → 목표 이동/회전 배율(1.0 == 1:1).
        grip_threshold, trigger_threshold: 클러치/그리퍼 판정 기준.
        grip_release_threshold: 클러치가 물린 뒤 이 미만으로 내려가야 놓은 것으로 친다(히스테리시스).
        axes: XR -> 월드 축 변환 행렬.
        euro_*: 1-Euro 필터 파라미터. dt는 제어 주기(s).
        pos_cap, rot_cap: 한 스텝 액션 상한. PD 없이 목표와 현재 EE의 차이를 그대로 위치·회전 명령으로 보낸다
            (goal_update_mode="achieved"라 OSC 목표 = 현재 EE + 차이 = 목표 포즈).
    """

    def __init__(self, reader=None, pos_scale=1.0, rot_scale=1.0, pos_cap=1.0, rot_cap=0.4,
                 grip_threshold=0.9, trigger_threshold=0.5, axes=M, grip_release_threshold=0.7,
                 euro_min_cutoff=1.0, euro_beta=0.01, euro_d_cutoff=1.0, dt=0.05,
                 rot_source="stick", max_tilt=np.deg2rad(30.0), stick_deadzone=0.15, headset_yaw_comp=True):
        super().__init__(pos_cap=pos_cap, rot_cap=rot_cap, dt=dt)
        assert rot_source in ("stick", "pose"), rot_source
        self.rot_source, self.max_tilt, self.stick_deadzone = rot_source, max_tilt, stick_deadzone
        self.reader = reader if reader is not None else xrt_reader(headset_yaw_comp)
        self.pos_scale, self.rot_scale = pos_scale, rot_scale
        self.grip_release_threshold = grip_release_threshold
        self.grip_threshold, self.trigger_threshold, self.axes = grip_threshold, trigger_threshold, np.asarray(axes)
        eu = dict(rate=1.0 / dt, min_cutoff=euro_min_cutoff, beta=euro_beta, d_cutoff=euro_d_cutoff)
        self._fp, self._fr = OneEuroFilter(3, **eu), OneEuroFilter(3, **eu)

    def reset(self):
        super().reset()
        self._armed = True      # 그립이 한 번 threshold 아래로 내려가야 다시 물린다(take_over 직후 False)
        self._ref = None        # 클러치 기준 (컨트롤러 월드 위치, 컨트롤러 월드 회전, EE 위치, EE 회전)
        self._rv = np.zeros(3)  # 연속화한 상대 rotvec(필터 입력)
        self._a_rot = np.zeros(3)  # 직전 회전 액션 rotvec(연속화 기준)
        self.target_p = self.target_R = None

    def take_over(self, state):
        super().take_over(state)
        self._ref, self._armed = None, False
        self._latch(np.asarray(state["grip"], float), Rot.from_matrix(state["R"]))
        self._sync_hud()

    @property
    def clutched(self):
        return self._ref is not None

    def _latch(self, p, R):
        self.target_p, self.target_R = p.copy(), R

    def set_cursor(self, world_xy):
        pass

    def wheel(self, notches, limit=False):
        return False

    def _sync_hud(self):
        self.target_xy = self.target_p[:2].copy()
        self.target_z = float(self.target_p[2])
        if self.rot_source == "pose":   # stick이면 target_yaw는 조이스틱이 직접 갖고 있다
            self.target_yaw = float(np.arctan2(*self.target_R.as_matrix()[[1, 0], 0]))

    def action(self, state):
        p, R = np.asarray(state["grip"], float), Rot.from_matrix(state["R"])
        pose, grip, trig, stick = self.reader()
        a = np.zeros(7, dtype=np.float32)
        a[6] = 1.0 if trig >= self.trigger_threshold else -1.0
        self.grip_cmd = float(a[6])

        pose = np.asarray(pose, float)
        valid = pose.shape == (7,) and np.all(np.isfinite(pose)) and np.linalg.norm(pose[3:]) > 1e-6
        if self.target_p is None:   # reset 뒤 첫 스텝 — 지금 EE를 한 번 잡아 둔다
            self._latch(p, R)
        if grip < (self.grip_release_threshold if self._ref is not None else self.grip_threshold):
            if self._ref is not None:   # 놓는 스텝 — 그 순간의 실제 EE를 목표로 고정한다
                self._latch(p, R)
            self._armed, self._ref = True, None
        if self._ref is None and valid and self._armed and grip >= self.grip_threshold:
            cp, cR = self.axes @ pose[:3], Rot.from_matrix(self.axes @ Rot.from_quat(pose[3:]).as_matrix() @ self.axes.T)
            self._ref = (cp, cR, p.copy(), R)
            self._fp.x_prev = self._fr.x_prev = None
            self._fp.dx_prev, self._fr.dx_prev = np.zeros(3), np.zeros(3)
            self._rv, self._a_rot = np.zeros(3), np.zeros(3)
            self._cy, self._dyaw = _ctrl_yaw(cR), 0.0   # stick: 클러치 동안 컨트롤러가 돈 야우(접지 않음)
            self._yaw0 = self.target_yaw if self.target_yaw is not None else yaw_of(state["R"])

        if self._ref is not None and not valid:
            # 잡은 채 포즈를 못 읽었다 — 놓을 때처럼 실제 EE에 고정하고, 다시 잡아야 따라간다
            # (낡은 기준·필터 상태로 재개돼 튀거나 접촉 중 계속 미는 것을 막는다)
            self._latch(p, R)
            self._armed, self._ref = False, None
        if self._ref is not None:   # 안 잡았으면 목표는 놓는 순간 고정한 것 그대로 — 실제 EE를 따라가지 않는다
            ref_cp, ref_cR, ref_p, ref_R = self._ref
            cp = self.axes @ pose[:3]
            cR = Rot.from_matrix(self.axes @ Rot.from_quat(pose[3:]).as_matrix() @ self.axes.T)
            self.target_p = ref_p + self.pos_scale * (self._fp.filter(cp) - ref_cp)
            self.target_p[2] = max(self.target_p[2], self.z_min)   # 테이블에 박지 않는다
            if self.rot_source == "pose":
                self._rv = _near((cR * ref_cR.inv()).as_rotvec(), self._rv)
                self.target_R = Rot.from_rotvec(self.rot_scale * self._fr.filter(self._rv)) * ref_R
            else:
                cy = _ctrl_yaw(cR)
                self._dyaw += _wrap(cy - self._cy)
                self._cy = cy
                self.target_yaw = self._yaw0 + self.rot_scale * float(self._fr.filter([self._dyaw, 0.0, 0.0])[0])
        self._sync_hud()

        a[:3] = np.clip((self.target_p - p) / _POS_SCALE, -self.pos_cap, self.pos_cap)
        if self.rot_source == "stick":
            lo, hi = self.yaw_limits(state)
            if self.target_yaw is None:
                self.target_yaw = self.yaw
            # 손목 범위 안으로 자르고, 반 바퀴 안쪽만 앞세운다(마우스 텔레옵 yaw_rot과 같다)
            yaw = self.yaw + np.clip(np.clip(self.target_yaw, lo, hi) - self.yaw, -2.5, 2.5)
            x, z = np.array([np.cos(yaw), np.sin(yaw), 0.0]), np.array([0.0, 0.0, -1.0])
            s = np.asarray(stick, float).reshape(-1)[:2] if np.size(stick) >= 2 else np.zeros(2)
            if not np.all(np.isfinite(s)) or np.linalg.norm(s) < self.stick_deadzone:
                s = np.zeros(2)
            # 헬기처럼 민 쪽으로 위가 기운다: 앞(+y 스틱) → 끝이 월드 +x(사람 쪽)로, 오른쪽(+x 스틱) → 끝이 월드 −y로
            self.target_R = Rot.from_rotvec(-self.max_tilt * np.array([s[0], s[1], 0.0])) * \
                Rot.from_matrix(np.column_stack([x, np.cross(z, x), z]))
        rot = _near((self.target_R * R.inv()).as_rotvec(), self._a_rot)
        self._a_rot = rot
        a[3:6] = np.clip(rot / _ROT_SCALE, -self.rot_cap, self.rot_cap)
        return a
