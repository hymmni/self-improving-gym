"""NutAssemblySquare를 특권 정보(sim 상태)로 직접 푸는 룰 기반 오라클.

학습 대상이 아니라 "성공 데모를 만들어내는 도구"다 — SIRIUS 스킴에서 사람이 개입해
실패를 수습하던 자리를 스크립트가 대신한다(scripted_intervention.py에서 트리거 키로
호출되고, 그 프레임들은 intervention_fn 경로라 INTV로 라벨된다).

정책처럼 관측(이미지)만 보는 게 아니라 sim에서 너트/핸들/peg 위치를 그대로 읽어
폐루프 P 제어로 웨이포인트를 따라간다. 그래서 정책이 어떤 실패 상태에 있든(너트를
떨어뜨렸든, 잘못 쥐었든) 일단 놓고 처음부터 다시 집어서 꽂는다. 한 번에 안 되면
(들어올리기 실패 등) 같은 절차를 처음부터 반복한다 — 포기하지 않는다.

실측 성공률(2026-09-17, 고정 시드 20개, robosuite raw env):
- reset 직후부터 돌리면 20/20, 성공까지 중앙값 195스텝
- 무작위 액션 60스텝으로 흐트러뜨린 뒤 돌리면 20/20(스텝 상한 900), 중앙값 224·최대 539스텝
  → 실패 모드는 "못 한다"가 아니라 "스텝이 모자란다"뿐이라, max_steps를 넉넉히 줘야 한다.

측정값(2026-09-17 서버 robosuite 1.5.1 NutAssemblySquare/Panda 실측):
- OSC_POSE delta 스케일: action[:3] 1.0 == 0.05m/step, action[3:6] 1.0 == 0.5rad/step
- 핸들 site는 너트 중심에서 5.4cm 떨어져 있고, 그 방향각 == 너트 yaw
- OSC delta 회전은 월드/베이스 프레임 기준(robot0_base xmat == I) axis-angle이다
- peg1은 [0.23, 0.1, 0.85]의 box(half-height 0.1) — 윗면이 z=0.95라 그보다 높이 들어야
  너트가 peg를 넘어간다. 성공 판정(NutAssembly._check_success)은 너트 중심이 peg에서
  xy 3cm 이내 + z < 0.87 + **eef가 너트에서 4.3cm 이상 떨어져 있을 것**(r_reach<0.6)이라
  마지막에 반드시 놓고 물러나야 한다.
"""

import numpy as np

_POS_SCALE = 0.05
_ROT_SCALE = 0.5
_HALF_PI = np.pi / 2
_GRIPPER_OPEN, _GRIPPER_CLOSE = -1.0, 1.0

# 각 페이즈의 스텝 상한 — 조건을 못 맞춰도 여기서 다음(또는 재시도)으로 넘어간다.
_TIMEOUT = {
    "release": 10, "clear": 40, "align": 80, "descend": 60, "grasp": 10,
    "lift": 40, "over_peg": 80, "insert": 60, "open": 5, "retreat": 15,
}


def _wrap_axis(angle):
    """축(180° 대칭) 각도 오차를 [-pi/2, pi/2)로 접는다 — 그리퍼는 뒤집어 잡아도 같다."""
    return (angle + _HALF_PI) % np.pi - _HALF_PI


def yaw_of(R):
    """그리퍼 회전행렬의 손가락 축(x열)이 월드 xy 평면에서 향하는 각."""
    return float(np.arctan2(R[1, 0], R[0, 0]))


def rot_delta_toward(cur, yaw, rot_cap):
    """그리퍼를 수직 아래로 + 손가락 축을 yaw로 향하게 만드는 OSC delta 회전(axis-angle).

    Returns:
        (delta(3,), angle): rot_cap으로 클립된 액션과 남은 회전각(rad).
    """
    x = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    z = np.array([0.0, 0.0, -1.0])
    target = np.column_stack([x, np.cross(z, x), z])

    dR = target @ cur.T
    angle = float(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1.0, 1.0)))
    if angle < 1e-6:
        return np.zeros(3), 0.0
    axis = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / (2 * np.sin(angle))
    return np.clip(axis * angle / _ROT_SCALE, -rot_cap, rot_cap), angle


def read_privileged_state(raw):
    """sim에서 그리퍼/너트/핸들/peg 상태를 한 번에 읽는다(오라클·텔레옵 맵 공용).

    인덱스는 robosuite가 env 생성 시 잡아둔 것 그대로 쓴다. 매 호출마다 새로 읽으므로
    env.reset() 뒤에도 안전하다.
    """
    sim = raw.sim
    nut = raw.nuts[getattr(raw, "nut_id", 0)]
    grip_id = raw.robots[0].eef_site_id[raw.robots[0].arms[0]]
    return {
        "grip": sim.data.site_xpos[grip_id].copy(),
        # 열: [손가락이 벌어지는 축, y, 접근 축] — x축이 손가락 축인 건 finger body 위치로 확인(2026-09-17)
        "R": sim.data.site_xmat[grip_id].reshape(3, 3).copy(),
        "nut": sim.data.body_xpos[raw.obj_body_id[nut.name]].copy(),
        "handle": sim.data.site_xpos[sim.model.site_name2id(nut.important_sites["handle"])].copy(),
        "peg": sim.data.body_xpos[raw.peg1_body_id].copy(),
    }


class SquareAssemblyOracle:
    """intervention_fn 계약(`(step, obs_raw) -> action | None`)을 그대로 따르는 오라클.

    Args:
        env: robomimic EnvRobosuite(또는 raw robosuite env). sim/nuts/peg1_body_id를 읽는다.
        carry_z (float): 너트를 옮길 때의 그리퍼 높이. peg 윗면(0.95)보다 높아야 한다.
        grasp_dz (float): 핸들 site 위 몇 m에서 그리퍼를 닫을지.
        insert_z (float): peg 위에서 내려가는 목표 높이. 너트 중심이 z<0.87까지 내려가야
            성공으로 쳐주는데, 살짝 기운 채로 얕게 놓으면 peg 중간(z~0.91)에 끼어서 영영
            안 내려간다 — 쥔 채로 여기까지 눌러 내리면 그 끼임이 풀린다(0.91: 성공 7/10,
            0.88: 5/10, 0.85: 20/20 — 2026-09-17 고정 시드 실측).
        retreat_z (float): 놓은 뒤 물러날 높이(성공 판정의 r_reach 조건 때문에 필요).
    """

    def __init__(self, env, carry_z=1.02, grasp_dz=0.004, insert_z=0.85, retreat_z=1.06,
                 pos_cap=0.5, rot_cap=0.4):
        self.raw = getattr(env, "env", env)
        self.carry_z = carry_z
        self.grasp_dz = grasp_dz
        self.insert_z = insert_z
        self.retreat_z = retreat_z
        self.pos_cap = pos_cap
        self.rot_cap = rot_cap
        self.attempts = 0
        self.phase = "release"
        self._t = 0

    def reset(self):
        self.attempts = 0
        self._go("release")

    def _go(self, phase):
        self.phase = phase
        self._t = 0

    def _state(self):
        return read_privileged_state(self.raw)

    def _move(self, cur, target):
        return np.clip((np.asarray(target) - cur) / _POS_SCALE, -self.pos_cap, self.pos_cap)

    def _rot(self, s):
        """그리퍼를 수직 아래로 + 손가락 축이 핸들 막대와 직교하게 만드는 delta 회전.

        yaw만 맞추면 안 된다 — 정책이 실패한 자세는 손목이 기울어져 있는 경우가 많고,
        그 상태로 내려가면 손가락이 핸들을 비껴 지나가 빈손으로 닫힌다(2026-09-17 실측,
        무작위 액션 60스텝으로 흐트러뜨린 20개 시드: yaw만 제어 7/20 -> 자세 전체 제어 20/20).
        """
        handle_dir = np.arctan2(*(s["handle"] - s["nut"])[[1, 0]])
        cur_angle = yaw_of(s["R"])
        # 손가락 축은 180° 대칭이라 뒤집힌 해 중 가까운 쪽을 목표로 잡는다.
        yaw = cur_angle + _wrap_axis(handle_dir + _HALF_PI - cur_angle)
        return rot_delta_toward(s["R"], yaw, self.rot_cap)

    def _drop_target_xy(self, s):
        """지금 쥔 자세를 유지한 채 너트 중심이 peg 위로 가는 그리퍼 xy."""
        return s["peg"][:2] + (s["grip"][:2] - s["nut"][:2])

    def __call__(self, step, obs_raw):
        s = self._state()
        action = np.zeros(7, dtype=np.float32)
        action[6] = _GRIPPER_OPEN
        self._t += 1
        timed_out = self._t >= _TIMEOUT[self.phase]

        if self.phase == "release":  # 뭘 쥐고 있든 놓고 떨어진 게 안정될 때까지 대기
            if timed_out:
                self.attempts += 1
                self._go("clear")

        elif self.phase == "clear":  # 물체 위에서 벗어나 안전 높이로
            action[:3] = self._move(s["grip"], [s["grip"][0], s["grip"][1], self.carry_z])
            if s["grip"][2] > self.carry_z - 0.01 or timed_out:
                self._go("align")

        elif self.phase == "align":  # 핸들 바로 위 + yaw 정렬
            action[:3] = self._move(s["grip"], [s["handle"][0], s["handle"][1], self.carry_z])
            action[3:6], rot_err = self._rot(s)
            xy_err = np.linalg.norm(s["grip"][:2] - s["handle"][:2])
            if (xy_err < 0.005 and rot_err < 0.05) or timed_out:
                self._go("descend")

        elif self.phase == "descend":  # 핸들 높이까지 하강(정렬은 계속 유지)
            action[:3] = self._move(s["grip"], [s["handle"][0], s["handle"][1], s["handle"][2] + self.grasp_dz])
            action[3:6], _ = self._rot(s)
            # grip site는 손가락 끝보다 ~1.2cm 위라 핸들 높이까지 완전히 내려가진 못한다 —
            # 목표를 못 맞추고 타임아웃을 다 쓰는 걸 막으려고 여유를 두고 끊는다(2026-09-17).
            if s["grip"][2] - s["handle"][2] < 0.015 or timed_out:
                self._go("grasp")

        elif self.phase == "grasp":
            action[6] = _GRIPPER_CLOSE
            if timed_out:
                self._go("lift")

        elif self.phase == "lift":
            action[:3] = self._move(s["grip"], [s["grip"][0], s["grip"][1], self.carry_z])
            action[6] = _GRIPPER_CLOSE
            if s["nut"][2] > 0.95:  # peg 윗면보다 높이 떴다 = 제대로 쥐었다
                self._go("over_peg")
            elif timed_out:  # 못 쥐었거나 놓쳤다 → 처음부터 재시도
                self._go("release")

        elif self.phase == "over_peg":
            target_xy = self._drop_target_xy(s)
            action[:3] = self._move(s["grip"], [target_xy[0], target_xy[1], self.carry_z])
            action[6] = _GRIPPER_CLOSE
            if np.linalg.norm(s["nut"][:2] - s["peg"][:2]) < 0.005 or timed_out:
                self._go("insert")

        elif self.phase == "insert":  # xy는 계속 보정하면서 peg를 타고 내려간다
            target_xy = self._drop_target_xy(s)
            action[:3] = self._move(s["grip"], [target_xy[0], target_xy[1], self.insert_z])
            action[6] = _GRIPPER_CLOSE
            if s["grip"][2] < self.insert_z + 0.01 or timed_out:
                self._go("open")

        elif self.phase == "open":
            if timed_out:
                self._go("retreat")

        elif self.phase == "retreat":  # 성공 판정(r_reach<0.6)에 필요 — 충분히 물러난다
            action[:3] = self._move(s["grip"], [s["grip"][0], s["grip"][1], self.retreat_z])
            if timed_out:
                self._go("release")  # 여기까지 왔는데 성공이 아니면 통째로 재시도

        return action
