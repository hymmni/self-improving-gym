"""Square 성공 판정 = robosuite on_peg(너트 중심이 기둥 xy ±3cm, 높이 낮음) + 너트 기울기 ≤ 30°.

robosuite만으로는 기둥 옆에 세워진 너트(기울기 90°)도 성공이다 — turn30 branch_t0 demo_33(2026-11-22).
"""

import numpy as np
import pytest

pytest.importorskip("robosuite")


def _place_nut(env, dxy, quat_wxyz):
    sim, nut = env.env.sim, env.env.nuts[0]
    peg = sim.data.body_xpos[sim.model.body_name2id("peg1")]
    start, _ = sim.model.get_joint_qpos_addr(nut.joints[0])
    sim.data.qpos[start:start + 3] = [peg[0] + dxy, peg[1], env.env.table_offset[2] + 0.02]
    sim.data.qpos[start + 3:start + 7] = quat_wxyz
    sim.forward()


def test_square_success_requires_flat_nut():
    from square_assembly.envs.robomimic.factory import MAX_NUT_TILT_DEG, make_lowdim_env, nut_tilt_deg

    env = make_lowdim_env("NutAssemblySquare", "Panda", ["robot0_eef_pos"])
    env.reset()

    _place_nut(env, 0.0, [1, 0, 0, 0])                                   # 기둥에 누워서 끼워짐
    assert nut_tilt_deg(env.env) < 1 and env.is_success()["task"]

    c = np.cos(np.pi / 4)
    _place_nut(env, 0.02, [c, c, 0, 0])                                  # x축으로 90° 세워 기둥 옆(±3cm 안)
    assert abs(nut_tilt_deg(env.env) - 90) < 1 and not env.is_success()["task"]

    t = np.radians(MAX_NUT_TILT_DEG - 5) / 2
    _place_nut(env, 0.0, [np.cos(t), np.sin(t), 0, 0])                   # 25° 기운 것은 여전히 성공
    assert env.is_success()["task"]
