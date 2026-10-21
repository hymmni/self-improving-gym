"""AbsRot6dActionEnv: DP 절대 액션(pos3 + rot6d + grip) -> 지금 손끝 기준 OSC delta. robosuite delta 목표 식의 역이어야 한다."""

import numpy as np
from scipy.spatial.transform import Rotation

from square_assembly.envs.robomimic.factory import AbsRot6dActionEnv, rot6d_to_matrix


class _Ctrl:
    output_max, output_min = np.array([0.05] * 3 + [0.5] * 3), -np.array([0.05] * 3 + [0.5] * 3)
    input_max, input_min = 1.0, -1.0
    ref_pos = np.array([0.1, -0.2, 1.0])
    ref_ori_mat = Rotation.from_euler("xyz", [3.0, 0.1, 0.7]).as_matrix()

    def update(self, force=False):
        pass


class _Env:
    def __init__(self):
        self.env = type("Raw", (), {"robots": [type("R", (), {"part_controllers": {"right": _Ctrl()}})()]})()
        self.sent = None

    def step(self, a):
        self.sent = a
        return None, 0.0, False, {}


def test_abs_action_inverts_osc_delta_goal():
    env = AbsRot6dActionEnv(_Env())
    c = _Ctrl()
    delta = np.array([0.3, -0.8, 0.1, 0.2, -0.4, 0.6])
    goal_pos = c.ref_pos + 0.05 * delta[:3]                                    # robosuite: ref + scale*delta
    goal_R = Rotation.from_rotvec(0.5 * delta[3:]).as_matrix() @ c.ref_ori_mat
    rot6d = goal_R[:2].reshape(-1)                                             # pytorch3d 규약: 앞 두 행
    np.testing.assert_allclose(rot6d_to_matrix(rot6d), goal_R, atol=1e-9)
    env.step(np.concatenate([goal_pos, rot6d, [1.0]]))
    np.testing.assert_allclose(env._env.sent, np.concatenate([delta, [1.0]]), atol=1e-6)
    np.testing.assert_allclose(env.last_action, env._env.sent, atol=1e-6)
    env.step(np.zeros(7))                                                      # 사람·0 액션은 그대로
    np.testing.assert_array_equal(env._env.sent, np.zeros(7))
