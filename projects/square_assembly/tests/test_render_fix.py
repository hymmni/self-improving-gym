"""robosuite 렌더 결함(render_fix.py) — GC가 바꿔 끼운 옛 sim을 치워도 지금 env의 렌더가 그대로여야 한다. EGL이 필요하다."""
import gc
import os

import numpy as np
import pytest

suite = pytest.importorskip("robosuite")
pytestmark = pytest.mark.skipif(os.environ.get("MUJOCO_GL") != "egl", reason="EGL 렌더가 있어야 한다")


def _render(env):
    return env.sim.render(84, 84, camera_name="agentview").astype(int)


def test_collecting_a_replaced_sim_keeps_the_live_render():
    from square_assembly.envs.robomimic.render_fix import install

    install()
    env = suite.make("NutAssemblySquare", robots="Panda", has_renderer=False, has_offscreen_renderer=True,
                     use_camera_obs=True, camera_names=["agentview"], camera_heights=84, camera_widths=84)
    was = gc.isenabled()
    gc.disable()   # 옛 sim은 우리가 부르는 collect에서만 치워지게
    try:
        env.reset()
        env.reset()          # 첫 sim이 렌더 컨텍스트와 순환참조인 채 버려진다
        state = env.sim.get_state().flatten()
        before = _render(env)
        assert gc.collect() > 0
        env.sim.set_state_from_flattened(state)
        env.sim.forward()
        after = _render(env)
    finally:
        if was:
            gc.enable()
    rough = np.abs(np.diff(after, axis=1)).mean()
    assert 2.0 < rough < 15.0                       # 고치기 전: 단색(0.1) 또는 노이즈(50+)
    assert np.abs(after - before).mean() < 0.5      # 같은 상태는 같은 그림
