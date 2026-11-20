import numpy as np

from square_assembly.scripts.audit_render import classify, repair_steps


def _img(v, seed=None):
    if seed is None:
        return np.full((84, 84, 3), v, np.uint8)
    return np.random.default_rng(seed).integers(0, 256, (84, 84, 3), dtype=np.uint8)


def test_classify_separates_render_bugs_from_small_mismatches():
    scene = {k: _img(100 + 10 * k) for k in (-1, 0, 1)}       # 같은 카메라, 스텝마다 조금씩 다른 장면
    other = {k: _img(30 + k) for k in (-1, 0, 1)}             # 다른 카메라
    assert classify(_img(101), scene, other) == "ok"
    assert classify(_img(30), scene, other) == "other_cam"     # 다른 카메라 그림이 들어감
    assert classify(_img(90), scene, other) == "stale"         # 한 스텝 전 그림
    assert classify(_img(0, seed=1), scene, other) == "garbage"
    assert classify(_img(106), scene, other) == "mismatch"     # 장면은 맞는데 조금 다름


def test_repair_covers_bug_frames_and_stale_frames_of_the_same_event_only():
    cl = {"a": ["ok"] * 20, "w": ["ok"] * 20}
    cl["a"][10] = "garbage"
    cl["w"][12] = "other_cam"
    cl["a"][14] = "stale"     # 사건 옆 — 같은 고장
    cl["w"][0] = "stale"      # 멀리 떨어진 stale — 기록 시점 어긋남이라 두지 않는다
    cl["a"][5] = "mismatch"
    assert repair_steps(cl) == [10, 12, 14]
