"""수집기 렌더 노이즈 복구 — 연속 노이즈면 에피소드를 버리지 않고 회복을 기다린 뒤 노이즈 직전으로 되감는다.
창·robosuite 없이 가짜 env/개입 장치로 track(render_fn)만 돈다."""

import numpy as np

from square_assembly.scripts.collect_square_scripted_intervention import make_noise_tracker

_CAM = "agentview_image"


def _obs(noisy):
    rng = np.random.default_rng(0)
    stripes = np.tile(((np.arange(84) // 4) % 2) * 0.1, (3, 84, 1))  # 거칠기 약 6 — 정상 장면 범위(2~15)
    img = rng.random((3, 84, 84)) if noisy else stripes
    return {_CAM: img.astype(np.float32)}


class _Env:
    def render(self, **kw):
        return np.zeros((8, 8, 3), np.uint8)


class _Interv:
    def __init__(self, rewind=True):
        self.rendered, self.goto = [], None
        if rewind:
            self.rewind_to = lambda t: setattr(self, "goto", t)

    def render(self, frames):
        self.rendered.append(frames)
        return True


def _run(pattern, interv):
    """pattern: 스텝별 'c'(정상)/'n'(노이즈). obs_ep는 pre_step처럼 track 뒤에 쌓인다."""
    obs_ep, waits = [], []
    track, noise = make_noise_tracker(_Env(), interv, _CAM, ["agentview", "robot0_eye_in_hand"], 8,
                                      obs_ep, lambda: waits.append(len(obs_ep)))
    results = []
    for ch in pattern:
        results.append(track(_obs(ch == "n")))
        obs_ep.append(ch)
    return results, noise, waits


def test_three_noisy_frames_wait_then_rewind_to_the_last_clean_step():
    interv = _Interv()
    results, noise, waits = _run("ccccc" + "nnn", interv)
    assert all(results)                  # 에피소드를 끊지 않는다
    assert waits == [7]                  # 세 번째 노이즈에서 회복을 기다렸고
    assert interv.goto == 4              # 노이즈 직전 정상 스텝으로
    assert noise["streak"] == 0
    assert len(interv.rendered) == 7     # 복구 대기 스텝은 화면에 노이즈를 안 그린다
    assert set(interv.rendered[0]) == {"agentview", "robot0_eye_in_hand"}


def test_short_glitches_do_not_rewind():
    interv = _Interv()
    results, noise, waits = _run("ccnnccnc", interv)
    assert all(results) and waits == [] and interv.goto is None


def test_noise_from_the_first_frame_still_drops_the_episode():
    interv = _Interv()
    results, noise, waits = _run("nnn", interv)
    assert results[-1] is False and noise["streak"] == 3 and waits == []


def test_without_rewind_support_the_episode_is_dropped_as_before():
    interv = _Interv(rewind=False)
    results, noise, waits = _run("cccnnn", interv)
    assert results[-1] is False and waits == []
