"""인수인계 찾기와 선행시간 요약 — 창·sim 없이."""

from square_assembly.scripts.mark_preintv_onset import summarize, takeovers


def test_takeovers_find_each_policy_to_human_switch_and_its_policy_segment():
    #        0  1  2  3  4  5  6  7  8  9
    modes = [0, 0, -10, 1, 1, 0, 0, 1, 1, 0]
    assert takeovers(modes) == [(0, 3), (5, 7)]
    assert takeovers([1, 1, 0, 0]) == []  # 처음부터 사람이면 넘겨받은 게 아니다


def test_summary_counts_no_mistake_separately_and_measures_lead_in_frames():
    marks = {"a@50": {"onset": 50, "mark": 30}, "b@80": {"onset": 80, "mark": 70},
             "c@90": {"onset": 90, "mark": None}}
    s = summarize(marks, fps=20)
    assert s["n_marked"] == 3 and s["n_no_mistake"] == 1
    assert s["lead_frames_min_max"] == [10, 20]
    assert s["lead_sec_p10_25_50_75_90"][2] == 0.75


def test_marked_modes_replace_the_fixed_window_with_the_marked_span_and_skip_stalls():
    from square_assembly.scripts.mark_preintv_onset import marked_modes

    #        0  1  2   3   4  5  6  7  8   9  10 11
    modes = [0, 0, 0, -10, -10, 1, 1, 0, 0, -10, 1, 1]
    out = marked_modes(modes, [{"onset": 5, "mark": 1}, {"onset": 10, "mark": None}])
    assert list(out) == [0, -10, -10, -10, -10, 1, 1, 0, 0, 0, 1, 1]
