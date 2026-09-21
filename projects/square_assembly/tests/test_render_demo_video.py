import numpy as np

from square_assembly.scripts.render_demo_video import spread


def test_spread_drops_the_tail_but_keeps_two_peaks():
    vals = np.arange(100, dtype=np.float64)
    # 봉우리 둘(10, 90) + 전 구간에 깔린 얇은 꼬리
    p = np.full(100, 0.001)
    p[10] = p[90] = 0.45
    p /= p.sum()
    total, trunc = spread(p[None], vals, 0.9)
    assert total[0] > 30          # 봉우리가 둘이라 원래 크다
    assert trunc[0] > 30          # 꼬리를 잘라도 봉우리 둘은 남는다 — 진짜 애매함은 안 감춘다


def test_spread_is_zero_for_a_delta():
    vals = np.arange(10, dtype=np.float64)
    p = np.zeros(10); p[3] = 1.0
    total, trunc = spread(p[None], vals, 0.9)
    assert total[0] == 0 and trunc[0] == 0


def test_truncation_shrinks_a_fat_tail():
    vals = np.arange(200, dtype=np.float64)
    p = np.full(200, 0.5 / 199)
    p[100] = 0.5
    assert spread(p[None], vals, 0.9)[1][0] < spread(p[None], vals, 0.9)[0][0]
    total, trunc = spread(p[None], vals, 0.5)
    assert trunc[0] < total[0] / 2
