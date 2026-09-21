"""STG 예측기 평가 지표 검증 — 순수 numpy 함수라 시뮬레이터·GPU 없이 돈다.

완벽한 예측기/잡음 예측기/상수 예측기를 넣었을 때 지표가 기대대로 갈리는지만 본다
(실제 예측기 품질은 서버에서 잰다 — scripts/eval_dstg.py 상단 표 참고).
"""

import numpy as np
import pytest

from square_assembly.datasets.labels import LABEL_PREINTV, LABEL_ROLLOUT
from square_assembly.scripts.eval_dstg import auroc, best_f1, evaluate


def _episodes(lengths=(10, 8)):
    """성공 데모 몇 개 -> (label, demo, t). label은 끝까지 남은 스텝(L-1-t)."""
    label, demo, t = [], [], []
    for i, L in enumerate(lengths):
        label += list(range(L - 1, -1, -1))
        demo += [f"demo_{i}"] * L
        t += list(range(L))
    return np.array(label), np.array(demo), np.array(t)


def test_perfect_predictor_scores_perfectly():
    label, demo, t = _episodes()
    m = evaluate(label.astype(float), np.zeros(len(label)), label, demo, t, None)
    assert m["mae"] == 0.0
    assert m["reward_sign_acc"] == 1.0       # 매 스텝 r = +1
    assert m["reward_mae"] == pytest.approx(0.0)
    assert m["spearman"] == pytest.approx(1.0)
    assert m["success_f1"] == pytest.approx(1.0)
    assert np.isinf(m["reward_snr"])         # 분산 0


def test_noise_breaks_the_reward_even_when_mae_is_small():
    """핵심 실패 모드: 평균오차는 작은데 스텝 차분이 잡음이면 보상으로 못 쓴다."""
    label, demo, t = _episodes((40, 40))
    rng = np.random.default_rng(0)
    noisy = label + rng.normal(0, 3.0, len(label))   # MAE ~2.4로 작은 편
    m = evaluate(noisy, np.zeros(len(label)), label, demo, t, None)
    assert m["mae"] < 3.0
    assert m["reward_sign_acc"] < 0.75               # 부호가 자주 뒤집힌다
    assert m["reward_mae"] > 1.0
    assert m["reward_snr"] < 1.0


def test_constant_predictor_has_no_signal():
    label, demo, t = _episodes()
    m = evaluate(np.full(len(label), 5.0), np.zeros(len(label)), label, demo, t, None)
    assert m["reward_sign_acc"] == 0.0
    assert np.isnan(m["spearman"]) or m["spearman"] == 0.0


def test_auroc_direction_and_edges():
    assert auroc([3, 4, 5], [0, 1, 2]) == 1.0
    assert auroc([0, 1, 2], [3, 4, 5]) == 0.0
    assert auroc([1, 2, 3], [1, 2, 3]) == pytest.approx(0.5)
    assert np.isnan(auroc([], [1, 2]))


def test_best_f1_finds_the_separating_threshold():
    score = np.array([0.0, 0.5, 9.0, 10.0])
    out = best_f1(score, np.array([True, True, False, False]))
    assert out["success_f1"] == pytest.approx(1.0)
    assert 0.5 <= out["success_threshold"] < 9.0


def test_preintv_metrics_report_predicted_and_oracle():
    label, demo, t = _episodes((12,))
    mode = np.full(len(label), LABEL_ROLLOUT)
    mode[:4] = LABEL_PREINTV               # 에피소드 초반 = 끝까지 멀다
    m = evaluate(label.astype(float), np.zeros(len(label)), label, demo, t, mode)
    assert m["n_preintv"] == 4
    assert m["preintv_auroc"] == pytest.approx(m["preintv_auroc_oracle"])  # 완벽한 예측기
    assert m["preintv_auroc_oracle"] == 1.0


def test_trailing_mean_only_looks_backwards():
    """미래를 보면 RL에서 재현이 안 된다 — 앞쪽은 있는 만큼만 평균낸다."""
    from square_assembly.scripts.eval_dstg import trailing_mean

    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert np.allclose(trailing_mean(x, 1), x)
    assert np.allclose(trailing_mean(x, 3), [1.0, 1.5, 2.0, 3.0, 4.0])


def test_smoothing_recovers_the_reward_when_the_error_is_independent_jitter():
    """떨림만 있으면 이동평균이 살려낸다 — 상관된 드리프트에는 안 통한다는 대조군과 짝."""
    from square_assembly.scripts.eval_dstg import evaluate

    rng = np.random.default_rng(0)
    label = np.arange(300, 0, -1).astype(float)
    demo = np.array(["demo_0"] * len(label))
    t = np.arange(len(label))

    jitter = label + rng.normal(0, 8.0, len(label))
    m = evaluate(jitter, np.zeros(len(label)), label, demo, t, None)
    assert m["by_smooth"]["1"]["1"]["reward_sign_acc"] < 0.7
    assert m["by_smooth"]["20"]["1"]["reward_sign_acc"] > 0.9

    drift = label + np.cumsum(rng.normal(0, 1.0, len(label)))   # 느리게 끌려가는 오차
    md = evaluate(drift, np.zeros(len(label)), label, demo, t, None)
    assert md["by_smooth"]["20"]["1"]["reward_sign_acc"] < 0.9
