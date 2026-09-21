r"""저장된 hdf5 데모를 mp4로 재생한다 — 오른쪽에 STG 분포·곡선·불확실성을 붙인다.

롤아웃을 새로 돌리는 record_si_video.py와 달리 이건 **이미 수집된 에피소드**를 되감아
보는 용도다(텔레옵 데이터에서 예측기가 이상하게 군 구간을 눈으로 확인하려고 만들었다).

화면 구성: [카메라 | 현재 프레임 분포 히스토그램 배경 + 참값/예측 곡선(+-1 std 띠) | action_mode 띠].
세로축은 **남은 스텝 수**다 — 예측기는 진행률x1000(label_horizon)으로 학습했지만
표시할 때 (L-1)/1000을 곱해 스텝으로 되돌린다.

주의: 수집 스크립트가 states/model_file을 저장하지 않아(무게 때문) 시뮬레이터 재현이
불가능하다. 카메라 화면은 저장된 84x84를 보간 확대한 것이고, 없던 디테일이 생기지는 않는다.

    python -m square_assembly.scripts.render_demo_video \
        --hdf5 data/square_demo50_mouse50.hdf5 --demos demo_71 \
        --traces-dir outputs/traces_s1 --out-dir outputs/demo_videos
"""

import argparse
import os

import cv2
import h5py
import imageio.v2 as imageio
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 서버 컨테이너엔 한글 폰트가 없다 — 패널 글자는 영어로 둔다.

from square_assembly.datasets.labels import LABEL_DEMO, LABEL_INTV, LABEL_PREINTV, LABEL_ROLLOUT

MODE_COLOR = {LABEL_DEMO: "#999999", LABEL_ROLLOUT: "#4c72b0",
              LABEL_INTV: "#2ca02c", LABEL_PREINTV: "#d62728"}
MODE_NAME = {LABEL_DEMO: "DEMO", LABEL_ROLLOUT: "ROLLOUT",
             LABEL_INTV: "INTV", LABEL_PREINTV: "PREINTV"}


def spread(probs, vals, mass):
    """카테고리컬의 전체 표준편차와, 확률 상위 mass만 남기고 다시 정규화한 절단 표준편차.

    꼬리(거의 0인 확률이 수백 bin에 깔린 부분)가 분산을 부풀리므로, 봉우리 쪽 질량만
    남겨 보면 "정말 헷갈리는지(봉우리가 둘)" 와 "그냥 꼬리가 두꺼운지"가 갈린다.
    """
    p = probs.astype(np.float64)
    mu = (p * vals).sum(1)
    total = np.sqrt((p * (vals - mu[:, None]) ** 2).sum(1))
    order = np.argsort(-p, axis=1)
    ps = np.take_along_axis(p, order, 1)
    keep = np.cumsum(ps, axis=1) - ps < mass          # 상위 확률부터 mass까지
    ps = np.where(keep, ps, 0.0)
    vs = np.take_along_axis(np.broadcast_to(vals, p.shape), order, 1)
    ps /= ps.sum(1, keepdims=True)
    mu_t = (ps * vs).sum(1)
    trunc = np.sqrt((ps * (vs - mu_t[:, None]) ** 2).sum(1))
    return total, trunc


def _panel(mode, d, label, probs, vals, width, height, mass, tag=None, dpi=100):
    """정적인 부분(곡선·불확실성 띠·모드 띠)을 한 번만 그린다.

    프레임마다 바뀌는 건 커서와 히스토그램뿐이라, 나머지는 여기서 픽셀로 구워두고
    _overlay_hist가 그 위에 덧그린다(프레임마다 matplotlib을 다시 돌리면 느리다).
    반환하는 geom은 그 덧그리기에 필요한 축의 픽셀 위치와 y 눈금이다.
    """
    n = len(mode)
    x = np.arange(n)
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes([0.09, 0.14, 0.88, 0.78])
    ymax = float(max(label.max(), np.nanmax(d) if d is not None else 0)) * 1.08

    if d is not None and probs is not None:
        # 불확실성은 따로 그리지 않고 예측선 위에 +-1 표준편차 띠로 겹친다.
        total, trunc = spread(probs, vals, mass)
        ax.fill_between(x, d - total, d + total, color="#ff7f0e", alpha=0.16,
                        lw=0, label="+-1 std (total)")
        ax.fill_between(x, d - trunc, d + trunc, color="#ff7f0e", alpha=0.28,
                        lw=0, label=f"+-1 std (top {mass:.0%} mass)")
    ax.plot(x, label, color="#000000", lw=1.4, label="truth (steps to go)")
    if d is not None:
        ax.plot(x, d, color="#d95f02", lw=1.5, label="predicted d(o)")
    ax.set_xlim(0, max(n - 1, 1))
    ax.set_ylim(0, ymax)
    ax.set_ylabel("steps to go")
    ax.set_xlabel("frame")
    if tag:
        ax.set_title(tag, fontsize=10, pad=4)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)

    band = fig.add_axes([0.09, 0.04, 0.88, 0.055])
    for v in (LABEL_DEMO, LABEL_ROLLOUT, LABEL_INTV, LABEL_PREINTV):
        m = mode == v
        if m.any():
            band.fill_between(x, 0, 1, where=m, color=MODE_COLOR[v], step="mid", lw=0)
    band.set_xlim(0, max(n - 1, 1))
    band.set_yticks([]); band.set_xticks([])

    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    px = np.clip(ax.transData.transform(np.c_[x, np.zeros(n)])[:, 0].astype(int),
                 0, rgb.shape[1] - 1)
    bb = ax.get_window_extent()
    h = rgb.shape[0]
    rows = np.arange(int(h - bb.y1) + 1, int(h - bb.y0))          # 축 안쪽 이미지 행
    y_of_row = ax.transData.inverted().transform(
        np.c_[np.full(len(rows), bb.x0), h - rows])[:, 1]          # 각 행이 가리키는 스텝 값
    geom = {"x0": int(bb.x0) + 1, "x1": int(bb.x1), "rows": rows, "y": y_of_row,
            # 빈 배경(흰색)에만 칠한다 — 곡선과 눈금선은 히스토그램에 안 덮이게.
            "blank": (rgb > 242).all(-1)}
    plt.close(fig)
    return rgb, px, geom


def _overlay_hist(frame, probs_t, vals, geom, nbars=110, frac=0.40, color=(70, 120, 190)):
    """그 프레임의 카테고리컬을 가로 막대(=히스토그램)로 배경에 깔아준다.

    bin 1000여 개를 화면 막대 nbars개로 모아 담는다 — bin 단위로 그대로 그리면 이웃
    bin끼리 들쭉날쭉해 줄무늬만 보인다. 길이는 그 프레임 안에서만 정규화하므로,
    확신이 셀수록 좁고 뾰족한 덩어리, 헷갈릴수록 넓고 퍼진 덩어리로 매 프레임 달라진다.
    """
    rows = geom["rows"]
    y_top, y_bot = geom["y"][0], geom["y"][-1]
    if y_top == y_bot:
        return frame
    u = (y_top - vals) / (y_top - y_bot)                      # 0(위) ~ 1(아래)
    ok = (u >= 0) & (u < 1)
    acc = np.bincount((u[ok] * nbars).astype(int), weights=probs_t[ok], minlength=nbars)
    if acc.max() <= 0:
        return frame
    span = geom["x1"] - geom["x0"]
    ln = (acc / acc.max() * span * frac).astype(int)
    per_row = ln[np.minimum((np.arange(len(rows)) / len(rows) * nbars).astype(int), nbars - 1)]
    bar = np.arange(span)[None, :] < per_row[:, None]
    sub = frame[rows, geom["x0"]:geom["x1"]]
    sub[bar & geom["blank"][rows, geom["x0"]:geom["x1"]]] = color
    frame[rows, geom["x0"]:geom["x1"]] = sub
    return frame


def render(hdf5, demos, traces_dir, out_dir, fps, cam, wrist_cam, size, mass, tag, suffix):
    os.makedirs(out_dir, exist_ok=True)
    with h5py.File(hdf5, "r") as f:
        for name in demos:
            g = f["data"][name]
            imgs = np.asarray(g["obs"][cam])
            wrist = np.asarray(g["obs"][wrist_cam]) if wrist_cam in g["obs"] else None
            mode = np.asarray(g["action_mode"])[: len(imgs)]
            n = len(imgs)
            label = (np.arange(n - 1, -1, -1)).astype(np.float64)   # 남은 스텝 = L-1-t
            d = probs = vals = None
            tp = os.path.join(traces_dir, f"{name}.npz") if traces_dir else None
            if tp and os.path.exists(tp):
                z = np.load(tp)
                # 학습 단위(진행률 x label_horizon)를 스텝으로 되돌린다.
                scale = (n - 1) / max(float(z["label"][0]), 1.0)
                d = np.asarray(z["d"], dtype=np.float64)[:n] * scale
                probs = np.asarray(z["probs"], dtype=np.float32)[:n]
                vals = np.arange(probs.shape[1], dtype=np.float64) * scale

            panel, px, geom = _panel(mode, d, label, probs, vals,
                                      width=int(size * 1.8) // 2 * 2, height=size,
                                      mass=mass, tag=tag)
            out = os.path.join(out_dir, f"{name}{suffix}.mp4")
            with imageio.get_writer(out, fps=fps, macro_block_size=1) as w:
                for t in range(n):
                    frame = cv2.resize(imgs[t], (size, size), interpolation=cv2.INTER_LANCZOS4)
                    if wrist is not None:
                        s = size // 4
                        pip = cv2.resize(wrist[t], (s, s), interpolation=cv2.INTER_LANCZOS4)
                        frame[size - s - 6:size - 6, size - s - 6:size - 6] = pip
                        cv2.rectangle(frame, (size - s - 6, size - s - 6),
                                      (size - 7, size - 7), (255, 255, 255), 1)
                    col = tuple(int(MODE_COLOR[int(mode[t])][i:i + 2], 16) for i in (1, 3, 5))
                    cv2.putText(frame, f"{MODE_NAME.get(int(mode[t]), '?')}  t={t}/{n - 1}",
                                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
                    p = panel.copy()
                    if probs is not None:
                        _overlay_hist(p, probs[t].astype(np.float64), vals, geom)
                    p[:, max(px[t] - 1, 0):px[t] + 2] = (214, 39, 40)
                    w.append_data(np.concatenate([frame, p], axis=1))
            print(f"{out}  ({n} frames)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--demos", nargs="+", required=True)
    ap.add_argument("--traces-dir", default=None, help="<demo>.npz(probs,d,label,mode)가 든 폴더")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--cam", default="agentview_image")
    ap.add_argument("--wrist-cam", default="robot0_eye_in_hand_image")
    ap.add_argument("--size", type=int, default=512, help="카메라 화면 한 변(보간 확대)")
    ap.add_argument("--mass", type=float, default=0.9, help="절단 분산이 남길 확률 질량")
    ap.add_argument("--tag", default=None, help="패널 제목(예측기 이름 등)")
    ap.add_argument("--suffix", default="", help="출력 파일명 꼬리표")
    render(**vars(ap.parse_args()))


if __name__ == "__main__":
    main()
