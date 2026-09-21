r"""저장된 hdf5 데모를 mp4로 재생한다 — 오른쪽에 STG 분포·곡선·불확실성을 붙인다.

롤아웃을 새로 돌리는 record_si_video.py와 달리 이건 **이미 수집된 에피소드**를 되감아
보는 용도다(텔레옵 데이터에서 예측기가 이상하게 군 구간을 눈으로 확인하려고 만들었다).

화면 구성: [카메라 | 카테고리컬 분포 배경 + 참값/예측 곡선 | 불확실성 | action_mode 띠].
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
from matplotlib.colors import PowerNorm

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
    """곡선·분포·불확실성을 한 번만 그려 RGB 배열로 돌려준다(프레임마다 커서만 덧그린다)."""
    n = len(mode)
    x = np.arange(n)
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes([0.09, 0.42, 0.88, 0.50])
    ymax = float(max(label.max(), np.nanmax(d) if d is not None else 0)) * 1.08

    if probs is not None:
        # 프레임마다 최대값으로 정규화 — 분포의 '모양'이 보이게. 색이 곧 상대 확률.
        col = probs.T / np.maximum(probs.max(1), 1e-8)
        ax.imshow(col, origin="lower", aspect="auto", cmap="Blues",
                  norm=PowerNorm(0.45), extent=(0, max(n - 1, 1), vals[0], vals[-1]))
    ax.plot(x, label, color="#000000", lw=1.4, label="truth (steps to go)")
    if d is not None:
        ax.plot(x, d, color="#ff7f0e", lw=1.4, label="predicted d(o)")
    ax.set_xlim(0, max(n - 1, 1))
    ax.set_ylim(0, ymax)
    ax.set_ylabel("steps to go")
    ax.set_xticklabels([])
    if tag:
        ax.set_title(tag, fontsize=10, pad=4)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)

    av = fig.add_axes([0.09, 0.17, 0.88, 0.22])
    if probs is not None:
        total, trunc = spread(probs, vals, mass)
        av.plot(x, total, color="#8c564b", lw=1.2, label="total")
        av.plot(x, trunc, color="#17becf", lw=1.2, label=f"truncated (top {mass:.0%} mass)")
        av.legend(loc="upper right", fontsize=7, ncol=2)
    av.set_xlim(0, max(n - 1, 1))
    av.set_ylabel("std (steps)")
    av.set_xlabel("frame")
    av.grid(alpha=0.25)

    band = fig.add_axes([0.09, 0.06, 0.88, 0.06])
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
    plt.close(fig)
    return rgb, px


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

            panel, px = _panel(mode, d, label, probs, vals,
                               width=int(size * 1.8) // 2 * 2, height=size, mass=mass, tag=tag)
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
