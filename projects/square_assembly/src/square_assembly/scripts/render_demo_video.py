r"""저장된 hdf5 데모를 mp4로 재생한다 — 화면 오른쪽에 action_mode 띠와 STG 곡선을 붙인다.

롤아웃을 새로 돌리는 record_si_video.py와 달리 이건 **이미 수집된 에피소드**를 되감아
보는 용도다(텔레옵 데이터에서 예측기가 이상하게 군 구간을 눈으로 확인하려고 만들었다).

    python -m square_assembly.scripts.render_demo_video \
        --hdf5 data/square_demo50_mouse50.hdf5 --demos demo_71 demo_55 \
        --traces outputs/seg_all.json --out-dir outputs/demo_videos
"""

import argparse
import json
import os

import cv2
import h5py
import imageio.v2 as imageio
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from square_assembly.datasets.labels import LABEL_DEMO, LABEL_INTV, LABEL_PREINTV, LABEL_ROLLOUT

MODE_COLOR = {LABEL_DEMO: "#999999", LABEL_ROLLOUT: "#4c72b0",
              LABEL_INTV: "#2ca02c", LABEL_PREINTV: "#d62728"}
MODE_NAME = {LABEL_DEMO: "DEMO", LABEL_ROLLOUT: "ROLLOUT",
             LABEL_INTV: "INTV", LABEL_PREINTV: "PREINTV"}


def _panel(mode, d, label, width, height, dpi=100):
    """곡선 + 모드 띠를 한 번만 그려 RGB 배열로 돌려준다(프레임마다 커서만 덧그린다)."""
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes([0.10, 0.16, 0.88, 0.70])
    n = len(mode)
    x = np.arange(n)
    if label is not None:
        ax.plot(x, label, color="#000000", lw=1.2, label="label (truth)")
    if d is not None:
        ax.plot(x, d, color="#ff7f0e", lw=1.2, label="d(o) predicted")
    ax.set_xlim(0, max(n - 1, 1))
    ax.set_xlabel("frame")
    ax.set_ylabel("steps-to-go")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(alpha=0.3)
    # 모드 띠 — 축 아래 얇은 밴드
    band = fig.add_axes([0.10, 0.06, 0.88, 0.06])
    for v in (LABEL_DEMO, LABEL_ROLLOUT, LABEL_INTV, LABEL_PREINTV):
        m = mode == v
        if m.any():
            band.fill_between(x, 0, 1, where=m, color=MODE_COLOR[v], step="mid", lw=0)
    band.set_xlim(0, max(n - 1, 1))
    band.set_yticks([])
    band.set_xticks([])
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    # 프레임 t -> 픽셀 x. 축 좌표계를 그대로 쓴다.
    px = ax.transData.transform(np.c_[x, np.zeros(n)])[:, 0].astype(int)
    px = np.clip(px, 0, rgb.shape[1] - 1)
    plt.close(fig)
    return rgb, px


def render(hdf5, demos, traces, out_dir, fps, cam, scale):
    traces = json.load(open(traces))["traces"] if traces else {}
    os.makedirs(out_dir, exist_ok=True)
    with h5py.File(hdf5, "r") as f:
        for name in demos:
            g = f["data"][name]
            imgs = np.asarray(g["obs"][cam])
            mode = np.asarray(g["action_mode"])[: len(imgs)]
            tr = traces.get(name, {})
            d = np.asarray(tr["d"])[: len(imgs)] if tr else None
            label = np.asarray(tr["label"])[: len(imgs)] if tr else None
            side = imgs.shape[1] * scale
            panel, px = _panel(mode, d, label, width=int(side * 1.6) // 2 * 2, height=side)
            out = os.path.join(out_dir, f"{name}.mp4")
            with imageio.get_writer(out, fps=fps, macro_block_size=1) as w:
                for t in range(len(imgs)):
                    frame = cv2.resize(imgs[t], (side, side), interpolation=cv2.INTER_NEAREST)
                    cv2.putText(frame, f"{MODE_NAME.get(int(mode[t]), '?')} t={t}", (6, 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                tuple(int(MODE_COLOR[int(mode[t])][i:i + 2], 16)
                                      for i in (1, 3, 5)), 1, cv2.LINE_AA)
                    p = panel.copy()
                    p[:, max(px[t] - 1, 0):px[t] + 2] = (214, 39, 40)
                    w.append_data(np.concatenate([frame, p], axis=1))
            print(f"{out}  ({len(imgs)} frames)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--demos", nargs="+", required=True)
    ap.add_argument("--traces", default=None, help="seg_all.json 형식: {traces: {demo: {d,label}}}")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--cam", default="agentview_image")
    ap.add_argument("--scale", type=int, default=4)
    render(**vars(ap.parse_args()))


if __name__ == "__main__":
    main()
