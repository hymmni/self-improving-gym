r"""텔레옵 데이터의 인수인계(정책 -> 사람) 직전을 다시 재생하며, 정책이 잘못되기 시작한 프레임을 사람이 찍는다.

PREINTV 창 길이(수집 때 15프레임 고정, rise 실험 때 20)에는 근거가 없었다. 인수인계마다 "여기서부터
틀어졌다"를 찍어 두면 선행시간(인수인계 - 표시) 분포가 나오고, 창 길이를 그걸로 정한다.

화면은 수집 때 보던 것과 같다 — 왼쪽 2D 맵(mouse_teleop의 _draw_map), 오른쪽 agentview. 저장된
sim 상태(states)로 다시 렌더하므로 84픽셀 obs가 아니라 480 해상도다. 아래 띠는 정책(파랑)/사람(초록),
흰 선 = 인수인계, 빨간 선 = 찍은 곳, 노란 선 = 지금.

키: Space 재생/정지, <-/-> 1프레임, Shift+<-/-> 10프레임, Enter(또는 m) 여기 표시하고 다음으로,
    x 뚜렷한 실수 없음(미리 잡은 경우) 표시하고 다음으로, n/p 다음/이전 인수인계, 띠 클릭 = 이동,
    q 저장하고 종료. 표시할 때마다 바로 저장하고, 다시 켜면 아직 안 찍은 첫 인수인계부터 연다.

    # 컨테이너 안, 수집기와 같은 화면(DISPLAY=:1)
    python -m square_assembly.scripts.mark_preintv_onset --hdf5 data/square_mouse_intv_r0v3.hdf5
    # 찍은 결과 요약만
    python -m square_assembly.scripts.mark_preintv_onset --hdf5 data/square_mouse_intv_r0v3.hdf5 --summary
"""

import argparse
import json
import os

import numpy as np

from square_assembly.datasets.labels import LABEL_INTV

FPS = 20          # 수집 control_fps — 초 단위 환산과 재생 속도
AFTER = 40        # 인수인계 뒤로 보여줄 프레임(사람이 무엇을 고쳤는지 봐야 무엇이 틀렸는지 안다)
BAR_H = 44


def takeovers(modes):
    """[(구간 시작, 인수인계)] — 인수인계 = 사람이 잡은 첫 프레임, 구간 시작 = 그 앞 정책 구간의 첫 프레임."""
    human = np.asarray(modes) == LABEL_INTV
    out = []
    for onset in np.flatnonzero(human & ~np.roll(human, 1)):
        if onset == 0:
            continue
        start = onset
        while start > 0 and not human[start - 1]:
            start -= 1
        out.append((int(start), int(onset)))
    return out


def summarize(marks, fps=FPS):
    """찍은 결과 -> 선행시간(인수인계 - 표시, 프레임) 분포. 'x'(실수 없음)는 따로 센다."""
    done = [m for m in marks.values() if "mark" in m]
    leads = np.array([m["onset"] - m["mark"] for m in done if m["mark"] is not None])
    out = {"n_marked": len(done), "n_no_mistake": sum(m["mark"] is None for m in done)}
    if len(leads):
        pct = np.percentile(leads, [10, 25, 50, 75, 90])
        out.update({"lead_frames_p10_25_50_75_90": pct.round(1).tolist(),
                    "lead_sec_p10_25_50_75_90": (pct / fps).round(2).tolist(),
                    "lead_frames_min_max": [int(leads.min()), int(leads.max())]})
    return out


def _load_marks(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)["marks"]
    return {}


def _save_marks(path, hdf5, marks):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"hdf5": hdf5, "fps": FPS, "marks": marks}, f, indent=1)
    os.replace(tmp, path)  # 쓰다 죽어도 이전 표시가 날아가지 않게


class Replayer:
    """저장된 states로 sim을 되돌려 [맵 | agentview] 프레임을 만든다."""

    def __init__(self, base_ckpt, size=480):
        from square_assembly.runners.mouse_teleop import MouseTeleopIntervention
        from square_assembly.scripts.collect_square_scripted_intervention import wait_for_renderer
        from square_assembly.utils.checkpoints import load_run_config
        from square_assembly.utils.task_utils import make_eval_env

        task_cfg = load_run_config(base_ckpt).task
        self.env = wait_for_renderer(lambda: make_eval_env(task_cfg), "agentview_image", size)
        self.view = MouseTeleopIntervention(self.env, map_size=size)  # 창은 안 연다 — _draw_map만 빌린다
        self.size = size

    def frames(self, demo, t0, t1):
        """demo(h5py 그룹)의 [t0, t1] 프레임들. 첫 프레임에서 되돌린 eef가 저장된 obs와 맞는지 확인한다."""
        states, acts = demo["states"], demo["actions"]
        eef, grip = demo["obs"]["robot0_eef_pos"], demo["obs"]["robot0_gripper_qpos"]
        out = []
        for t in range(t0, t1 + 1):
            obs = self.env.reset_to({"states": np.asarray(states[t])})
            if t == t0:
                err = float(np.abs(np.asarray(obs["robot0_eef_pos"]) - eef[t]).max())
                if err > 1e-3:
                    print(f"  경고: 되돌린 eef가 저장값과 {err * 100:.1f}cm 다르다 — 재생이 수집 장면과 다를 수 있다",
                          flush=True)
            self.view._last_obs = {"robot0_gripper_qpos": grip[t]}
            self.view.controller.grip_cmd = float(acts[t][6])
            cam = self.env.render(mode="rgb_array", height=self.size, width=self.size, camera_name="agentview")
            out.append(np.concatenate([self.view._draw_map()[:, :, ::-1], cam], axis=1))  # RGB
        return out


def compose(frame, t, t0, onset, human, mark, header):
    """frame(RGB) 아래에 띠를 붙이고 글자를 얹는다. human: 창 안 프레임별 사람 제어 여부."""
    import cv2

    h, w = frame.shape[:2]
    img = np.zeros((h + BAR_H, w, 3), dtype=np.uint8)
    img[:h] = frame
    n = len(human)
    x = lambda tt: int(round((tt - t0) / max(n - 1, 1) * (w - 1)))
    for i, hum in enumerate(human):
        img[h + 8:h + BAR_H - 8, x(t0 + i):x(t0 + i + 1) + 1] = (60, 170, 60) if hum else (70, 100, 190)
    img[h + 2:h + BAR_H - 2, max(x(onset) - 1, 0):x(onset) + 2] = (255, 255, 255)
    if mark is not None:
        img[h + 2:h + BAR_H - 2, max(x(mark) - 1, 0):x(mark) + 2] = (255, 40, 40)
    img[h:h + BAR_H, max(x(t) - 1, 0):x(t) + 2] = (255, 230, 0)

    d = t - onset
    who = "HUMAN" if human[t - t0] else "policy"
    lines = [header,
             f"t={t}  {who}  {d:+d}f ({d / FPS:+.2f}s) from takeover   mark: "
             + ("-" if mark is None else f"{mark} ({mark - onset:+d}f)")]
    img[:52] //= 3        # 글자 뒤를 어둡게 — 외곽선을 굵게 그리면 글자 폭이 달라져 겹쳐 보인다
    img[h - 26:h] //= 3
    for i, s in enumerate(lines):
        cv2.putText(img, s, (8, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    help_ = "[Space]play [<-/->]1f [Shift]10f [Enter]mark [x]no mistake [n/p]next/prev [q]quit"
    cv2.putText(img, help_, (8, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return img


def run(hdf5, out, base_ckpt):
    import tkinter as tk

    import h5py
    from PIL import Image, ImageTk

    marks = _load_marks(out)
    f = h5py.File(hdf5, "r")
    names = sorted(f["data"], key=lambda k: int(k.split("_")[-1]))
    items = []  # (key, demo, 구간 시작, 인수인계, 데모 안 순번, 데모 안 개수)
    for n in names:
        tos = takeovers(f["data"][n]["action_mode"][()])
        for k, (s, o) in enumerate(tos):
            items.append((f"{n}@{o}", n, s, o, k + 1, len(tos)))
    if not items:
        raise SystemExit(f"{hdf5}에 인수인계가 없다")
    print(f"인수인계 {len(items)}개, 이미 찍은 것 {sum(k in marks for k, *_ in items)}개 -> {out}", flush=True)

    rep = Replayer(base_ckpt)
    root = tk.Tk()
    root.tk.call("tk", "useinputmethods", False)  # ibus-hangul이 Shift+키를 가져가지 않게(mouse_teleop 참고)
    root.title("preintv onset marker")
    label = tk.Label(root, bd=0, highlightthickness=0)
    label.pack()
    st = {"i": next((j for j, it in enumerate(items) if it[0] not in marks), 0), "t": 0, "play": False,
          "frames": None, "photo": None, "msg": ""}

    def show():
        key, demo, s, o, k, kn = items[st["i"]]
        e = st["t"] - s
        human = st["human"]
        mark = marks.get(key, {}).get("mark")
        n_done = sum(it[0] in marks for it in items)
        head = f"{demo} takeover {k}/{kn}   [{st['i'] + 1}/{len(items)}]   done {n_done}/{len(items)}"
        if key in marks and mark is None:
            head += "   (marked: no mistake)"
        if st["msg"]:
            head += "   ! " + st["msg"]
        img = Image.fromarray(compose(st["frames"][e], st["t"], s, o, human, mark, head))
        if st["photo"] is None:
            st["photo"] = ImageTk.PhotoImage(img)
            label.configure(image=st["photo"])
        else:
            st["photo"].paste(img)

    def load(i):
        st["i"], st["play"] = i % len(items), False
        key, demo, s, o, *_ = items[st["i"]]
        g = f["data"][demo]
        t1 = min(o + AFTER, len(g["actions"]) - 1)
        root.title(f"preintv onset marker — rendering {demo} {s}..{t1}")
        root.update()
        st["frames"] = rep.frames(g, s, t1)
        st["human"] = g["action_mode"][s:t1 + 1] == LABEL_INTV
        st["t"], st["play"] = s, True  # 구간 처음부터 재생하며 연다
        root.title("preintv onset marker")
        show()

    def seek(t):
        _, _, s, *_ = items[st["i"]]
        st["t"], st["msg"] = int(np.clip(t, s, s + len(st["frames"]) - 1)), ""
        show()

    def record(mark):
        key, demo, s, o, *_ = items[st["i"]]
        marks[key] = {"demo": demo, "seg_start": s, "onset": o, "mark": mark}
        _save_marks(out, hdf5, marks)
        load(st["i"] + 1)

    def on_key(e):
        shift = e.state & 0x1
        k = e.keysym if len(e.keysym) > 1 else e.keysym.lower()
        _, _, s, o, *_ = items[st["i"]]
        if k == "space":
            if not st["play"] and st["t"] >= s + len(st["frames"]) - 1:
                seek(s)  # 끝에서 누르면 처음부터 다시
            st["play"] = not st["play"]
        elif k in ("Left", "Right"):
            st["play"] = False
            seek(st["t"] + (10 if shift else 1) * (1 if k == "Right" else -1))
        elif k in ("Return", "KP_Enter", "m"):
            if st["t"] < o:
                record(st["t"])
            else:  # 사람이 잡은 뒤는 선행시간이 음수라 의미가 없다
                st["play"], st["msg"] = False, "mark must be before the takeover (white line)"
                show()
        elif k == "x":
            record(None)
        elif k in ("n", "p"):
            load(st["i"] + (1 if k == "n" else -1))
        elif k == "q":
            root.destroy()
        return "break"

    def on_click(e):
        if e.y >= rep.size:  # 띠 위
            _, _, s, *_ = items[st["i"]]
            st["play"] = False
            seek(s + e.x / (2 * rep.size - 1) * (len(st["frames"]) - 1))

    def tick():
        if st["play"]:
            _, _, s, *_ = items[st["i"]]
            if st["t"] >= s + len(st["frames"]) - 1:
                st["play"] = False
            else:
                seek(st["t"] + 1)
        root.after(int(1000 / FPS), tick)

    root.bind("<KeyPress>", on_key)
    label.bind("<Button-1>", on_click)
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    load(st["i"])
    root.focus_force()
    tick()
    root.mainloop()
    f.close()
    print(json.dumps(summarize(marks), ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--hdf5", required=True, help="수집 원본(states가 있어야 한다)")
    ap.add_argument("--out", default=None, help="표시 저장 json(기본: <hdf5>.preintv_marks.json)")
    ap.add_argument("--base-ckpt",
                    default="projects/square_assembly/checkpoints/square_base_policy/policy_epoch1060.pt",
                    help="env 설정을 읽어올 정책 체크포인트(수집 때와 같은 것)")
    ap.add_argument("--summary", action="store_true", help="창 없이 찍은 결과 요약만")
    args = ap.parse_args()
    out = args.out or os.path.splitext(args.hdf5)[0] + ".preintv_marks.json"
    if args.summary:
        print(json.dumps(summarize(_load_marks(out)), ensure_ascii=False, indent=1))
    else:
        run(args.hdf5, out, args.base_ckpt)


if __name__ == "__main__":
    main()
