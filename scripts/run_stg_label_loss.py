"""Run v3 VIP comparisons without overwriting existing results.

Use the same train seed (1), split seeds (0..4), head and bin count as existing A.
E adds modality dropout to A; F trains on main-camera features only.
Rerunning resumes completed checkpoints; selected arms are evaluated again.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def training_provenance(root, cache, source):
    """A same-path data/code change must never silently reuse a trained checkpoint."""
    import torch

    base = root / "projects/square_assembly/src/square_assembly"
    paths = [root / cache, root / source] + [base / p for p in [
        "scripts/train_dstg_vip.py", "configs/train_dstg_vip.yaml", "datasets/stg_labels.py", "datasets/labels.py",
        "datasets/dino_feature_dataset.py", "policies/diffusion/dino_stg_predictor.py"]]
    digests = {}
    for path in paths:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        digests[str(path)] = digest.hexdigest()
    return {"sha256": digests, "python": sys.version, "torch": torch.__version__}


def run(command, log, root, env):
    print("RUN", " ".join(map(str, command)), flush=True)
    with log.open("w") as stream:
        subprocess.run(command, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT,
                       check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default="data/square_demo50_mouse50v3.vip.h5")
    parser.add_argument("--source", default="data/square_demo50_mouse50v3m.hdf5")
    parser.add_argument("--baseline", default="outputs/pre_v3/norollout_anchor13")
    parser.add_argument("--out", default="outputs/stg_label_loss_20260928")
    parser.add_argument("--arms", nargs="+", choices=list("ABCDEF"), default=list("ABCD"))
    args = parser.parse_args()
    if "A" not in args.arms or len(args.arms) != len(set(args.arms)):
        parser.error("include A and specify each arm once")
    root = Path(__file__).resolve().parents[1]
    out = root / args.out
    requested = set(args.arms)
    study_path = out / "study_arms.json"
    existing = {a for a in "ABCDEF" if (out / a).is_dir()}
    if study_path.exists():
        existing = set(json.loads(study_path.read_text()))
        if existing != requested:
            raise ValueError("different arm set: use a separate --out directory")
    elif (out / "summary.json").exists():
        existing = set(json.loads((out / "summary.json").read_text())["rows"])
    if existing - requested:
        raise ValueError("different arm set: use a separate --out directory")
    out.mkdir(parents=True, exist_ok=True)
    study_path.write_text(json.dumps(sorted(requested)))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "projects/square_assembly/src")
    provenance = training_provenance(root, args.cache, args.source)
    rows = {}
    settings = {"A": (0, 0), "B": (0, 0), "C": (2, 0), "D": (2, 1), "E": (0, 0), "F": (0, 0)}
    for arm in args.arms:
        sigma, weight = settings[arm]
        rows[arm] = []
        for seed in range(5):
            folder = out / arm / f"split{seed}"
            folder.mkdir(parents=True, exist_ok=True)
            ckpt = (root / args.baseline / f"vip_s{seed}/predictor.pt" if arm == "A"
                    else folder / "predictor.pt")
            if arm != "A":
                command = [sys.executable, "-m", "square_assembly.scripts.train_dstg_vip",
                           f"cache_path={args.cache}", f"mode_hdf5={args.source}",
                           f"label_horizon={'1000' if arm in ('E', 'F') else 'null'}", "num_bins_override=1400",
                           "train_modes=[demo,intv,preintv]", "preintv=anchor", "preintv_len=13",
                           "preintv_weight=1", f"obs_parts={'agentview' if arm == 'F' else 'all'}", "obs_horizon=2",
                           "seed=1", f"split_seed={seed}", "num_epochs=30",
                           f"gauss_sigma={sigma}", f"pair_weight={weight}", "pair_stride=8",
                           f"modality_dropout={0.2 if arm == 'E' else 0.0}",
                           f"out={ckpt}", f"hydra.run.dir={folder}/hydra"]
                command_path = folder / "train_manifest.json"
                manifest = {"command": command, "provenance": provenance}
                if ckpt.exists():
                    if not command_path.exists() or json.loads(command_path.read_text()) != manifest:
                        raise ValueError(f"existing checkpoint has a different run command: {ckpt}")
                else:
                    command_path.write_text(json.dumps(manifest, indent=2))
                    run(command, folder / "train.log", root, env)
            run([sys.executable, "-m", "square_assembly.scripts.eval_dstg",
                 "--predictor", str(ckpt), "--cache", args.cache, "--source-hdf5", args.source,
                 "--split-seed", str(seed), "--out", str(folder / "metrics.json")],
                folder / "eval.log", root, env)
            m = json.loads((folder / "metrics.json").read_text())
            step = m["step_units"]
            rows[arm].append({
                "split_seed": seed, "k8": m["by_stride"]["8"]["reward_sign_acc"],
                "null_k8": m["null_time_only"]["by_smooth"]["1"]["8"],
                "mae_steps": step["mae_steps"],
                "closed_mae_steps": step["closed_nonterminal"]["mae_steps"],
                "closed_bias_steps": step["closed_nonterminal"]["bias_steps"],
                "closed_n": step["closed_nonterminal"]["n"],
                "pre_closed_mae_steps": step["closed_by_mode"]["preintv"]["mae_steps"],
                "pre_closed_n": step["closed_by_mode"]["preintv"]["n"],
                "pre_reward_steps_per_frame": m["by_segment"]["preintv"]["reward_steps_per_frame"],
                "pre_positive_fraction": m["by_segment"]["preintv"]["sign_acc"],
                "success_f1": m["success_f1"],
            })
            print("RESULT", arm, json.dumps(rows[arm][-1]), flush=True)
        (out / "progress.json").write_text(json.dumps(rows, indent=2))
    fields = [k for k in rows["A"][0] if k not in ("split_seed", "closed_n", "pre_closed_n")]
    summary = {"rows": rows, "mean_std": {}, "paired_deltas": {}}
    for arm, values in rows.items():
        summary["mean_std"][arm] = {
            k: {"mean": float(np.mean([v[k] for v in values if v[k] is not None])),
                "std": float(np.std([v[k] for v in values if v[k] is not None]))}
            for k in fields if any(v[k] is not None for v in values)}
    for arm, reference in [("B", "A"), ("C", "B"), ("D", "C"), ("D", "A"), ("E", "A"), ("F", "A")]:
        if arm not in rows or reference not in rows:
            continue
        summary["paired_deltas"][f"{arm}-{reference}"] = {}
        for k in fields:
            delta = [v[k] - r[k] for v, r in zip(rows[arm], rows[reference])
                     if v[k] is not None and r[k] is not None]
            if delta:
                summary["paired_deltas"][f"{arm}-{reference}"][k] = {
                    "mean": float(np.mean(delta)), "std": float(np.std(delta))}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("SUMMARY", json.dumps(summary["mean_std"]), flush=True)


if __name__ == "__main__":
    main()
