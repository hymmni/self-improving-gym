"""Soft-label boundary mass and episode-safe pair supervision."""

import numpy as np
import pytest
import torch

from square_assembly.scripts import train_dstg_vip as vip


def test_gaussian_targets_integrate_bins_and_normalize_at_boundaries():
    p = vip._gaussian_targets(torch.tensor([0, 2, 4]), 5, 1.0)
    torch.testing.assert_close(p.sum(-1), torch.ones(3))
    assert torch.isfinite(p).all() and (p >= 0).all()
    torch.testing.assert_close(
        p[1], torch.tensor([0.06136, 0.24477, 0.38774, 0.24477, 0.06136]),
        atol=3e-5, rtol=0,
    )
    torch.testing.assert_close(p[0], p[2].flip(0))
    for sigma in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            vip._gaussian_targets(torch.tensor([2]), 5, sigma)


def test_pairs_keep_classification_samples_and_exclude_filtered_endpoints():
    class Windows(torch.utils.data.Dataset):
        samples = [("a", t) for t in range(5)] + [("b", t) for t in range(3)]

        def __getitem__(self, i):
            return torch.tensor([float(i)]), self.samples[i][1]

    ds = vip._PairedWindows(Windows(), [0, 1, 3, 4, 5, 6], stride=2)
    assert len(ds) == 6
    assert not ds[0][-1]  # a:2 was filtered out
    assert ds[1][-1] and ds[1][3] == 3  # a:1 -> a:3, labels increase
    assert not ds[3][-1]  # a:4 cannot cross into episode b
    assert not ds[4][-1]  # b:2 was filtered out


def test_pair_loss_respects_reverse_progress_and_ignores_invalid_pairs():
    before = torch.tensor([[0., -1000.], [-1000., 0.]], requires_grad=True)
    after = before.flip(1)
    y, yp = torch.tensor([0, 0]), torch.tensor([1, 1])
    loss = vip._pair_loss(before, after, y, yp, torch.tensor([True, False]), 1)
    assert loss.item() == pytest.approx(0.0)
    wrong = vip._pair_loss(before[1:], after[1:], y[1:], yp[1:], torch.tensor([True]), 1)
    assert wrong.item() == pytest.approx(1.5)
    empty = vip._pair_loss(before, after, y, yp, torch.tensor([False, False]), 1)
    empty.backward()
    assert torch.isfinite(before.grad).all()


def test_closed_gripper_metrics_use_raw_units_and_report_empty_subsets():
    from square_assembly.scripts import eval_dstg as ev

    d, y = np.array([20., 10., 50.]), np.array([10., 10., 10.])
    m = ev.closed_gripper_metrics(d, y, np.array([[.0005, -.0005], [.02, -.02], [.0004, -.0004]]))
    assert m["n"] == 2 and m["mae_steps"] == pytest.approx(25.)
    assert m["bias_steps"] == pytest.approx(25.)
    assert ev.closed_gripper_metrics(d, y, np.full((3, 2), .02))["n"] == 0


def test_step_diagnostics_use_action_length_even_with_an_extra_observation(tmp_path):
    import h5py
    from square_assembly.scripts.eval_dstg import step_diagnostics

    path = tmp_path / "source.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("data/a/actions", data=np.zeros((4, 7)))
        f.create_dataset("data/a/obs/robot0_gripper_qpos", data=np.full((5, 2), .0005))
    t = np.arange(4)
    label = 3 - t
    arrays = (label.astype(float) * (1000 / 3), np.zeros(4), label,
              np.full(4, "a"), t, None)
    metrics = step_diagnostics(arrays, path, 1000)
    assert metrics["mae_steps"] == pytest.approx(0., abs=1e-12)
    assert metrics["closed_gripper"]["mae_steps"] == pytest.approx(0., abs=1e-12)


def test_resume_provenance_changes_when_data_changes_at_the_same_path(tmp_path):
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location("stg_runner", root / "scripts/run_stg_label_loss.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    cache, source = tmp_path / "cache", tmp_path / "source"
    cache.write_bytes(b"old cache")
    source.write_bytes(b"same source")
    before = runner.training_provenance(root, cache, source)
    cache.write_bytes(b"new cache")
    assert runner.training_provenance(root, cache, source) != before


def test_modality_mask_keeps_one_input_and_uses_the_same_mask_across_history():
    # Seed 0 gives draws .496, .768, .088, .132, .307: two vision masks and one lowdim mask.
    torch.manual_seed(0)
    x = torch.ones(5, 2 * 5)
    masked = vip._mask_modalities(x, obs_horizon=2, feat_dim=3, probability=.4).reshape(5, 2, 5)
    assert torch.all(masked[0] == 1) and torch.all(masked[1] == 1)
    assert torch.all(masked[2, :, :3] == 0) and torch.all(masked[2, :, 3:] == 1)
    assert torch.all(masked[4, :, :3] == 1) and torch.all(masked[4, :, 3:] == 0)
    assert torch.all(masked.sum(-1) > 0)
    assert torch.all(x == 1)  # do not mutate the underlying feature cache/batch


def test_modality_dropout_never_masks_validation_inputs():
    torch.manual_seed(8)
    head = torch.nn.Linear(6, 3)
    data = torch.utils.data.TensorDataset(torch.arange(24).float().reshape(4, 6), torch.tensor([0, 1, 2, 0]))
    loader = torch.utils.data.DataLoader(data, batch_size=2)
    normal = vip._run_epoch(head, loader, "cpu", 3)
    masked_config = vip._run_epoch(head, loader, "cpu", 3, modality_dropout=1,
                                   obs_horizon=2, feat_dim=2)
    assert masked_config == normal


def test_runner_rejects_another_arm_set_before_overwriting_results(tmp_path, monkeypatch):
    import importlib.util
    import json
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location("stg_runner", root / "scripts/run_stg_label_loss.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    saved = json.dumps({"rows": {"A": [], "B": [], "C": [], "D": []}})
    (tmp_path / "summary.json").write_text(saved)
    monkeypatch.setattr(sys, "argv", ["runner", "--arms", "A", "E", "--out", str(tmp_path)])
    with pytest.raises(ValueError, match="arm set"):
        runner.main()
    assert (tmp_path / "summary.json").read_text() == saved


def test_curve_collection_preserves_rng_and_head_mode_and_uses_final_weights():
    class Windows(torch.utils.data.Dataset):
        samples = [("a", t) for t in range(9)]

        def __len__(self):
            return 9

        def __getitem__(self, i):
            label = 8 - i
            x = torch.full((9,), -1000.)
            x[label] = 0
            return x, label

    head = torch.nn.Sequential(torch.nn.Dropout(.8), torch.nn.Identity()).train()
    torch.manual_seed(73)
    before = torch.get_rng_state().clone()
    arrays = vip._collect_curve(head, Windows(), list(range(9)), "cpu", 9, 4)
    assert head.training
    torch.testing.assert_close(torch.get_rng_state(), before)
    stats = vip._curve_metrics(arrays)
    assert stats == {"n": 9, "mae": 0., "nll": 0., "k8": 1.}


def test_curve_metrics_reports_absent_cohort_as_null():
    arrays = (np.array([2., 1., 0.]), np.zeros(3), np.array([2, 1, 0]),
              np.repeat("a", 3), np.arange(3))
    assert vip._curve_metrics(arrays, np.zeros(3, bool)) == {
        "n": 0, "mae": None, "nll": None, "k8": None}


def test_wandb_curve_logging_uses_epoch_and_skips_missing_cohort(tmp_path):
    import json
    wandb = pytest.importorskip("wandb")
    from wandb.sdk.internal.datastore import DataStore
    from wandb.proto.wandb_internal_pb2 import Record
    with wandb.init(project="square_assembly_tests", mode="offline", dir=str(tmp_path),
                    save_code=False, settings=wandb.Settings(disable_git=True, console="off")) as run:
        vip._log_curve(run, {"epoch": 2, "train": {"mae": 3., "k8": .9},
                             "val_retained": {"mae": 12., "k8": .8},
                             "val_policy_success": {"n": 0, "mae": None, "k8": None}})
    # SDK logging is asynchronous; inspect the flushed offline history after finish.
    store = DataStore()
    store.open_for_scan(str(next(tmp_path.glob("wandb/offline-run-*/run-*.wandb"))))
    history = {}
    while (data := store.scan_data()) is not None:
        record = Record()
        record.ParseFromString(data)
        if record.HasField("history"):
            history.update({item.key or "/".join(item.nested_key): json.loads(item.value_json)
                            for item in record.history.item})
    assert history["epoch"] == 2
    assert history["train/mae"] == 3.
    assert history["val_retained/k8"] == .8
    assert history["val_policy_success/n"] == 0
    assert "val_policy_success/mae" not in history
