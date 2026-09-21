"""DINO 특징 캐시(scripts/cache_dino_feats.py 산출물)를 STG 학습 샘플로 자른다.

인덱싱은 기준선(scripts/train_dstg.py)이 쓰는 robomimic SequenceDataset
(frame_stack=To, seq_length=1, pad_frame_stack=True, pad_seq_length=True) +
RobomimicSequenceDataset.get_time_to_success와 **정확히 같게** 맞춘다:

  - 데모 d의 샘플 수 = L_d, index_in_demo t = 0..L_d-1
  - 관측 윈도우 = 프레임 clip(t-To+1 .. t, 0, L_d-1)   (앞쪽 패딩 = 첫 프레임 복제)
  - 라벨 = L_d - 1 - t (성공 데모) / fail_bin (실패 데모)

여기가 반 칸만 어긋나도 학습은 멀쩡히 돌고 val 지표만 조용히 틀리므로, 이 동일성은
tests/test_dino_stg.py가 손으로 계산한 기대값과 train_dstg._episode_split 대조로 못 박는다.

robomimic을 import하지 않는다 — 캐시에 필요한 정보(데모 이름, 길이, 성공 여부)가 전부 있다.
"""

import h5py
import numpy as np
import torch

from square_assembly.datasets.stg_labels import build_labels


def episode_split_by_name(demo_names, val_fraction, seed):
    """train_dstg._episode_split과 동일한 val 데모 집합을, 데모 이름만으로 재현한다.

    원본은 robomimic의 _index_to_demo_id에서 데모 이름을 뽑아
    sorted(set(...)) (사전순) -> RandomState(seed).permutation -> 앞 n_val개를 val로 쓴다.
    캐시엔 데모 이름이 그대로 남아 있으므로 robomimic 없이 같은 결과가 나온다.
    """
    unique = sorted(set(demo_names))
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(unique))
    n_val = max(1, int(round(len(unique) * val_fraction)))
    return {unique[i] for i in perm[:n_val]}


def _load_modes(mode_hdf5, frames, lengths):
    """원본 hdf5에서 데모별 action_mode를 읽는다 — 캐시엔 없는 정보다."""
    out = {}
    with h5py.File(mode_hdf5, "r") as f:
        for name in frames:
            g = f["data"].get(name)
            if g is None or "action_mode" not in g:
                continue
            m = np.asarray(g["action_mode"])
            if len(m) < lengths[name]:  # 마지막 프레임 복제로 길이를 맞춘다
                m = np.concatenate([m, np.repeat(m[-1:], lengths[name] - len(m))])
            out[name] = m[: lengths[name]]
    return out


class DinoFeatureWindows(torch.utils.data.Dataset):
    """캐시를 통째로 메모리에 올려 (obs 윈도우, steps-to-go 라벨) 샘플을 낸다.

    Args:
        cache_path (str): cache_dino_feats.py가 만든 hdf5.
        obs_horizon (int): To — 기준선 정책과 같은 값을 써야 한다
            (configs/policy/diffusion_unet.yaml: 2).
        fail_bin (int | None): 실패 데모에 붙일 별도 클래스. 실패 데모가 있는데 None이면 죽는다.
    """

    def __init__(self, cache_path, obs_horizon, fail_bin=None, label_horizon=None,
                 mode_hdf5=None, preintv="none"):
        self.obs_horizon = obs_horizon
        self.fail_bin = fail_bin
        self.label_horizon = label_horizon
        self.frames, self.lengths, self.success = {}, {}, {}
        self.samples = []

        with h5py.File(cache_path, "r") as f:
            for name in sorted(f["data"].keys()):
                g = f["data"][name]
                length = int(g.attrs["length"])
                # ponytail: 캐시 전체를 float32로 메모리에 올린다. 7.5k 프레임(=46MB)엔 과하지 않지만
                # 19만 프레임(square_scale3_1000)이면 ~1.2GB — 그때 float16 유지 + 배치 단위
                # 캐스팅으로 바꾼다.
                feat = np.asarray(g["feat"][:], dtype=np.float32)
                low = np.asarray(g["lowdim"][:], dtype=np.float32)
                self.frames[name] = np.concatenate([feat, low], axis=-1)
                self.lengths[name] = length
                self.success[name] = bool(g.attrs["is_success"])
                self.samples += [(name, t) for t in range(length)]

        if not self.samples:
            raise ValueError(f"{cache_path}에 데모가 없다")
        self.frame_dim = next(iter(self.frames.values())).shape[1]
        self.modes = _load_modes(mode_hdf5, self.frames, self.lengths) if mode_hdf5 else {}
        names = [n for n, _ in self.samples]
        ts = [t for _, t in self.samples]
        self._labels, self._preintv_mask = build_labels(
            names, ts, self.lengths, self.success, modes=self.modes, fail_bin=fail_bin,
            label_horizon=label_horizon, preintv=preintv)

    def labels(self):
        """(N,) int64 — num_bins 결정과 로깅용."""
        return self._labels

    def preintv_indices(self):
        """PREINTV 프레임의 샘플 인덱스 — preintv='drop'일 때 train에서만 빼기 위함."""
        return np.flatnonzero(self._preintv_mask).tolist()

    def split_indices(self, val_fraction, seed):
        val_demos = episode_split_by_name(list(self.frames), val_fraction, seed)
        train_idx = [i for i, (n, _) in enumerate(self.samples) if n not in val_demos]
        val_idx = [i for i, (n, _) in enumerate(self.samples) if n in val_demos]
        return train_idx, val_idx, val_demos

    def compute_frame_stats(self, indices):
        """주어진 샘플 인덱스가 속한 데모들의 프레임으로 (mean, std)를 낸다 — val 누출 방지."""
        names = sorted({self.samples[i][0] for i in indices})
        stacked = np.concatenate([self.frames[n] for n in names], axis=0)
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        std[std < 1e-6] = 1.0  # 상수 차원에서 0으로 나누지 않게
        return mean.astype(np.float32), std.astype(np.float32)

    def apply_frame_stats(self, mean, std):
        for name in self.frames:
            self.frames[name] = (self.frames[name] - mean) / std

    def __getitem__(self, i):
        name, t = self.samples[i]
        length = self.lengths[name]
        idx = np.clip(np.arange(t - self.obs_horizon + 1, t + 1), 0, length - 1)
        x = self.frames[name][idx].reshape(-1)
        return torch.from_numpy(x.copy()), int(self._labels[i])
