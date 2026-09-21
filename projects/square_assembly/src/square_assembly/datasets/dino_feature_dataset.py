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
        self.preintv = preintv
        self.preintv_base = {}   # (demo, t) -> 이 PREINTV 구간에 쓸 대체 라벨
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
        if fail_bin is None and not all(self.success.values()):
            raise ValueError(
                "실패 데모가 섞여 있는데 fail_bin이 없다 — 실패 transition에 성공 기준 "
                "steps-to-go 라벨을 붙이는 건 범주 오류다(train_dstg_failaware.py 참고)"
            )
        self.frame_dim = next(iter(self.frames.values())).shape[1]
        self.modes = _load_modes(mode_hdf5, self.frames, self.lengths) if mode_hdf5 else {}
        if preintv not in ("none", "drop", "flat", "rise"):
            raise ValueError(f"preintv={preintv!r}는 none/drop/flat/rise 중 하나여야 한다")
        if preintv != "none":
            if not self.modes:
                raise ValueError("preintv 처리를 쓰려면 action_mode를 읽을 mode_hdf5가 필요하다")
            self._build_preintv_labels()

    def _build_preintv_labels(self):
        """PREINTV 연속 구간마다 라벨을 다시 매긴다.

        카운트다운 라벨은 정책이 망가지는 중인 이 구간에서도 매 스텝 1씩 줄어 "좋아지고
        있다"고 말한다 — 사람이 곧 개입한다는 사실과 정면으로 어긋난다. flat은 구간 내내
        같은 값(= 진전 0), rise는 스텝마다 1씩 올린다(= 후퇴). 어느 쪽도 라벨의 수준은
        구간 시작점의 참값으로 유지하고 기울기만 바꾼다.
        """
        from square_assembly.datasets.labels import LABEL_PREINTV
        for name, mode in self.modes.items():
            if not self.success.get(name, False):
                continue
            idx = np.flatnonzero(mode == LABEL_PREINTV)
            if len(idx) == 0:
                continue
            brk = np.flatnonzero(np.diff(idx) > 1)
            starts = np.concatenate([[idx[0]], idx[brk + 1]])
            ends = np.concatenate([idx[brk], [idx[-1]]])
            for a, b in zip(starts, ends):
                base = self.lengths[name] - 1 - a
                for t in range(a, b + 1):
                    self.preintv_base[(name, t)] = base + (t - a if self.preintv == "rise" else 0)

    def preintv_indices(self):
        """PREINTV 프레임의 샘플 인덱스 — preintv='drop'일 때 train에서만 빼기 위함."""
        from square_assembly.datasets.labels import LABEL_PREINTV
        if not self.modes:
            return []
        return [i for i, (n, t) in enumerate(self.samples)
                if n in self.modes and self.modes[n][t] == LABEL_PREINTV]

    def __len__(self):
        return len(self.samples)

    def _label(self, name, t):
        """이 데이터셋의 유일한 라벨 정의 — labels()와 __getitem__이 갈라지면 학습과 평가가
        조용히 어긋난다.

        label_horizon(H)을 주면 '남은 스텝 수' 대신 '남은 비율 x H'를 라벨로 쓴다. 사람이
        얼마나 빨리 몰았는지에 불변이라, 에피소드 길이가 들쭉날쭉한 텔레옵 데이터에서
        "관측 -> 남은 스텝"이 ill-posed가 되는 걸 막는다(experiments 2026-09-18 §7).
        """
        if not self.success[name]:
            return self.fail_bin
        remaining = self.preintv_base.get((name, t), self.lengths[name] - 1 - t)
        if self.label_horizon is None:
            return remaining
        span = max(self.lengths[name] - 1, 1)
        return int(round(remaining / span * self.label_horizon))

    def labels(self):
        """(N,) int64 — num_bins 결정과 로깅용. __getitem__을 N번 부르지 않고 한 번에 계산한다."""
        return np.array([self._label(n, t) for n, t in self.samples], dtype=np.int64)

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
        return torch.from_numpy(x.copy()), int(self._label(name, t))
