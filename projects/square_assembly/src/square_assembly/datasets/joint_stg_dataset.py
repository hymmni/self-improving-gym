"""공유 백본 DP + STG 헤드 학습 데이터 — 병합 hdf5(PH 시연 + 개입 롤아웃)를 메모리에 올려 윈도우를 만든다.

윈도우 규칙은 기본 정책을 학습한 ZarrSequenceDataset과 같다(가장자리 반복 패딩, action_mask). robomimic
SequenceDataset의 cache "all"은 인덱스마다 17프레임 이미지를 미리 펼쳐 이 데이터에서 RAM 31GB를 쓰므로 쓰지 않는다.

샘플마다 역할이 둘이다. 인덱스 i < N은 BC 역할(시연 프레임, 디퓨전 손실), i >= N은 같은 프레임 i-N의 STG 역할
(CE 손실). 같은 PH 프레임이 두 역할로 다 뽑힐 수 있고, 역할은 JointBatchSampler가 배치 안 위치로 정한다.
- BC 풀: action_mode가 전부 시연(-1)인 데모의 모든 프레임 — 기본 정책(PH 50)과 같은 BC 데이터. 검증 분할과 무관.
- STG 풀: ① 예측기(experiments/2026-10-05_stg-ph50-r0v9-r0v10.md) split과 같은 학습 데모의 성공 구간(첫 개입부터,
  success_tail_from=first). 라벨 = 남은 스텝 L-1-t(build_labels), num_bins-1에서 자름.
"""

import h5py
import numpy as np

from square_assembly.datasets.dino_feature_dataset import episode_split_by_name
from square_assembly.datasets.labels import LABEL_DEMO
from square_assembly.datasets.replay_buffer import ReplayBuffer
from square_assembly.datasets.stg_labels import build_labels, success_tail_start
from square_assembly.datasets.zarr_dataset import ZarrSequenceDataset


class JointStgDataset(ZarrSequenceDataset):
    def __init__(self, hdf5_path, obs_keys, obs_horizon, pred_horizon, normalizer, rgb_keys,
                 val_fraction, split_seed, num_bins):
        self.rgb_keys, self.obs_keys = list(rgb_keys), list(obs_keys)
        self.obs_horizon, self.pred_horizon = obs_horizon, pred_horizon
        self.normalizer, self.action_key, self.extra_keys = normalizer, "action", []

        with h5py.File(hdf5_path, "r") as f:
            names = sorted(f["data"].keys(), key=lambda n: int(n.split("_")[1]))
            data = {k: np.concatenate([f[f"data/{n}/obs/{k}"][()] for n in names]) for k in self.obs_keys}
            data["action"] = np.concatenate([f[f"data/{n}/actions"][()] for n in names])
            modes = {n: f[f"data/{n}/action_mode"][()] for n in names}
            if not all(bool(f[f"data/{n}"].attrs.get("is_success", True)) for n in names):
                raise ValueError("실패 데모가 섞여 있다 — 이 데이터셋은 성공 데모만 받는다")
        lengths = np.array([len(modes[n]) for n in names])
        self._buffer = ReplayBuffer({"data": data, "meta": {"episode_ends": np.cumsum(lengths)}})
        self._episode_ends = np.cumsum(lengths)
        self._episode_starts = np.concatenate([[0], self._episode_ends[:-1]])
        self._index = [(d, t) for d, n in enumerate(names) for t in range(lengths[d])]

        sample_names = [names[d] for d, _ in self._index]
        ts = np.array([t for _, t in self._index])
        labels, _ = build_labels(sample_names, ts, dict(zip(names, lengths)), {n: True for n in names})
        self.stg_labels = np.minimum(labels, num_bins - 1)
        val = episode_split_by_name(names, val_fraction, split_seed)
        # ponytail: ①은 성공 구간 첫 프레임의 이력(obs_horizon=2의 직전 프레임)을 구간 시작으로 고정했지만 여기선 안 한다
        # — 개입 데모마다 1프레임(약 200/32k)만 다르다.
        starts = {n: success_tail_start(modes[n], first=True) for n in names}
        demo_only = {n: bool((modes[n] == LABEL_DEMO).all()) for n in names}
        self.bc_idx = np.array([i for i, (d, _) in enumerate(self._index) if demo_only[names[d]]])
        self.stg_idx = np.array([i for i, (d, t) in enumerate(self._index)
                                 if names[d] not in val and t >= starts[names[d]]])
        self._stg_set = set(self.stg_idx.tolist())

    def __len__(self):
        return 2 * len(self._index)

    def __getitem__(self, index):
        n = len(self._index)
        frame, stg_role = index % n, index >= n
        item = super().__getitem__(frame)
        item["bc_mask"] = not stg_role
        item["stg_mask"] = stg_role and frame in self._stg_set
        item["stg_label"] = int(self.stg_labels[frame])
        return item


class JointBatchSampler:
    """배치 = BC 풀 bc_batch개(epoch마다 섞어 겹침 없이, drop_last) + STG 풀 stg_batch개(복원 추출, 인덱스 +N).

    epoch 길이는 BC 풀 기준이라 기본 정책(PH 50, batch 64 → 118스텝/epoch)과 같은 스텝 수로 학습한다.
    """

    def __init__(self, bc_idx, stg_idx, n_frames, bc_batch, stg_batch, seed):
        self.bc_idx, self.stg_idx, self.n = np.asarray(bc_idx), np.asarray(stg_idx), n_frames
        self.bc_batch, self.stg_batch, self.seed, self.epoch = bc_batch, stg_batch, seed, 0

    def __len__(self):
        return len(self.bc_idx) // self.bc_batch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        bc = rng.permutation(self.bc_idx)
        for b in range(len(self)):
            stg = rng.choice(self.stg_idx, self.stg_batch) + self.n
            yield [int(i) for i in bc[b * self.bc_batch:(b + 1) * self.bc_batch]] + [int(i) for i in stg]
