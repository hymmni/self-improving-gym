"""사전추출된 DINO 특징 위의 STG(steps-to-go) 카테고리컬 헤드.

역할은 기존 DstgPredictor(dstg_predictor.py)와 같다 — 논문 식(1)
d(o,g) := E[steps-to-go | o]를 bin categorical로 예측한다. 다른 건 입력뿐이다:
얼린 diffusion policy 인코더의 global_cond 대신, scripts/cache_dino_feats.py가
미리 구운 DINO 특징 윈도우를 받는다. 원본은 비교 기준선이라 건드리지 않고 새 파일로 둔다
(ADR-005).

헤드 구조는 기준선과 동일하게 유지한다: MLP(head_hidden) -> Linear(num_bins, bias=False).
인코더 말고 다른 변인이 끼면 비교가 성립하지 않는다.

체크포인트엔 전처리 메타(cache_meta: model/crop/size)와 표준화 통계(frame_mean/frame_std)를
같이 저장한다 — 후속 작업(SI 루프 온라인 보상)이 cache_dino_feats.encode_frames()에 같은
설정을 그대로 먹여 train/serve 불일치 없이 재현할 수 있게 하기 위함이다.
"""

import numpy as np
import torch
import torch.nn as nn


class DinoStgHead(nn.Module):
    def __init__(self, in_dim, num_bins, head_hidden=(256, 256)):
        super().__init__()
        layers, d = [], in_dim
        for h in head_hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, num_bins, bias=False))
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        """x: (B, obs_horizon * frame_dim) 표준화된 특징 윈도우 -> (B, num_bins) logits."""
        return self.head(x)


def save_checkpoint(path, head, meta):
    payload = {"model": head.head.state_dict()}
    payload.update(meta)
    for key in ("frame_mean", "frame_std"):
        if isinstance(payload.get(key), np.ndarray):
            payload[key] = payload[key].tolist()
    torch.save(payload, path)


def load_checkpoint(path, device="cpu"):
    """반환: (head, ckpt). ckpt의 cache_meta/frame_mean/frame_std로 전처리를 그대로 재현할 수 있다."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    head = DinoStgHead(ckpt["in_dim"], ckpt["num_bins"], tuple(ckpt["head_hidden"]))
    head.head.load_state_dict(ckpt["model"])
    return head.to(device).eval(), ckpt
