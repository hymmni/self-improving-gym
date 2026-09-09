"""DINO 특징 캐시·데이터셋·STG 헤드 검증.

이 파일은 robomimic 없이 돌아야 한다(로컬 PC에 robomimic이 없다). 사전학습
가중치 다운로드도 하지 않는다(pretrained=False) — 네트워크 없이 어디서나 돈다.
실제 가중치 다운로드는 캐시 스크립트 스모크 실행이 확인한다.
"""

import numpy as np
import torch

from square_assembly.scripts.cache_dino_feats import DEFAULT_MODEL, build_encoder, encode_frames


def test_encode_frames_shape_and_dtype():
    model = build_encoder(DEFAULT_MODEL, pretrained=False, device="cpu")
    imgs = np.random.randint(0, 256, size=(3, 84, 84, 3), dtype=np.uint8)

    feat = encode_frames(model, imgs, crop=76, device="cpu", batch_size=2)

    assert feat.shape == (3, 2 * model.embed_dim)  # [CLS ; mean(patch)]
    assert feat.dtype == np.float16
    assert np.isfinite(feat).all()


def test_encode_frames_is_deterministic():
    """얼린 인코더이므로 같은 프레임은 항상 같은 특징이어야 한다 — 캐싱 방식 자체의 전제."""
    model = build_encoder(DEFAULT_MODEL, pretrained=False, device="cpu")
    imgs = np.random.randint(0, 256, size=(2, 84, 84, 3), dtype=np.uint8)

    a = encode_frames(model, imgs, crop=76, device="cpu", batch_size=2)
    b = encode_frames(model, imgs, crop=76, device="cpu", batch_size=1)

    np.testing.assert_array_equal(a, b)
