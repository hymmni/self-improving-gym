"""VIP 특징 캐시 검증. robomimic·네트워크 없이 돌아야 한다(pretrained=False, DINO 쪽과 동일 원칙).

다운스트림 데이터셋/헤드(DinoFeatureWindows, DinoStgHead)는 캐시 hdf5의 feat/lowdim
포맷에만 의존하고 인코더를 모르므로 test_dino_stg.py가 이미 검증한다 — 여기서는
cache_vip_feats.py 고유 로직(전처리, 출력 shape/dtype)만 본다.
"""

import numpy as np

from square_assembly.scripts.cache_vip_feats import EMBED_DIM, build_encoder, encode_frames


def test_encode_frames_shape_and_dtype():
    model = build_encoder(pretrained=False, device="cpu")
    imgs = np.random.randint(0, 256, size=(3, 84, 84, 3), dtype=np.uint8)

    feat = encode_frames(model, imgs, crop=76, device="cpu", batch_size=2)

    assert feat.shape == (3, EMBED_DIM)
    assert feat.dtype == np.float16
    assert np.isfinite(feat).all()


def test_encode_frames_is_deterministic():
    """얼린 인코더이므로 같은 프레임은 항상 같은 특징이어야 한다 — 캐싱 방식 자체의 전제.

    ViT(DINO)와 달리 ResNet(VIP)은 conv의 im2col 크기가 배치 크기에 따라 달라져
    CPU에서 마지막 몇 비트가 흔들릴 수 있다(부동소수점 결합법칙 위반, 값 자체는 동일
    분포) — 그래서 완전 동일이 아니라 허용오차 비교로 배치 불변성만 확인한다.
    """
    model = build_encoder(pretrained=False, device="cpu")
    imgs = np.random.randint(0, 256, size=(2, 84, 84, 3), dtype=np.uint8)

    a = encode_frames(model, imgs, crop=76, device="cpu", batch_size=2)
    b = encode_frames(model, imgs, crop=76, device="cpu", batch_size=1)

    np.testing.assert_allclose(a.astype(np.float32), b.astype(np.float32), rtol=1e-2, atol=1e-2)
