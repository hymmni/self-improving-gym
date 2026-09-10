"""robomimic 형식 hdf5의 모든 프레임을 얼린 VIP로 인코딩해 특징 캐시(hdf5)로 굽는다.

cache_dino_feats.py와 목적·출력 포맷이 동일하다 — STG 예측기 d(o,g)의 비전 인코더를
자기지도 사전학습 백본으로 갈아끼우는 두 번째 시도다. DINO(범용 시각 표현) 대신,
사전학습 목적 자체가 "임베딩 거리 ≈ 목표까지 남은 시간"인 VIP(Value-Implicit
Pre-training, Ma et al. 2023)를 쓴다 — steps-to-go 라벨과 사전학습 목적이 더 가깝다.

VIP는 timm에 없다 — 공식 구현(github.com/facebookresearch/vip)을 설치해 쓴다:
    pip install git+https://github.com/facebookresearch/vip.git

전처리는 DINO 쪽과 같은 이유로 84x84 -> CenterCrop 76 -> resize 224 (기준선
eval 경로와 시야각을 맞춤). 다만 정규화는 다르다 — VIP 모델은 forward 내부에서 자체
전처리(0-255 스케일 입력을 기대, ImageNet 정규화를 스스로 적용)를 하므로 여기서는
ImageNet mean/std를 따로 적용하지 않는다(공식 레포 README 사용 예시: ToTensor()로
[0,1]을 만든 뒤 다시 *255 해서 넘긴다 — 결국 0-255 float를 그대로 넣는 것과 같다).

특징 형태: 카메라당 임베딩 1024차원(VIP는 ViT처럼 CLS/patch 토큰이 없는 단일 벡터).
카메라 2대 -> 프레임당 2048. (DINO의 프레임당 1536보다 크다 — 사전학습 목적의 적합성이
관건이지, 이번엔 차원 축소가 강점은 아니다.)

robomimic을 쓰지 않는다 — h5py로 hdf5를 직접 읽는다.

사용:
    python -m square_assembly.scripts.cache_vip_feats \
        --hdf5 data/square_scale3_0_seed1.hdf5 \
        --out data/square_scale3_0_seed1.vip.h5
"""

import argparse
import json

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

EMBED_DIM = 1024  # VIP 논문: ResNet50 backbone -> 1024차원 임베딩
DEFAULT_SIZE = 224
DEFAULT_RGB_KEYS = ("agentview_image", "robot0_eye_in_hand_image")
DEFAULT_LOWDIM_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")


def build_encoder(pretrained=True, device="cpu"):
    """pretrained=False면 네트워크 없이 구조(resnet50 -> 1024d)만 맞춘 랜덤 가중치를 쓴다
    — VIP 공식 로더(vip.load_vip())는 항상 사전학습 체크포인트를 내려받으므로 로컬
    테스트(test_vip_stg.py)에서는 이 경로로 encode_frames의 shape/전처리만 검증한다.
    실제 값 검증은 서버에서의 캐시 스모크 실행이 한다.
    """
    if pretrained:
        import vip

        model = vip.load_vip()
    else:
        import torchvision

        model = torchvision.models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, EMBED_DIM)
    return model.eval().to(device)


@torch.no_grad()
def encode_frames(model, imgs_u8, crop=76, size=DEFAULT_SIZE, device="cpu", batch_size=128):
    """(T,H,W,3) uint8 -> (T, EMBED_DIM) float16.

    ImageNet 정규화를 안 하는 이유는 모듈 docstring 참고 — VIP가 forward 안에서 한다.
    """
    out = []
    for i in range(0, len(imgs_u8), batch_size):
        chunk = np.ascontiguousarray(imgs_u8[i : i + batch_size])
        x = torch.from_numpy(chunk).to(device).permute(0, 3, 1, 2).float()  # 0-255 스케일 유지
        if crop:
            x = TF.center_crop(x, [crop, crop])
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        feat = model(x)
        out.append(feat.half().cpu())
    return torch.cat(out).numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hdf5", required=True, help="robomimic 형식 입력 hdf5")
    p.add_argument("--out", required=True, help="출력 특징 캐시 hdf5")
    p.add_argument("--rgb-keys", nargs="+", default=list(DEFAULT_RGB_KEYS))
    p.add_argument("--lowdim-keys", nargs="+", default=list(DEFAULT_LOWDIM_KEYS))
    p.add_argument("--crop", type=int, default=76, help="0이면 crop 없이 원본 해상도 사용")
    p.add_argument("--size", type=int, default=DEFAULT_SIZE)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--limit-demos", type=int, default=0, help=">0이면 앞 N개 데모만 (스모크용)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    model = build_encoder(pretrained=True, device=args.device)
    print(f"encoder=vip embed_dim={EMBED_DIM} device={args.device}", flush=True)

    with h5py.File(args.hdf5, "r") as src, h5py.File(args.out, "w") as dst:
        demos = sorted(src["data"].keys())
        if args.limit_demos:
            demos = demos[: args.limit_demos]
        for i, name in enumerate(demos):
            g = src["data"][name]
            length = int(g["actions"].shape[0])
            feat = np.concatenate(
                [
                    encode_frames(
                        model, g["obs"][k][:], args.crop, args.size, args.device, args.batch_size
                    )
                    for k in args.rgb_keys
                ],
                axis=-1,
            )
            low = np.concatenate(
                [np.asarray(g["obs"][k][:], dtype=np.float32) for k in args.lowdim_keys], axis=-1
            )
            if feat.shape[0] != length or low.shape[0] != length:
                raise ValueError(
                    f"{name}: 프레임 수 불일치 feat={feat.shape[0]} lowdim={low.shape[0]} actions={length}"
                )

            og = dst.create_group(f"data/{name}")
            og.create_dataset("feat", data=feat)
            og.create_dataset("lowdim", data=low)
            og.attrs["is_success"] = bool(np.asarray(g.attrs.get("is_success", True)))
            og.attrs["length"] = length
            print(f"[{i + 1}/{len(demos)}] {name} T={length} feat={feat.shape}", flush=True)

        dst.attrs["meta"] = json.dumps(
            {
                "source_hdf5": args.hdf5,
                "model": "vip",
                "embed_dim": EMBED_DIM,
                "crop": args.crop,
                "size": args.size,
                "rgb_keys": list(args.rgb_keys),
                "lowdim_keys": list(args.lowdim_keys),
            }
        )
    print(f"saved: {args.out}", flush=True)


if __name__ == "__main__":
    main()
