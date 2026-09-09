"""robomimic 형식 hdf5의 모든 프레임을 얼린 DINOv2로 인코딩해 특징 캐시(hdf5)로 굽는다.

STG 예측기 d(o,g)의 비전 인코더를, 학습된 diffusion policy의 ResNet18+SpatialSoftmax
(policies/diffusion/dstg_predictor.py) 대신 자기지도 사전학습 ViT로 갈아끼우기 위한
1회성 전처리다. 인코더가 얼어 있으니 "같은 프레임 -> 항상 같은 특징"이고, 매 에폭
이미지를 다시 로드해 인코더를 다시 통과시킬 이유가 없다
(docs/superpowers/specs/2026-09-09-dino-stg-encoder-design.md).

전처리는 기준선의 **eval 시 경로**와 맞춘다: 84x84 -> CenterCrop 76 -> resize 224 ->
ImageNet 정규화. crop 76은 configs/policy/diffusion_unet.yaml의 crop_hw와 같은 값이다 —
시야각이 다르면 val 지표 비교 자체가 성립하지 않는다. (기준선은 학습 시엔 RandomCrop을
쓰지만 캐시 방식은 증강을 잃는다 — spec 3절의 알려진 트레이드오프.)

robomimic을 쓰지 않는다 — h5py로 hdf5를 직접 읽는다.

사용:
    python -m square_assembly.scripts.cache_dino_feats \
        --hdf5 data/square_scale3_0_seed1.hdf5 \
        --out data/square_scale3_0_seed1.dino.h5
"""

import argparse
import json

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# timm 0.9.16(서버 canonical env)·1.0.20(로컬 base env) 양쪽에 있는 이름.
# 없다면 --model vit_small_patch14_dinov2.lvd142m 로 폴백한다(레지스터 없는 버전).
DEFAULT_MODEL = "vit_small_patch14_reg4_dinov2.lvd142m"
DEFAULT_SIZE = 224  # patch14로 나눠떨어지는 표준 해상도(16x16 패치)
DEFAULT_RGB_KEYS = ("agentview_image", "robot0_eye_in_hand_image")
DEFAULT_LOWDIM_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")


def build_encoder(model_name=DEFAULT_MODEL, pretrained=True, device="cpu", img_size=DEFAULT_SIZE):
    """num_classes=0으로 분류 헤드를 떼고 특징 추출기로만 쓴다.

    img_size를 넘기는 이유: DINOv2 계열의 timm 기본 img_size는 518이라 224 입력을 그대로
    거부한다(patch_embed의 크기 assert). timm이 이 인자로 위치 임베딩을 리샘플해준다.
    """
    import timm

    model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size)
    return model.eval().to(device)


@torch.no_grad()
def encode_frames(model, imgs_u8, crop=76, device="cpu", batch_size=128):
    """(T,H,W,3) uint8 -> (T, 2*D) float16.  프레임당 [CLS ; mean(patch tokens)].

    캐싱과 (후속 작업의) 온라인 추론이 반드시 같은 전처리를 타도록, 전처리는 이 함수
    하나에만 존재한다. 리사이즈 목표 해상도는 인자로 받지 않고 모델에서 역산한다 —
    build_encoder의 img_size와 어긋날 여지를 아예 없앤다.
    """
    size = model.patch_embed.img_size[0]
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    n_prefix = model.num_prefix_tokens  # CLS(+register) 토큰 수 — 패치 토큰의 시작 위치

    out = []
    for i in range(0, len(imgs_u8), batch_size):
        chunk = np.ascontiguousarray(imgs_u8[i : i + batch_size])
        x = torch.from_numpy(chunk).to(device).permute(0, 3, 1, 2).float().div_(255.0)
        if crop:
            x = TF.center_crop(x, [crop, crop])
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        x = (x - mean) / std
        tokens = model.forward_features(x)  # (B, n_prefix + N, D)
        feat = torch.cat([tokens[:, 0], tokens[:, n_prefix:].mean(dim=1)], dim=-1)
        out.append(feat.half().cpu())
    return torch.cat(out).numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hdf5", required=True, help="robomimic 형식 입력 hdf5")
    p.add_argument("--out", required=True, help="출력 특징 캐시 hdf5")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--rgb-keys", nargs="+", default=list(DEFAULT_RGB_KEYS))
    p.add_argument("--lowdim-keys", nargs="+", default=list(DEFAULT_LOWDIM_KEYS))
    p.add_argument("--crop", type=int, default=76, help="0이면 crop 없이 원본 해상도 사용")
    p.add_argument("--size", type=int, default=DEFAULT_SIZE)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--limit-demos", type=int, default=0, help=">0이면 앞 N개 데모만 (스모크용)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    import timm

    model = build_encoder(args.model, pretrained=True, device=args.device, img_size=args.size)
    print(f"encoder={args.model} timm={timm.__version__} device={args.device}", flush=True)

    with h5py.File(args.hdf5, "r") as src, h5py.File(args.out, "w") as dst:
        demos = sorted(src["data"].keys())
        if args.limit_demos:
            demos = demos[: args.limit_demos]
        for i, name in enumerate(demos):
            g = src["data"][name]
            length = int(g["actions"].shape[0])
            feat = np.concatenate(
                [
                    encode_frames(model, g["obs"][k][:], args.crop, args.device, args.batch_size)
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
                "model": args.model,
                "timm": timm.__version__,
                "crop": args.crop,
                "size": args.size,
                "rgb_keys": list(args.rgb_keys),
                "lowdim_keys": list(args.lowdim_keys),
            }
        )
    print(f"saved: {args.out}", flush=True)


if __name__ == "__main__":
    main()
