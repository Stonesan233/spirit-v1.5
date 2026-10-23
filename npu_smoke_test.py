"""Smoke test: run SpiritVLAPolicy.select_action on Ascend NPU (offline).

Usage (inside the vllm-ascend container):
python npu_smoke_test.py --ckpt /home/Spirit-v1.5 --backbone /home/qwen3vl-4b-small [--bf16]
"""

import argparse
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

from model.modeling_spirit_vla import SpiritVLAPolicy


def build_dummy_batch(device):
    return {
        "observation.images.cam_high": torch.rand(1, 3, 240, 320, device=device),
        "observation.images.cam_left_wrist": torch.rand(1, 3, 240, 320, device=device),
        "observation.images.cam_right_wrist": torch.rand(1, 3, 240, 320, device=device),
        "observation.state": torch.rand(1, 1, 14, device=device),
        "task": ["move objects into the box"],
        "robot_type": ["Franka"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="/home/Spirit-v1.5")
    parser.add_argument("--backbone", default="/home/qwen3vl-4b-small")
    parser.add_argument("--bf16", action="store_true", help="cast non-norm modules to bf16 (halves HBM)")
    args = parser.parse_args()

    device = torch.device("npu")
    print(f"loading policy from {args.ckpt} (backbone: {args.backbone}) ...")
    policy = SpiritVLAPolicy.from_pretrained(args.ckpt, backbone=args.backbone)
    policy.eval()
    policy.to(device)

    if args.bf16:
        norm_prefixes = ("normalize_inputs", "normalize_targets", "unnormalize_outputs")
        for name, module in policy.named_children():
            if name.startswith(norm_prefixes):
                continue
            module.to(torch.bfloat16)
        print("cast non-norm modules to bf16")

    batch = build_dummy_batch(device)
    with torch.inference_mode(), torch.autocast("npu", dtype=torch.bfloat16):
        actions = policy.select_action(batch)

    actions = actions.float().cpu()
    print("action shape:", tuple(actions.shape))
    print("finite:", bool(torch.isfinite(actions).all()))
    print("first step:", np.round(actions[0, 0].tolist(), 4))
    print("last step: ", np.round(actions[0, -1].tolist(), 4))


if __name__ == "__main__":
    main()
