"""Operator-level determinism bisect for the Spirit-v1.5 DiT head.

Registers forward hooks on the DiT leaf modules, runs the same denoise pass
three times with identical inputs, and reports the first module whose output is
not bitwise identical. Intended as a diagnostic when two identical runs produce
different actions.

Usage:
  python dit_bisect.py --ckpt /home/Spirit-v1.5 --backbone /home/qwen3vl-4b-small \
      --device npu --runs 3 --log artifacts/dit_bisect.log
"""

import argparse
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from torch import nn

from perf_bench import (
    autocast_ctx,
    inputs_to_batch,
    load_inputs,
    load_policy,
    make_inputs,
    pick_device,
    sync,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="/home/Spirit-v1.5")
    parser.add_argument("--backbone", default="/home/qwen3vl-4b-small")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "npu", "cpu"])
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--no-autocast", action="store_true")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--inputs", default=None)
    parser.add_argument("--log", default="dit_bisect.log")
    args = parser.parse_args()

    device = pick_device(args.device)
    use_autocast = not args.no_autocast
    print(f"device={device} runs={args.runs} autocast={use_autocast}")
    print(f"loading policy from {args.ckpt} (backbone: {args.backbone}) ...")
    policy = load_policy(args.ckpt, args.backbone, device, args.bf16)
    if args.no_autocast and not args.bf16:
        policy.to(torch.float32)
        print("cast policy to float32 (no-autocast)")

    if args.inputs:
        inputs = load_inputs(args.inputs)
    else:
        inputs = make_inputs(args.batch, args.seed,
                             policy.config.n_action_steps, policy.config.max_action_dim)
    batch = inputs_to_batch(inputs, device)
    noise = inputs["noise"].to(device)

    with torch.inference_mode(), autocast_ctx(device, use_autocast):
        normed = policy.normalize_targets(policy.normalize_inputs(batch))
        state = policy.prepare_state(normed)
        vlm = policy._encode_vision(normed)
        suffix, _, _ = policy._embed_suffix(state, noise.clone(), mask_state=False)
        timestep = torch.ones(inputs["state"].shape[0], device=device)

    records = {}
    forward_order = []
    handles = []

    def make_hook(name):
        def hook(module, hook_inputs, output):
            tensor = output
            if isinstance(tensor, (tuple, list)):
                tensor = next((x for x in tensor if torch.is_tensor(x)), None)
            if not torch.is_tensor(tensor):
                return
            records.setdefault(name, []).append(tensor.detach().float().cpu())
            if len(records[name]) == 1:
                forward_order.append(name)
        return hook

    for name, module in policy.dit.named_modules():
        if not name:
            continue
        is_leaf = len(list(module.children())) == 0
        interesting = isinstance(module, (nn.Linear, nn.LayerNorm, nn.GELU, nn.SiLU))
        if is_leaf or interesting:
            handles.append(module.register_forward_hook(make_hook(name)))

    try:
        for _ in range(args.runs):
            with torch.inference_mode(), autocast_ctx(device, use_autocast):
                policy.dit(
                    hidden_states=suffix.clone(),
                    encoder_hidden_states=vlm.clone(),
                    timestep=timestep.clone(),
                )
            sync(device)
    finally:
        for handle in handles:
            handle.remove()

    mismatches = []
    compared = 0
    for name in forward_order:
        outputs = records[name]
        if len(outputs) != args.runs:
            continue
        compared += 1
        diffs = []
        for run_idx in range(1, args.runs):
            if outputs[0].shape != outputs[run_idx].shape:
                diffs.append(float("inf"))
                continue
            diffs.append(float((outputs[0] - outputs[run_idx]).abs().max().item()))
        if any(diff != 0.0 for diff in diffs):
            mismatches.append((name, diffs))

    lines = [
        f"device={device} runs={args.runs} autocast={use_autocast} bf16_weights={args.bf16}",
        f"modules called={len(forward_order)} compared={compared}",
    ]
    for run_idx in range(1, args.runs):
        lines.append(f"run0_vs_run{run_idx}: first_mismatch="
                     f"{mismatches[0][0] if mismatches else 'none'}")
    if mismatches:
        name, diffs = mismatches[0]
        lines.append(f"first divergent module: {name} max_abs={diffs}")
        lines.append(f"divergent modules: {len(mismatches)}")
        for name, diffs in mismatches[:10]:
            lines.append(f"  {name} {diffs}")
    passed = not mismatches
    lines.append(f"RESULT: {'PASS' if passed else 'FAIL'}")

    for line in lines:
        print(line)
    log_dir = os.path.dirname(os.path.abspath(args.log))
    os.makedirs(log_dir, exist_ok=True)
    with open(args.log, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"log -> {args.log}")


if __name__ == "__main__":
    main()
