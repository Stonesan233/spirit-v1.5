"""Stage-wise performance benchmark for Spirit-v1.5 (CUDA / Ascend NPU / CPU).

Also hosts the shared deterministic-input helpers used by precision_test.py and
dit_bisect.py:

  build a fixed CPU batch/noise -> batch tensor dict -> device
  --save-inputs/--inputs share the exact same inputs across devices.

Usage:
  python perf_bench.py --ckpt /home/Spirit-v1.5 --backbone /home/qwen3vl-4b-small \
      --device auto --warmup 5 --iters 30 --save-inputs artifacts/inputs.npz \
      --out artifacts/perf_npu.json
"""

import argparse
import contextlib
import json
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

try:  # registers torch.npu on Ascend; harmless on CUDA/CPU
    import torch_npu  # noqa: F401
except Exception:
    pass

from model.modeling_spirit_vla import SpiritVLAPolicy

IMAGE_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
INPUT_KEYS = ("img_high", "img_left", "img_right")
TASK_TEXT = "move objects into the box"
ROBOT_TYPE = "Franka"


def pick_device(name: str = "auto") -> torch.device:
    if name and name != "auto":
        return torch.device(name)
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "npu":
        torch.npu.synchronize()


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "npu":
        idx = device.index if device.index is not None else torch.npu.current_device()
        try:
            return torch.npu.get_device_name(idx)
        except Exception:
            return f"npu:{idx}"
    return "cpu"


def cast_bf16(policy: torch.nn.Module) -> None:
    norm_prefixes = ("normalize_inputs", "normalize_targets", "unnormalize_outputs")
    for name, module in policy.named_children():
        if name.startswith(norm_prefixes):
            continue
        module.to(torch.bfloat16)


def make_inputs(batch_size: int, seed: int, n_steps: int, max_action_dim: int) -> dict:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    imgs = [torch.rand(batch_size, 3, 240, 320, generator=gen) for _ in INPUT_KEYS]
    state = torch.rand(batch_size, 1, 14, generator=gen)
    noise_gen = torch.Generator(device="cpu").manual_seed(seed + 12345)
    noise = torch.randn(batch_size, n_steps, max_action_dim, generator=noise_gen)
    inputs = dict(zip(INPUT_KEYS, imgs))
    inputs["state"] = state
    inputs["noise"] = noise
    inputs["meta"] = {
        "task": TASK_TEXT,
        "robot_type": ROBOT_TYPE,
        "batch": int(batch_size),
        "seed": int(seed),
        "n_action_steps": int(n_steps),
        "max_action_dim": int(max_action_dim),
    }
    return inputs


def save_inputs(path: str, inputs: dict) -> None:
    payload = {}
    for key, value in inputs.items():
        if key == "meta":
            payload[key] = np.array([json.dumps(value)])
        elif isinstance(value, torch.Tensor):
            payload[key] = value.detach().cpu().numpy()
        else:
            payload[key] = np.array([value])
    out_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(out_dir, exist_ok=True)
    np.savez(path, **payload)


def load_inputs(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    inputs = {key: torch.from_numpy(data[key]).float() for key in INPUT_KEYS + ("state", "noise")}
    meta = {}
    if "meta" in data:
        meta = json.loads(str(data["meta"][0]))
    inputs["meta"] = meta
    return inputs


def inputs_to_batch(inputs: dict, device: torch.device) -> dict:
    meta = inputs.get("meta", {})
    task = meta.get("task", TASK_TEXT)
    robot_type = meta.get("robot_type", ROBOT_TYPE)
    batch_size = inputs["img_high"].shape[0]
    return {
        IMAGE_KEYS[0]: inputs["img_high"].to(device),
        IMAGE_KEYS[1]: inputs["img_left"].to(device),
        IMAGE_KEYS[2]: inputs["img_right"].to(device),
        "observation.state": inputs["state"].to(device),
        "task": [task] * batch_size,
        "robot_type": [robot_type] * batch_size,
    }


def load_policy(ckpt: str, backbone: str, device: torch.device, bf16: bool = False) -> SpiritVLAPolicy:
    policy = SpiritVLAPolicy.from_pretrained(ckpt, backbone=backbone)
    policy.eval()
    policy.to(device)
    if bf16:
        cast_bf16(policy)
    return policy


def autocast_ctx(device: torch.device, enabled: bool = True):
    if enabled:
        return torch.autocast(device.type, dtype=torch.bfloat16)
    return contextlib.nullcontext()


def infer(policy, batch, noise, device, use_autocast=True):
    with torch.inference_mode(), autocast_ctx(device, use_autocast):
        return policy.select_action(batch, noise=noise.clone())


def memory_stats(device: torch.device):
    if device.type == "cuda":
        return (torch.cuda.memory_allocated() / 2**30,
                torch.cuda.max_memory_allocated() / 2**30,
                torch.cuda.max_memory_reserved() / 2**30)
    if device.type == "npu":
        alloc = torch.npu.memory_allocated() / 2**30
        peak_alloc = torch.npu.max_memory_allocated() / 2**30
        try:
            peak_reserved = torch.npu.max_memory_reserved() / 2**30
        except Exception:
            peak_reserved = None
        return (alloc, peak_alloc, peak_reserved)
    return (None, None, None)


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    elif device.type == "npu":
        torch.npu.reset_peak_memory_stats()


def run_stage_timing(policy, batch, noise, device, use_autocast=True, samples=3):
    acc = {}
    for _ in range(samples):
        with torch.inference_mode(), autocast_ctx(device, use_autocast):
            sync(device)
            t0 = time.perf_counter()
            normed = policy.normalize_targets(policy.normalize_inputs(batch))
            state = policy.prepare_state(normed)
            sync(device)
            t1 = time.perf_counter()
            policy.preprocess_rb_batch(normed)
            sync(device)
            t2 = time.perf_counter()
            vlm = policy._encode_vision(normed)
            sync(device)
            t3 = time.perf_counter()
            policy._sample_actions_unified(state, vlm, noise=noise.clone())
            sync(device)
            t4 = time.perf_counter()
        acc.setdefault("prepare_ms", []).append((t1 - t0) * 1000.0)
        acc.setdefault("preprocess_cpu_ms", []).append((t2 - t1) * 1000.0)
        acc.setdefault("encode_vision_ms", []).append((t3 - t2) * 1000.0)
        acc.setdefault("denoise_ms", []).append((t4 - t3) * 1000.0)
    return {key: float(np.mean(vals)) for key, vals in acc.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="/home/Spirit-v1.5")
    parser.add_argument("--backbone", default="/home/qwen3vl-4b-small")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "npu", "cpu"])
    parser.add_argument("--bf16", action="store_true", help="cast non-norm modules to bf16")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--stages", type=int, default=3, help="stage-timing sample count")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-inputs", default=None, help="write shared inputs npz")
    parser.add_argument("--inputs", default=None, help="load shared inputs npz")
    parser.add_argument("--out", "--json", dest="out", default=None)
    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"device={device} ({device_name(device)}) bf16_weights={args.bf16}")
    print(f"loading policy from {args.ckpt} (backbone: {args.backbone}) ...")
    policy = load_policy(args.ckpt, args.backbone, device, args.bf16)

    if args.inputs:
        inputs = load_inputs(args.inputs)
        print(f"loaded inputs: {args.inputs}")
    else:
        inputs = make_inputs(args.batch, args.seed,
                             policy.config.n_action_steps, policy.config.max_action_dim)
        if args.save_inputs:
            save_inputs(args.save_inputs, inputs)
            print(f"saved inputs: {args.save_inputs}")

    batch = inputs_to_batch(inputs, device)
    noise = inputs["noise"].to(device)

    for _ in range(args.warmup):
        infer(policy, batch, noise, device)
    sync(device)

    reset_peak_memory(device)
    latencies = []
    for _ in range(args.iters):
        sync(device)
        start = time.perf_counter()
        infer(policy, batch, noise, device)
        sync(device)
        latencies.append((time.perf_counter() - start) * 1000.0)
    lat = np.asarray(latencies)
    latency = {
        "mean_ms": float(lat.mean()),
        "std_ms": float(lat.std()),
        "p50_ms": float(np.percentile(lat, 50)),
        "p95_ms": float(np.percentile(lat, 95)),
        "p99_ms": float(np.percentile(lat, 99)),
        "min_ms": float(lat.min()),
        "max_ms": float(lat.max()),
        "throughput_hz": float(1000.0 / lat.mean()),
    }

    stages = {}
    if args.stages > 0:
        stages = run_stage_timing(policy, batch, noise, device, samples=args.stages)

    alloc, peak_alloc, peak_reserved = memory_stats(device)
    import transformers
    import diffusers

    payload = {
        "device": str(device),
        "device_name": device_name(device),
        "bf16_weights": bool(args.bf16),
        "autocast": True,
        "batch": inputs["state"].shape[0],
        "warmup": args.warmup,
        "iters": args.iters,
        "latency": latency,
        "stages_ms": stages,
        "memory_allocated_gib": alloc,
        "peak_alloc_gib": peak_alloc,
        "peak_reserved_gib": peak_reserved,
        "num_params": int(sum(p.numel() for p in policy.parameters())),
        "weight_dtype_qwen": str(next(policy.qwen.parameters()).dtype),
        "weight_dtype_qwen_text": str(next(policy.qwen.model.language_model.parameters()).dtype),
        "weight_dtype_qwen_visual": str(next(policy.qwen.model.visual.parameters()).dtype),
        "weight_dtype_dit": str(next(policy.dit.parameters()).dtype),
        "target_15fps_p95_ms": 66.7,
        "meets_15fps_p95": bool(latency["p95_ms"] < 66.7),
        "inputs": {"loaded": args.inputs, "saved": args.save_inputs},
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "diffusers": diffusers.__version__,
        "platform": os.uname().machine if hasattr(os, "uname") else "unknown",
    }

    print()
    print(f"latency: mean={latency['mean_ms']:.2f}ms p50={latency['p50_ms']:.2f}ms "
          f"p95={latency['p95_ms']:.2f}ms p99={latency['p99_ms']:.2f}ms "
          f"std={latency['std_ms']:.2f}ms")
    print(f"throughput: {latency['throughput_hz']:.2f} Hz (15fps p95 gate: "
          f"{'PASS' if payload['meets_15fps_p95'] else 'FAIL'})")
    if stages:
        print("stages (mean): " + ", ".join(f"{k}={v:.2f}" for k, v in stages.items()))
    if peak_alloc is not None:
        print(f"memory: allocated={alloc:.2f} GiB peak_alloc={peak_alloc:.2f} GiB "
              f"peak_reserved={peak_reserved:.2f} GiB")
    print(f"params: {payload['num_params'] / 1e9:.2f} B, dtype qwen={payload['weight_dtype_qwen']} "
          f"dit={payload['weight_dtype_dit']}")

    if args.out:
        out_dir = os.path.dirname(os.path.abspath(args.out))
        os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"json -> {args.out}")


if __name__ == "__main__":
    main()
