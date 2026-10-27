"""Deterministic precision sampling and determinism check for Spirit-v1.5.

Unified CLI for both the RTX 4090 flow (--device/--dump) and the Ascend NPU
flow (--cases/--stages/--determinism/--inputs/--out).

Fixed inputs/noise are generated on CPU (or loaded from --inputs) and passed
explicitly, so the same inputs can be replayed on another device and compared
with compare_precision.py.

Usage (4090):
  python precision_test.py --ckpt /home/Spirit-v1.5 --backbone /home/qwen3vl-4b-small \
      --device cuda --dump results/ref_cuda_fp32.npz

Usage (NPU determinism):
  python precision_test.py --ckpt /home/Spirit-v1.5 --backbone /home/qwen3vl-4b-small \
      --device npu --cases 0 --stages 0 --determinism \
      --inputs artifacts/inputs.npz --out artifacts/det_bf16.npz
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

from perf_bench import (
    autocast_ctx,
    cast_bf16,
    inputs_to_batch,
    load_inputs,
    load_policy,
    make_inputs,
    pick_device,
    save_inputs,
    sync,
)


def perturb_inputs(inputs: dict, index: int, seed: int) -> dict:
    if index == 0:
        return inputs
    gen = torch.Generator(device="cpu").manual_seed(seed + 1000 + index)
    out = dict(inputs)
    out["state"] = inputs["state"] + 0.01 * torch.randn(inputs["state"].shape, generator=gen)
    out["noise"] = inputs["noise"] + 0.01 * torch.randn(inputs["noise"].shape, generator=gen)
    out["meta"] = dict(inputs.get("meta", {}))
    return out


def patch_dit_eager() -> None:
    import torch.nn.functional as F

    def _math_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                   is_causal=False, scale=None, enable_gqa=False):
        scale = (query.shape[-1] ** -0.5) if scale is None else scale
        attn = torch.matmul(query, key.transpose(-2, -1)) * scale
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = torch.softmax(attn, dim=-1)
        return torch.matmul(attn, value)

    F.scaled_dot_product_attention = _math_sdpa


def run_case(policy, inputs: dict, device: torch.device, capture_stages: bool,
             use_autocast: bool) -> tuple:
    batch = inputs_to_batch(inputs, device)
    noise = inputs["noise"].to(device)
    caps = {}
    vts = []

    orig_encode = policy._encode_vision
    orig_denoise = policy._denoise_step
    orig_sample = policy._sample_actions_unified

    def encode_capture(b):
        out = orig_encode(b)
        if capture_stages:
            caps["vlm_last_embed"] = out.detach().float().cpu().numpy()
        return out

    def denoise_capture(state, vlm, x_t, timestep):
        out = orig_denoise(state, vlm, x_t, timestep)
        if capture_stages:
            vts.append(out.detach().float().cpu().numpy())
        return out

    def sample_capture(state, vlm, noise=None):
        out = orig_sample(state, vlm, noise=noise)
        if capture_stages:
            caps["norm_action"] = out.detach().float().cpu().numpy()
        return out

    policy._encode_vision = encode_capture
    policy._denoise_step = denoise_capture
    policy._sample_actions_unified = sample_capture
    try:
        sync(device)
        start = time.perf_counter()
        with torch.inference_mode(), autocast_ctx(device, use_autocast):
            actions = policy.select_action(batch, noise=noise.clone())
        sync(device)
        elapsed = time.perf_counter() - start
    finally:
        policy._encode_vision = orig_encode
        policy._denoise_step = orig_denoise
        policy._sample_actions_unified = orig_sample

    if capture_stages:
        pixel_list, grid_list, _, _ = policy.preprocess_rb_batch(batch)
        caps["pixel_values"] = torch.cat(pixel_list, dim=0).detach().float().cpu().numpy()
        caps["image_grid_thw"] = torch.cat(grid_list, dim=0).detach().cpu().numpy()
        caps["v_t"] = np.stack(vts, axis=0) if vts else np.zeros((0,))

    return actions.detach().float().cpu().numpy(), caps, elapsed


def diff_metrics(a, b) -> tuple:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    max_abs = float(np.max(np.abs(a - b)))
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    denom = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    cos = float(np.dot(a_flat, b_flat) / denom) if denom > 0 else float("nan")
    return max_abs, cos


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="/home/Spirit-v1.5")
    parser.add_argument("--backbone", default="/home/qwen3vl-4b-small")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "npu", "cpu"])
    parser.add_argument("--bf16", action="store_true", help="cast non-norm modules to bf16")
    parser.add_argument("--no-autocast", action="store_true", help="disable bf16 autocast")
    parser.add_argument("--dit-eager", action="store_true",
                        help="replace F.scaled_dot_product_attention with explicit math")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cases", type=int, default=1,
                        help="number of deterministic cases (0 -> 1 base case)")
    parser.add_argument("--stages", type=int, default=1,
                        help="dump vlm/v_t/pixel/norm tensors for the first N cases (0 -> none)")
    parser.add_argument("--determinism", action="store_true",
                        help="run every case 3 times and require bitwise-identical outputs")
    parser.add_argument("--inputs", default=None, help="load shared inputs npz")
    parser.add_argument("--save-inputs", default=None, help="save generated inputs npz")
    parser.add_argument("--out", "--dump", dest="out", default="results/precision_dump.npz")
    args = parser.parse_args()

    if args.dit_eager:
        patch_dit_eager()
        print("dit attention: explicit math SDPA")

    device = pick_device(args.device)
    use_autocast = not args.no_autocast
    print(f"device={device} bf16_weights={args.bf16} autocast={use_autocast}")
    print(f"loading policy from {args.ckpt} (backbone: {args.backbone}) ...")
    policy = load_policy(args.ckpt, args.backbone, device, args.bf16)
    if args.no_autocast and not args.bf16:
        # The Qwen config may declare dtype=bfloat16, which builds the text
        # backbone in bf16 even though the Spirit checkpoint is fp32. Pure fp32
        # mode must cast the whole policy, otherwise DiT (fp32) receives bf16
        # cross-attention inputs.
        policy.to(torch.float32)
        print("cast policy to float32 (no-autocast)")

    if args.inputs:
        base_inputs = load_inputs(args.inputs)
        print(f"loaded inputs: {args.inputs}")
    else:
        base_inputs = make_inputs(args.batch, args.seed,
                                  policy.config.n_action_steps, policy.config.max_action_dim)
        if args.save_inputs:
            save_inputs(args.save_inputs, base_inputs)
            print(f"saved inputs: {args.save_inputs}")

    n_cases = max(1, args.cases)
    n_stage = min(max(0, args.stages), n_cases)
    repeats = 3 if args.determinism else 1

    all_actions = []
    stage_norm, stage_vlm, stage_vt, stage_pix, stage_grid = [], [], [], [], []
    elapsed = []
    det_checks = []

    for case_idx in range(n_cases):
        inputs = perturb_inputs(base_inputs, case_idx, args.seed)
        capture = case_idx < n_stage
        runs = []
        for repeat in range(repeats):
            actions, caps, dt = run_case(policy, inputs, device, capture, use_autocast)
            runs.append((actions, caps))
            if repeat == 0:
                elapsed.append(dt)
        all_actions.append(runs[0][0])
        if capture:
            stage_norm.append(runs[0][1]["norm_action"])
            stage_vlm.append(runs[0][1]["vlm_last_embed"])
            stage_vt.append(runs[0][1]["v_t"])
            stage_pix.append(runs[0][1]["pixel_values"])
            stage_grid.append(runs[0][1]["image_grid_thw"])

        if args.determinism:
            for r in range(1, repeats):
                max_abs, cos = diff_metrics(runs[0][0], runs[r][0])
                det_checks.append({
                    "case": case_idx, "compare": f"run0_vs_run{r}",
                    "action_max_abs": max_abs, "action_cos": cos,
                })
                if capture:
                    vlm_abs, vlm_cos = diff_metrics(runs[0][1]["vlm_last_embed"],
                                                    runs[r][1]["vlm_last_embed"])
                    det_checks.append({
                        "case": case_idx, "compare": f"run0_vs_run{r}",
                        "vlm_max_abs": vlm_abs, "vlm_cos": vlm_cos,
                    })
                    if runs[0][1]["v_t"].size and runs[r][1]["v_t"].size:
                        vt_abs, vt_cos = diff_metrics(runs[0][1]["v_t"], runs[r][1]["v_t"])
                        det_checks.append({
                            "case": case_idx, "compare": f"run0_vs_run{r}",
                            "v_t_max_abs": vt_abs, "v_t_cos": vt_cos,
                        })

    def pack(values):
        if not values:
            return None
        if n_cases == 1 and len(values) == 1:
            return values[0]
        return np.stack(values, axis=0)

    result = {
        "action": pack(all_actions),
        "meta": np.array([json.dumps({
            "device": str(device),
            "bf16_weights": bool(args.bf16),
            "autocast": use_autocast,
            "dit_eager": bool(args.dit_eager),
            "seed": args.seed,
            "cases": n_cases,
            "stage_cases": n_stage,
            "batch": int(base_inputs["state"].shape[0]),
            "elapsed_s": elapsed,
            "determinism": det_checks or None,
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "diffusers": __import__("diffusers").__version__,
        })]),
    }
    if n_stage > 0:
        result["norm_action"] = pack(stage_norm)
        result["vlm_last_embed"] = pack(stage_vlm)
        result["v_t"] = pack(stage_vt)
        result["pixel_values"] = pack(stage_pix)
        result["image_grid_thw"] = pack(stage_grid)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    np.savez(args.out, **result)

    actions0 = all_actions[0]
    print(f"cases: {n_cases} (staged: {n_stage})")
    print(f"action shape: {tuple(actions0.shape)}")
    print(f"finite: {bool(np.isfinite(actions0).all())}")
    if n_stage > 0:
        print(f"num denoise steps: {stage_vt[0].shape[0]}")
    print(f"first step: {np.round(actions0[0, 0], 4).tolist()}")
    print(f"last step:  {np.round(actions0[0, -1], 4).tolist()}")
    print(f"select_action elapsed: {np.mean(elapsed) * 1000:.1f} ms")
    if args.determinism:
        for check in det_checks:
            keys = [k for k in ("action_max_abs", "vlm_max_abs", "v_t_max_abs") if k in check]
            for key in keys:
                print(f"det case={check['case']} {check['compare']} {key}={check[key]:.3e}")
        passed = all(
            check.get(key, 0.0) == 0.0
            for check in det_checks
            for key in ("action_max_abs", "vlm_max_abs", "v_t_max_abs")
        )
        print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    print(f"dumped -> {args.out}")


if __name__ == "__main__":
    main()
