"""Compare two precision dumps produced by precision_test.py.

Supports both the legacy single-case layout (no leading case dim) and the
multi-case layout produced with --cases/--stages.

Usage:
  python compare_precision.py --ref results_4090.npz --test results_npu.npz
  python compare_precision.py results_4090.npz results_npu.npz
"""

import argparse
import json

import numpy as np

# Porting-doc thresholds (section 6); physical thresholds follow the NPU guide.
TH_COS = 0.999
TH_REL_L2 = 0.02
TH_VT_COS = 0.999
TH_NORM_MAX = 0.05
TH_POS_M = 0.01
TH_ROT_RAD = 0.05
TH_GRIP = 0.05


def as_cases(array, kind):
    array = np.asarray(array)
    if kind in ("action", "norm"):
        if array.ndim == 3:
            return array[None]
        if array.ndim == 4:
            return array
    elif kind == "vlm":
        if array.ndim == 3:
            return array[None]
        if array.ndim == 4:
            return array
    elif kind == "v_t":
        if array.ndim == 4:
            return array[None]
        if array.ndim == 5:
            return array
    elif kind == "pixel":
        if array.ndim == 2:
            return array[None]
        if array.ndim == 3:
            return array
    elif kind == "grid":
        if array.ndim == 2:
            return array[None]
        if array.ndim == 3:
            return array
    return array


def _flat(array):
    return np.asarray(array, dtype=np.float64).reshape(-1)


def cosine(a, b):
    a_flat, b_flat = _flat(a), _flat(b)
    denom = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    return float(np.dot(a_flat, b_flat) / denom) if denom > 0 else float("nan")


def rel_l2(a, b):
    a_flat, b_flat = _flat(a), _flat(b)
    denom = np.linalg.norm(b_flat)
    return float(np.linalg.norm(a_flat - b_flat) / denom) if denom > 0 else float("nan")


def max_abs(a, b):
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def step_cosines(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    steps = min(len(a), len(b))
    out = []
    for i in range(steps):
        out.append(cosine(a[i], b[i]))
    return out


def case_metrics(ref_case, test_case, available, scale):
    metrics = {}
    checks = {}

    if "pixel_values" in available:
        r = np.asarray(ref_case["pixel_values"])
        t = np.asarray(test_case["pixel_values"])
        same = r.shape == t.shape and np.array_equal(r, t)
        metrics["pixel_bitwise"] = bool(same)
        checks["pixel_bitwise"] = same

    if "vlm_last_embed" in available:
        if np.asarray(ref_case["vlm_last_embed"]).shape == np.asarray(test_case["vlm_last_embed"]).shape:
            c = cosine(ref_case["vlm_last_embed"], test_case["vlm_last_embed"])
            r = rel_l2(ref_case["vlm_last_embed"], test_case["vlm_last_embed"])
            metrics["vlm_cos"] = c
            metrics["vlm_rel_l2"] = r
            checks["vlm_cos"] = c > TH_COS
            checks["vlm_rel_l2"] = r < TH_REL_L2 * scale
        else:
            metrics["vlm_cos"] = float("nan")
            metrics["vlm_rel_l2"] = float("nan")
            checks["vlm_cos"] = False
            checks["vlm_rel_l2"] = False

    if "v_t" in available:
        if np.asarray(ref_case["v_t"]).shape == np.asarray(test_case["v_t"]).shape:
            cosines = step_cosines(ref_case["v_t"], test_case["v_t"])
            metrics["v_t_min_cos"] = min(cosines) if cosines else float("nan")
            metrics["v_t_mean_cos"] = float(np.mean(cosines)) if cosines else float("nan")
            checks["v_t_min_cos"] = metrics["v_t_min_cos"] > TH_VT_COS
        else:
            metrics["v_t_min_cos"] = float("nan")
            checks["v_t_min_cos"] = False

    if "norm_action" in available:
        if np.asarray(ref_case["norm_action"]).shape == np.asarray(test_case["norm_action"]).shape:
            m = max_abs(ref_case["norm_action"], test_case["norm_action"])
            metrics["norm_max_abs"] = m
            checks["norm_max_abs"] = m < TH_NORM_MAX * scale
        else:
            metrics["norm_max_abs"] = float("inf")
            checks["norm_max_abs"] = False

    action_ref = np.asarray(ref_case["action"], dtype=np.float64)
    action_test = np.asarray(test_case["action"], dtype=np.float64)
    if action_ref.shape == action_test.shape:
        metrics["action_max_abs"] = max_abs(action_ref, action_test)
        metrics["action_cos"] = cosine(action_ref, action_test)
    else:
        metrics["action_max_abs"] = float("inf")
        metrics["action_cos"] = float("nan")
        checks["action_shape"] = False
    if action_ref.shape == action_test.shape and action_ref.shape[-1] >= 7:
        diff = np.abs(action_ref - action_test)
        pos = np.linalg.norm(diff[..., 0:3], axis=-1)
        rot = np.linalg.norm(diff[..., 3:6], axis=-1)
        grip = diff[..., 6]
        metrics["pos_max_m"] = float(pos.max())
        metrics["rot_max_rad"] = float(rot.max())
        metrics["grip_max"] = float(grip.max())
        checks["pos_max_m"] = metrics["pos_max_m"] < TH_POS_M * scale
        checks["rot_max_rad"] = metrics["rot_max_rad"] < TH_ROT_RAD * scale
        checks["grip_max"] = metrics["grip_max"] < TH_GRIP * scale

    metrics["pass"] = bool(all(checks.values()))
    metrics["checks"] = checks
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", help="ref test (positional form)")
    parser.add_argument("--ref", default=None)
    parser.add_argument("--test", default=None)
    parser.add_argument("--bf16", action="store_true", help="relax thresholds by 2x")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    ref_path = args.ref or (args.paths[0] if len(args.paths) > 0 else None)
    test_path = args.test or (args.paths[1] if len(args.paths) > 1 else None)
    if not ref_path or not test_path:
        parser.error("provide --ref/--test or two positional paths")

    ref = np.load(ref_path, allow_pickle=True)
    test = np.load(test_path, allow_pickle=True)
    ref_meta = json.loads(str(ref["meta"][0])) if "meta" in ref else {}
    test_meta = json.loads(str(test["meta"][0])) if "meta" in test else {}
    scale = 2.0 if args.bf16 else 1.0

    kinds = {
        "pixel_values": "pixel",
        "image_grid_thw": "grid",
        "vlm_last_embed": "vlm",
        "v_t": "v_t",
        "norm_action": "norm",
        "action": "action",
    }
    ref_cases = {key: as_cases(ref[key], kind) for key, kind in kinds.items() if key in ref}
    test_cases = {key: as_cases(test[key], kind) for key, kind in kinds.items() if key in test}
    available = {key for key in ref_cases if key in test_cases}
    n_cases = min(ref_cases["action"].shape[0], test_cases["action"].shape[0])

    print(f"ref : {ref_path} {ref_meta}")
    print(f"test: {test_path} {test_meta}")
    print(f"cases: ref={ref_cases['action'].shape[0]} test={test_cases['action'].shape[0]} "
          f"compared={n_cases} bf16_relax={args.bf16}")
    print()

    header = ["case", "pixel", "vlm_cos", "vlm_rel", "vt_min_cos", "norm_max",
              "pos_m", "rot_rad", "grip", "pass"]
    print(f"{header[0]:<5}{header[1]:<8}{header[2]:<10}{header[3]:<10}{header[4]:<12}"
          f"{header[5]:<10}{header[6]:<9}{header[7]:<9}{header[8]:<10}{header[9]:<6}")

    all_metrics = []
    for case_idx in range(n_cases):
        ref_case = {key: values[case_idx] for key, values in ref_cases.items()}
        test_case = {key: values[case_idx] for key, values in test_cases.items()}
        metrics = case_metrics(ref_case, test_case, available, scale)
        all_metrics.append({"case": case_idx, **metrics})

        def fmt(key, spec=".6f"):
            value = metrics.get(key)
            if value is None:
                return "-"
            if isinstance(value, bool):
                return str(value)
            return format(value, spec)

        print(f"{case_idx:<5}{fmt('pixel_bitwise'):<8}{fmt('vlm_cos'):<10}"
              f"{fmt('vlm_rel_l2'):<10}{fmt('v_t_min_cos'):<12}{fmt('norm_max_abs'):<10}"
              f"{fmt('pos_max_m'):<9}{fmt('rot_max_rad'):<9}{fmt('grip_max'):<10}"
              f"{'PASS' if metrics['pass'] else 'FAIL':<6}")

    overall = all(m["pass"] for m in all_metrics)
    worst = {}
    for key in ("vlm_cos", "v_t_min_cos"):
        values = [m[key] for m in all_metrics if key in m]
        if values:
            worst[key] = min(values)
    for key in ("vlm_rel_l2", "norm_max_abs", "pos_max_m", "rot_max_rad", "grip_max",
                "action_max_abs"):
        values = [m[key] for m in all_metrics if key in m]
        if values:
            worst[key] = max(values)
    print()
    print("worst:", {key: round(value, 8) for key, value in worst.items()})
    print(f"RESULT: {'PASS' if overall else 'FAIL'}")

    if args.json:
        payload = {
            "ref": ref_path,
            "test": test_path,
            "bf16_relax": bool(args.bf16),
            "ref_meta": ref_meta,
            "test_meta": test_meta,
            "cases_compared": n_cases,
            "per_case": all_metrics,
            "worst": worst,
            "overall": overall,
        }
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2, default=float)
        print(f"json -> {args.json}")

    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
