"""Accelerator backend detection.

``torch.npu`` is not part of stock PyTorch: it exists only after ``torch_npu``
has been imported, because that import registers the namespace on ``torch``.
Code that inspects ``hasattr(torch, "npu")`` therefore depends on something
else in the process having imported it first - in practice the Ascend
container image, not this repository. Import it here so detection no longer
depends on the ambient environment.

The import stays optional: on hosts without the package it is skipped and the
helpers simply report that no NPU is usable.
"""

import torch

try:
    import torch_npu  # noqa: F401  (registers torch.npu)
except (ImportError, OSError):
    _TORCH_NPU_IMPORTED = False
else:
    _TORCH_NPU_IMPORTED = True


def npu_available() -> bool:
    """Whether an Ascend NPU can actually be used right now."""
    return _TORCH_NPU_IMPORTED and torch.npu.is_available()


def pick_device() -> torch.device:
    """First usable accelerator, in preference order: npu > cuda > cpu."""
    if npu_available():
        return torch.device("npu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_index() -> int:
    """Ordinal of the current device on the active accelerator."""
    if npu_available():
        return torch.npu.current_device()
    return torch.cuda.current_device()


def autocast(dtype: torch.dtype):
    """Autocast context for the active accelerator.

    ``torch.autocast("npu")`` only works on torch_npu builds that register the
    npu autocast backend, so the vendor namespace is used explicitly when an
    NPU is present.
    """
    if npu_available():
        return torch.npu.amp.autocast(enabled=True, dtype=dtype)
    return torch.autocast("cuda", dtype=dtype)


def grad_scaler(enabled: bool):
    """Gradient scaler for the active accelerator, or ``None`` when disabled.

    Ascend uses bf16 by default, so the scaler is normally unused; it is only
    constructed for the fp16 fallback path.
    """
    if not enabled:
        return None
    if npu_available():
        return torch.npu.amp.GradScaler(enabled=True)
    return torch.amp.GradScaler("cuda", enabled=True)


def bf16_supported() -> bool:
    """Whether the active accelerator supports bfloat16."""
    if npu_available():
        check = getattr(torch.npu, "is_bf16_supported", None)
        return True if check is None else check()
    return torch.cuda.is_bf16_supported()
