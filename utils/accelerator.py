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
