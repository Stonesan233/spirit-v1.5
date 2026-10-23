# Spirit-v1.5 昇腾 NPU 迁移适配需求文档

> 版本: v1.0 (2026-10-22)  
> 状态: 推理链路已在目标硬件冒烟验证通过；精度对齐与性能优化待做  
> 适用对象: 需要在外网环境复现本适配的工程师

---

## 1. 背景与目标

Spirit-v1.5 是 Spirit AI 开源的 VLA（Vision-Language-Action）机器人操作模型：  
**Qwen3-VL-4B-Instruct 骨干 + DiT flow-matching 动作头**，输入多视角图像 + 状态 + 任务描述，输出 60 步动作块（Euler 10 步去噪）。

原始代码仅支持 NVIDIA GPU（torch 2.8 + CUDA + flash-attn 可选）。本文档描述将其迁移到  
**华为昇腾 910B**（推理，边缘部署形态）与 **910C A3**（训练，预埋）的完整适配方案。

**迁移原则：**

1. 代码最小改动——只改"后端胶水层"（设备字符串、AMP、分布式后端），模型结构与数值逻辑零改动；
2. 推理优先——先在单卡 910B 上跑通并验证精度，再启动训练侧；
3. 离线优先——目标机无外网，所有物料（权重、小文件、wheel、镜像）离线携带。

---

## 2. 目标环境（已实测验证）

### 2.1 硬件

| 项 | 值 |
| --- | --- |
| NPU | Ascend **910B4** × 1（PCIE 形态，32GB HBM） |
| 设备节点 | **`/dev/davinci5`**（注意：不是 davinci0）+ `/dev/davinci_manager` + `/dev/devmm_svm` + `/dev/hisi_hdc` |
| NPU 序号 | npu-smi 显示为 NPU 5 |

### 2.2 宿主机软件

| 项 | 值 |
| --- | --- |
| OS | openEuler 22.03 LTS-SP4 (aarch64)，kernel 5.10.0-216.0.0.115.oe2203sp4 |
| NPU 驱动 | **25.3.rc1.1**（`/usr/local/Ascend/driver/version.info`） |
| NPU 固件 | 25.2.0（binary 7.7.0.6.236） |
| Docker | docker-engine 18.09.0.300 + docker-runc 1.0.0.rc3-300（openEuler 22.03 rpm） |

> **驱动兼容性实测结论：CANN 9.1.0 与驱动 25.3.rc1.1 兼容，无需升级驱动。**  
> 固件 25.2.0 的 `compatible_version_drv` 声明上限为 25.2.x，与驱动 25.3.rc1.1 存在轻微版本偏差，  
> 实测 torch_npu 初始化与推理均正常；量产环境建议将固件对齐到 25.3.x（如可获取）。

### 2.3 容器镜像

`quay.io/ascend/vllm-ascend:v0.26.0rc2`（17.8GB），内含（安装 wheel 前）：

| 包 | 镜像内置 | 覆盖后（实际使用） |
| --- | --- | --- |
| CANN | 9.1.0（ascend-toolkit） | 9.1.0 |
| torch | 2.10.0+cpu | 2.10.0+cpu |
| torch_npu | 2.10.0.post4 | 2.10.0.post4 |
| torchvision | 0.25.0+cpu | 0.25.0+cpu |
| transformers | 5.14.1 | **4.57.1** |
| huggingface_hub | 1.31.0 | **0.36.2** |
| diffusers | **无** | **0.35.2** |
| triton / triton_ascend | 3.5.0 / 3.2.2 | 同左 |
| vllm / vllm_ascend | 0.26.0 / 0.1.dev4525 | 降级 hub 后 vllm 可能不可用——**本任务不用 vllm serve，可接受** |
| 其他 | numpy 1.26.4, pillow 12.3.0, safetensors 0.8.0, scipy 1.13.1, einops 0.8.2, tokenizers 0.22.2, opencv-python-headless 5.0.0.93 | 同左 |

> 选镜像的理由：CANN 与 torch_npu 版本由镜像保证匹配，免去版本考古。  
> transformers 必须降回 4.57.1：Spirit-v1.5 代码基于 4.57 的 `Qwen3VLForConditionalGeneration`；  
> hub 必须降到 0.36.2：transformers 4.57.1 要求 `huggingface_hub<1.0`。

---

## 3. 离线物料清单

在有网环境准备以下物料，整体拷贝到目标机 `/home`：

### 3.1 模型权重（21GB）

```
Spirit-v1.5/
├── config.json # 模型配置（含 attention_implementation: "eager"）
├── model.safetensors # 21GB fp32，含全部 qwen.* + dit.* + proj.* + 归一化 buffer
└── README.md
```

来源: HuggingFace `Spirit-AI-robotics/Spirit-v1.5`

> **不需要**单独下载 Qwen3-VL-4B-Instruct 的权重（约 8GB）：checkpoint 已含完整 `qwen.*`，  
> `load_state_dict` 会整体覆盖。只需 3.2 的小文件。

### 3.2 Qwen3-VL-4B-Instruct 小文件（约 12MB）

```
qwen3vl-4b-small/
├── config.json # 模型结构配置（离线建结构用）
├── generation_config.json
├── chat_template.json
├── tokenizer.json # ~7MB
├── tokenizer_config.json # use_fast=False 时仍建议携带
├── vocab.json # 慢分词器必需
├── merges.txt # 慢分词器必需
├── preprocessor_config.json # 图像预处理器
├── video_preprocessor_config.json # Qwen3VLProcessor 构造需要
└── model.safetensors.index.json # 可选，未使用
```

下载命令（新版 hf CLI 用**位置参数** patterns，不要混用 `--include`，有坑）：

```bash
export HF_ENDPOINT=https://hf-mirror.com # 内网环境: http://mirrors.tools.huawei.com/huggingface
export HF_HUB_DOWNLOAD_TIMEOUT=10000
hf download Qwen/Qwen3-VL-4B-Instruct "*.json" "merges.txt" "*.jinja" "*.model" \
--local-dir ./qwen3vl-4b-small
```

### 3.3 Python wheel（约 17MB，全部 py3-none-any）

| wheel | 用途 |
| --- | --- |
| transformers-4.57.1-py3-none-any.whl | 覆盖镜像内 5.14.1 |
| huggingface_hub-0.36.2-py3-none-any.whl | 覆盖镜像内 1.31.0（4.57.1 强依赖 hub<1.0） |
| diffusers-0.35.2-py3-none-any.whl | DiT 头依赖（镜像内缺失） |

```bash
pip download transformers==4.57.1 --no-deps -d wheels/ -i <pypi镜像>
pip download huggingface_hub==0.36.2 --no-deps -d wheels/ -i <pypi镜像>
pip download diffusers==0.35.2 --no-deps -d wheels/ -i <pypi镜像>
```

### 3.4 Docker（若宿主机无 docker）

openEuler 22.03 aarch64 rpm：`docker-engine-18.09.0-300`、`docker-runc-1.0.0.rc3-300`。  
安装与启动：

```bash
rpm -ivh docker-runc-1.0.0.rc3-300.oe2203.aarch64.rpm
rpm -ivh docker-engine-18.09.0-300.oe2203.aarch64.rpm
systemctl enable --now docker
```

（docker-compose 需 10+ 个 python3 依赖、docker-client-java 需 maven jar，均非必需，可跳过。）

### 3.5 适配后的代码

见第 4 节改动明细；完整仓库（本仓库 `NPU_PORTING.md` 所在目录）即适配后代码。

---

## 4. 代码改动明细

共 7 个文件：6 处修改 + 1 个新增。**模型结构、DiT、flow-matching、数据管线零改动。**

### 4.1 `model/modeling_spirit_vla.py`（3 处）

**① 配置默认设备**

```python
# SpiritVLAConfig
- device: str = "cuda"
+ device: str = "npu"
```

**② backbone 离线构造**（`__init__` 中，替换原 `Qwen3VLForConditionalGeneration.from_pretrained(...)`）

```python
# Backbone + DiT head (flattened; state_dict prefix: model.*)
# If backbone is a local dir (offline edge box), build the backbone
# structurally from its config: real weights are restored afterwards
# from the Spirit checkpoint via load_state_dict, so no HF weight
# download is needed.
if os.path.isdir(config.backbone):
    from transformers import Qwen3VLConfig

    qwen_config = Qwen3VLConfig.from_pretrained(
        config.backbone, attn_implementation=config.attention_implementation
    )
    self.qwen = Qwen3VLForConditionalGeneration(qwen_config)
else:
    self.qwen = Qwen3VLForConditionalGeneration.from_pretrained(
        config.backbone,
        attn_implementation=config.attention_implementation,
        dtype=torch.float32,
    )
self.image_processor = AutoProcessor.from_pretrained(self.config.backbone).image_processor
```

原理：`SpiritVLAPolicy.from_pretrained` 随后用 checkpoint 做整体 `load_state_dict`，  
HF 上下载的权重本来就会被覆盖，因此只需 config/tokenizer/processor 小文件即可离线实例化。  
注意此处是**普通 CPU 随机初始化**（约 1 分钟），刻意不用 meta device——  
保持 `load_state_dict` 的 copy 语义，避免 strict=False 时残留 meta 张量的坑。

**③ `from_pretrained` 增加 backbone 覆盖参数**

```python
@classmethod
def from_pretrained(cls, ckpt_path: str, strict: bool = True, train: bool = False,
                    backbone: Optional[str] = None):
    ...
    config = SpiritVLAConfig(**filtered_cfg)
    if backbone is not None:
        config.backbone = backbone
```

**未改动但需说明：**

- FA2 自动启用块（`from flash_attn import ...`）：NPU 上 Import 失败走 except 分支，  
  且 checkpoint 配置 `attention_implementation: "eager"` 不含 "flash_attention"，不会 raise，保持原样；
- `safe_load_file(weight_path, device=load_device)`：ckpt 配置 `device: "cuda"` 在 NPU 机上  
  `torch.cuda.is_available()` 为 False，自动落 CPU 加载，随后调用方 `.to(npu)`，保持原样。

### 4.2 `robochallenge/runner/executor.py`（3 处）

```python
+ def _pick_device() -> torch.device:
+     if hasattr(torch, "npu") and torch.npu.is_available():
+         return torch.device("npu")
+     if torch.cuda.is_available():
+         return torch.device("cuda")
+     return torch.device("cpu")

- device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # ×2 处
+ device = _pick_device()  # _prepare_batch 与 RoboChallengeExecutor.__init__ 各一处

- if self.device.type == "cuda"
+ if self.device.type in ("cuda", "npu")  # torch.autocast 条件
```

### 4.3 `utils/distributed.py`（训练侧，预埋）

```python
+ _IS_NPU = hasattr(torch, "npu")
+ _DEV = torch.npu if _IS_NPU else torch.cuda
+ _BACKEND = "hccl" if _IS_NPU else "nccl"
+ _DEVICE_NAME = "npu" if _IS_NPU else "cuda"

dist.init_process_group(backend=_BACKEND)          # 原 "nccl"
_DEV.set_device(local_rank)                        # 原 torch.cuda.set_device
mesh = init_device_mesh(_DEVICE_NAME, (world_size,))  # 原 "cuda"
device_id=_DEV.current_device(),                   # FSDP 参数，原 torch.cuda.current_device()
```

### 4.4 `train.py`（训练侧，预埋）

```python
use_npu = hasattr(torch, "npu")
device = torch.device("npu" if use_npu else "cuda", local_rank)
...
pin_memory=not use_npu,  # NPU 无 pin_memory 收益
...
bf16_ok = torch.npu.is_bf16_supported() if use_npu else torch.cuda.is_bf16_supported()
amp_dtype = torch.bfloat16 if bf16_ok else torch.float16
dev_name = "npu" if use_npu else "cuda"
scaler = torch.amp.GradScaler(dev_name) if use_scaler else None
...
with torch.autocast(dev_name, dtype=amp_dtype):
```

### 4.5 `utils/normalization.py`

```python
if hasattr(torch, "npu"):
    bc_device = torch.device("npu", torch.npu.current_device())
else:
    bc_device = torch.device("cuda", torch.cuda.current_device())
dist.broadcast_object_list(object_list, src=0, device=bc_device)
```

### 4.6 `utils/logger.py`（边缘机无 wandb，改为可选）

```python
try:
    import wandb
except ImportError:
    wandb = None

# Logger.__init__ 中:
if rank == 0:
    if wandb is None:
        print("wandb not installed; wandb logging disabled")
        return
    self.run = wandb.init(...)
```

### 4.7 新增 `npu_smoke_test.py`（完整源码）

```python
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
```

### 4.8 明确未改动清单

| 项 | 说明 |
| --- | --- |
| 模型结构 / DiT / flow-matching | 零改动 |
| attention 实现 | 沿用 ckpt 的 `eager`，不引入 flash-attn / npu_fusion_attention |
| 归一化 / 采样 / 推理 API | 零改动 |
| `utils/sampling.py` 的 `Beta.sample` | 仅训练路径使用（`_standard_gamma` 在 NPU 可能回退 CPU），训练阶段再处理 |

---

## 5. 部署与运行步骤

### 5.1 目录布局（目标机 /home）

```
/home/
├── Spirit-v1.5/               # 3.1 权重
├── qwen3vl-4b-small/          # 3.2 小文件
├── wheels/                    # 3.3 三个 wheel
├── spirit-run/spirit-v1.5/    # 3.5 适配后代码
├── v0.26.0rc2.tar.gz          # 容器镜像 tar
└── docker-rpms/               # 3.4（仅宿主机无 docker 时）
```

### 5.2 加载镜像与安装依赖

```bash
docker load -i /home/v0.26.0rc2.tar.gz
# 依赖安装随容器启动执行（见 5.3），三个 wheel 必须 --no-deps 一次装齐：
# pip install --no-deps huggingface_hub-0.36.2...whl transformers-4.57.1...whl diffusers-0.35.2...whl
```

### 5.3 冒烟测试（完整 docker run）

```bash
docker run --rm --name spirit_smoke \
  --device /dev/davinci5 --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /home/spirit-run:/home/spirit-run \
  -v /home/Spirit-v1.5:/home/Spirit-v1.5:ro \
  -v /home/qwen3vl-4b-small:/home/qwen3vl-4b-small:ro \
  -v /home/wheels:/home/wheels:ro \
  -w /home/spirit-run/spirit-v1.5 \
  --entrypoint bash quay.io/ascend/vllm-ascend:v0.26.0rc2 -c "
pip install --no-deps /home/wheels/huggingface_hub-0.36.2-py3-none-any.whl \
  /home/wheels/transformers-4.57.1-py3-none-any.whl \
  /home/wheels/diffusers-0.35.2-py3-none-any.whl &&
python npu_smoke_test.py --ckpt /home/Spirit-v1.5 --backbone /home/qwen3vl-4b-small
"
```

**预期输出（实测）：**

```
0.36.2 4.57.1 0.35.2
loading policy from /home/Spirit-v1.5 (backbone: /home/qwen3vl-4b-small) ...
action shape: (1, 60, 14)
finite: True
first step: [ 4.17e-02 1.80e-03 -7.53e-02 -1.83e-01 -9.39e-02 ... ]
last step:  [-0.0348 0.0075 0.0223 ... ]
```

（另有无害警告：`NPUCachingAllocator ... 32 padding size`。）

运行模式说明：fp32 权重（21GB）+ bf16 autocast，32GB HBM 实测不 OOM。  
边缘量产建议改用 `--bf16`（权重减半至 ~10.5GB，需重测精度）。

### 5.4 环境固化（待办）

pip 安装在 `--rm` 容器内，退出即失。验证通过后执行：

```bash
# 去掉 --rm 跑一次安装，然后:
docker commit <container_id> spirit-npu:v1
```

---

## 6. 精度验证方案（迁移质量门槛）

**核心原则：固定一切非硬件变量，只让硬件不同。**

1. **控制变量**
   - 噪声：CPU 上固定种子生成（`torch.Generator(device="cpu")`），`.to(device)` 后经  
     `select_action(batch, noise=noise)` 显式传入两平台。禁止设备端 RNG；
   - 代码：GPU 参考机跑**同一份适配代码**（`_pick_device()` 自动落 cuda）+ 同版本 transformers/diffusers；
   - attention：两边均 `eager`。

2. **分层比对与建议阈值**

| 层级 | 指标 | 阈值 |
| --- | --- | --- |
| pixel_values（CPU 预处理） | bitwise | 完全一致 |
| vlm_last_embed | cos / rel-L2 | > 0.999 / < 2% |
| 每步 v_t（×10） | cos | > 0.999 |
| 归一化 action | max abs diff | < 0.05 |
| 反归一化后 60 步积分 | 末端位置 / 旋转 / 夹爪 | < 1cm / < 0.05rad / < 0.05 |

实现方式：`precision_test.py`（--device npu/cuda，--dump-stages 存 npz 中间量）+ `compare_precision.py`（指标计算与判定）。待编写。

3. **附加检查**
   - 确定性：NPU 同输入跑两遍，diff 应为 0；
   - fp32+autocast 与 `--bf16` 两种模式分别测，bf16 模式阈值放宽约 2 倍；
   - 行为验证（终审）：NPU 版直接跑 RoboChallenge / 仿真，比成功率。

---

## 7. 训练侧预埋说明（910C A3，未启动）

已完成：`hccl` 后端自适应、`init_device_mesh("npu")`、autocast/GradScaler 设备自适应、  
`pin_memory=False`、FSDP `device_id` 自适应。

训练启动前需额外验证/处理：

1. **FSDP on NPU**：`MixedPrecision(fp32 param)` + 大量 `ignored_modules`（LayerNorm/RMSNorm 等）  
   需实测；`use_orig_params=True`、`sync_module_states=True` 行为确认；
2. **`sample_beta`（`_standard_gamma`）**：疑似 NPU 无算子会回退 CPU，建议改为 CPU 侧生成再 `.to(device)`；
3. **动态 shape**：Ascend 图编译对变长敏感，`_encode_vision` 的按 batch padding 建议固定到常数长度；  
   `preprocess_rb_batch` 的逐样本 CPU↔NPU 往返（PIL 预处理）是吞吐瓶颈，需 profiling；
4. **A3 SuperPoD**：HCCL 拓扑/多机通信配置（RANK_TABLE 等）与单机不同，另行处理；
5. 数据侧 `scipy.Rotation`、ColorJitter 均为 CPU 逻辑，无需改动。

---

## 8. 验收标准 Checklist

- [x] 容器内 `import torch_npu`，`torch.npu.device_count() == 1`
- [x] 冒烟测试：action shape (1, 60, 14)，全部有限值，无 OOM
- [x] 驱动/CANN 兼容性确认（无需升驱动）
- [ ] 容器 commit 固化
- [ ] 精度对齐（第 6 节三层全部通过）
- [ ] RoboChallenge 真任务复跑（`robochallenge/runner/executor.py` 已适配 NPU）
- [ ] 延迟达标（RoboChallenge 15fps 约束下 `select_action` 端到端计时）
- [ ] A3 训练链路跑通

---

## 9. 附：踩坑记录

| 坑 | 现象 / 结论 |
| --- | --- |
| `pgrep -f` 自匹配 | 远端 ssh 会话命令行含关键字导致误判"进程已在跑"，改用 `docker ps --format` 判断 |
| 新版 `hf` CLI | `--include` 与位置参数 patterns 混用会被忽略 include，只下载到 1 个文件；应全部用位置参数 |
| `tr -d "\r"` 跨 ssh | 反斜杠经 Windows ssh 客户端传递时丢失，变成删字母 `r`；跨机传脚本避免依赖转义，用 base64 或 heredoc |
| 嵌套 ssh 引号 | 三层以上嵌套引号极易被 PowerShell/ssh 吃掉；统一用 `base64 -d` |
| 镜像内 hub 1.x | transformers 4.57.1 必须搭配 hub<1.0，三个 wheel 需一次 `--no-deps` 装齐 |
| 固件/驱动版本偏差 | 固件 25.2.0 兼容声明上限 25.2.x < 驱动 25.3.rc1.1，实测可用；量产建议对齐 |
