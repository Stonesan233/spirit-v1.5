# Spirit-v1.5 RTX 4090 48G 测试报告

| 项 | 值 |
| --- | --- |
| 报告版本 | v1.0 |
| 日期 | 2026-10-26 |
| 代码基线 | `ascend-npu-porting` @ `7382626` |
| 测试对象 | `Spirit-v1.5`（base）<br>`Spirit-v1.5-for-RoboChallenge-move-objects-into-box`（微调） |
| 参考文档 | `Spirit-v1.5-Ascend-NPU-Porting.md` |
| 状态 | 推理链路、RoboChallenge 接口、性能与离线精度已完成；官方真机成功率待线上提交 |

---

## 1. 目标与范围

本报告验证 Spirit-v1.5 在单卡 **NVIDIA RTX 4090（显存改装 48GB）** 上的：

1. 推理正确性与数值确定性；
2. 与 RoboChallenge 官方推理 API 的对接（mock 回放）；
3. 延迟、吞吐、显存；
4. 数值精度对齐（fp32 / bf16 权重）；
5. 零样本能力（LIBERO 仿真、RoboChallenge 录制离线动作打分）；
6. 微调权重 `move_objects_into_box` 的效果。

**不在范围内**：RoboChallenge 官方真机成功率（需要线上 token / run_id 提交）、训练吞吐与收敛。

---

## 2. 测试环境

### 2.1 硬件

| 项 | 值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4090，49140 MiB（改装 48GB） |
| CPU | 16 核 |
| 内存 | 62 GB |
| 磁盘 | 916 GB（可用 470 GB） |
| 平台 | Linux x86_64 |

### 2.2 软件栈

系统自带 torch 复用，其余依赖装在独立 venv（`/home/spirit-run/venv`，`--system-site-packages`），系统 Python 未被修改。

| 包 | 版本 |
| --- | --- |
| Python | 3.10.12 |
| torch / torchvision | 2.12.0+cu130 / 0.27.0+cu130 |
| transformers | 4.57.1 |
| huggingface_hub | 0.36.2 |
| diffusers | 0.35.2 |
| numpy / scipy | 1.26.4 / 1.15.3 |
| pillow / safetensors | 10.4.0 / 0.7.0 |
| opencv-python-headless | 5.0.0.93（声明需 numpy>=2，实测可用） |
| accelerate | 1.15.0 |

仿真与接口测试额外依赖（仅测试环境）：

| 包 | 版本 |
| --- | --- |
| robosuite / mujoco / bddl | 1.4.0 / 2.3.7 / 1.0.1 |
| gym | 0.25.2 |
| fastapi / uvicorn / httpx / loguru | 0.162.0 / 0.54.0 / 0.28.1 / — |

### 2.3 物料与目录

| 物料 | 路径 | 说明 |
| --- | --- | --- |
| 代码 | `/home/spirit-run/spirit-v1.5` | `ascend-npu-porting` 分支 |
| RC 客户端 | `/home/spirit-run/RoboChallengeInference` | 官方仓库副本 + Spirit 适配 |
| base 权重 | `/home/Spirit-v1.5` | 21,603,726,776 B |
| 微调权重 | `/home/Spirit-v1.5-move-box` | 21,603,726,776 B |
| Qwen 小文件 | `/home/qwen3vl-4b-small` | 约 12 MB（config/tokenizer/preprocessor） |
| 测试产物 | `/home/spirit-run/results` | npz / json |

权重经 ModelScope 下载（目标机无法访问 HuggingFace），文件大小与源站一致。

### 2.4 模型配置（checkpoint）

`n_action_steps=60`、`max_state_dim=max_action_dim=32`、`num_steps=10`（Euler）、`attention_implementation=eager`、DiT hidden 1536 / num_heads 32 / num_layers 16、`dit_cross_attention_dim=2560`、动作维度 14；参数量约 **5.00B**（4,999,362,164）。

---

## 3. 接口与集成测试（RoboChallenge mock）

### 3.1 方法

使用官方 `RoboChallengeInference` 的 mock server 回放录制 episode，Spirit 作为策略侧通过 HTTP API 拉取 `state.pkl` 并 `POST /action`。图像请求 `320×240`（宽×高），`duration=1/15`，每次下发一个 60 步 chunk。

> mock server 为录播回放，机器人侧忽略下发动作，因此本节只验证接口契约、动作形状与延迟，**不代表任务成功率**。

### 3.2 结果

| 任务 / 录制 | 机器人 | action_type | 动作形状 | 推理延迟（稳态 / 冷启动） | 结果 |
| --- | --- | --- | --- | --- | --- |
| press_three_buttons | Franka | `leftpos` | 60 × 8 | 318 ms / 641 ms | 通过 |
| pour_fries_into_plate | aloha | `pos` | 60 × 16 | 345 ms / 653 ms | 通过 |
| arrange_paper_cups | ARX5 | `leftpos` | 60 × 7 | 334 ms / 668 ms | 通过 |
| shred_scrap_paper | UR5 | `leftjoint` | 60 × 7 | 317 ms / 705 ms | 通过 |

- 四种机器人的位姿/关节映射全部走通，`post_action` 无形状或队列错误。
- 动作形状与 RoboChallenge 文档一致（Franka 位姿 8 = xyz + quat + gripper；aloha 双臂 16；ARX5 7；UR5 关节 7）。
- 说明：由于该仓库顶层 `utils/` 与 Spirit 的 `utils/` 同名冲突，集成脚本使用 Spirit 内 vendored 的同源官方客户端（逻辑一致），mock server 用官方原版。

---

## 4. 数值精度

### 4.1 确定性

同一输入（CPU 固定种子 batch + noise）连续跑两次，逐阶段比对：

| 阶段 | 结果 |
| --- | --- |
| `pixel_values` | bitwise 相同 |
| `vlm_last_embed` | cos = 1.0，rel-L2 = 0 |
| 每步 `v_t`（10 步） | cos = 1.0，max-abs = 0 |
| 归一化动作 / 最终动作 | max-abs = 0 |

**结论：同设备重复运行 bitwise 确定。**

### 4.2 fp32 权重 vs bf16 权重（阈值按文档 ×2 放宽）

| 指标 | 实测 | 阈值（×2） | 判定 |
| --- | --- | --- | --- |
| `pixel_values` | bitwise 相同 | — | 通过 |
| `vlm_last_embed` cos | 0.996554 | > 0.999 | 不通过 |
| `vlm_last_embed` rel-L2 | 8.30% | < 4% | 不通过 |
| `v_t` min cos | 0.990932 | > 0.999 | 不通过 |
| 归一化动作 max-abs | 0.175781 | < 0.1 | 不通过 |
| 最终动作 max-abs | 0.443409 | — | 夹爪维度最差（0.443） |

**结论：**

- `--bf16`（权重转 bf16）量化损失显著，即使放宽 2 倍仍不达标，**不建议用于精度敏感场景**；
- 迁移对齐应使用 **fp32 权重 + bf16 autocast**；
- 环境从 numpy 2.2.6 降到 1.26.4 后，与旧参考 dump 比对仍为 **bitwise 0 差异**，参考数据有效。

---

## 5. 性能

### 5.1 端到端（batch=1，3 路 240×320，10 步去噪，eager attention，warmup=5 / iters=30）

| 模式 | mean | p50 | p95 | p99 | 吞吐 | 15fps（p95<66.7ms） |
| --- | --- | --- | --- | --- | --- | --- |
| fp32 权重 + bf16 autocast | 283.18 ms | 283.23 ms | 283.71 ms | 283.82 ms | 3.53 Hz | 不达标 |
| bf16 权重 | 262.88 ms | 262.79 ms | 263.49 ms | 264.38 ms | 3.80 Hz | 不达标 |

### 5.2 分阶段耗时（均值）

| 阶段 | fp32 | bf16 |
| --- | --- | --- |
| 归一化/状态准备 | 0.35 ms | 0.35 ms |
| CPU 图像预处理（PIL） | 14.33 ms | 14.24 ms |
| VLM 前向（含预处理） | 114.90 ms | 116.44 ms |
| DiT 去噪（10 步） | 164.66 ms | 141.59 ms |

### 5.3 显存

| 模式 | 常驻分配 | 峰值分配 | 保留 |
| --- | --- | --- | --- |
| fp32 | 11.30 GiB | 11.49 GiB | 20.35 GiB |
| bf16 | 9.45 GiB | 9.64 GiB | 20.35 GiB（含 fp32 加载缓存） |

**分析：**

- 瓶颈是 eager VLM（约 115 ms）与 DiT 每步约 14–16 ms × 10；
- 15fps 门槛需约 4× 加速；可选优化：sdpa/flash attention（需重验精度）、`torch.compile` DiT、减少去噪步数/蒸馏、CPU 预处理与 GPU 重叠；
- 60 步 chunk 按 3.5–3.8 Hz 推理摊薄，开环约 210–230 动作步/秒；是否满足 15fps 取决于推理调用节奏（整块推理 vs 15Hz 重推理）。

---

## 6. 零样本能力（参考）

### 6.1 LIBERO 仿真

`libero_spatial` task 0/1/2，init state 0，最多 120 步，每 8 步重规划：

| task | 成功 | EEF 位移 | 推理次数 | mean / p95 延迟 |
| --- | --- | --- | --- | --- |
| 0 | 否 | 0.82 m | 15 | 305.4 / 403.6 ms |
| 1 | 否 | 0.55 m | 15 | 300.1 / 371.1 ms |
| 2 | 否 | 0.52 m | 15 | 296.8 / 368.1 ms |

0/3 成功，机械臂有运动但动作分布明显越界（位置/旋转大量饱和，夹爪几乎常开）。**Spirit-v1.5 训练于 RoboChallenge 真机数据，与 LIBERO 存在域差异，此结果仅为链路验证，不作为能力评分。**

### 6.2 RoboChallenge 录制 episode 离线动作打分

方法：把录制 episode 的每帧观测喂给模型，将预测的绝对位姿/关节与录制中真实未来状态比对；30fps 录像按 `stride=2` 对齐 15Hz 控制，horizon 60 步，每 episode 抽 6 帧；并给出“保持当前位姿”基线（同一协议）。

| 任务 / 机器人 | 指标 | base 模型 | 保持不动基线 |
| --- | --- | --- | --- |
| press_three_buttons / Franka | 位置 MAE | 11.62 cm（首步 5.91） | 10.07 cm |
| | 姿态 MAE | 13.66°（首步 14.69） | 2.12° |
| pour_fries_into_plate / aloha | 位置 MAE | 8.74 cm（首步 2.37） | 5.65 cm |
| | 姿态 MAE | 24.57°（首步 4.46） | 18.21° |
| arrange_paper_cups / ARX5 | 位置 MAE | 13.67 cm（首步 13.85） | 9.75 cm |
| | 姿态 MAE | 21.60°（首步 32.53） | 6.74° |
| shred_scrap_paper / UR5 | 关节 MAE | 0.171 rad（首步 0.094） | 0.114 rad |

**结论：base 模型在四组评测轨迹上均未跑赢“保持不动”弱基线，零样本不可直接用于真机评测。**

> 该指标是离线模仿代理，不是官方成功率；未提供 `raw_embodiment_stats.json`，夹爪尺度不可比。

---

## 7. 微调权重：`move_objects_into_box`

### 7.1 权重

来源：HuggingFace `Spirit-AI-robotics/Spirit-v1.5-for-RoboChallenge-move-objects-into-box`，ModelScope 有同步副本。结构与 base 一致（21.6 GB），`strict=True` 可直接加载，无需修改模型代码。其 config 的训练相机名 `main/handeye/side_realsense_rgb` 与 `TASK_INFO["move_objects_into_box"]` 的 `right_hand/left_hand/high` 映射顺序一致。

### 7.2 评测数据

从 HF 数据集 `RoboChallenge/task_table30_move_objects_into_box`（11.6 GB split-tar）用 HTTP Range 仅抽取 `episode_000000`（1352 帧，约 45 s，17 MB）。

### 7.3 离线动作打分（同一 episode、6 采样帧、horizon 60、stride 2）

| 指标 | base | **微调** | 保持不动基线 |
| --- | --- | --- | --- |
| 位置 MAE | 17.93 cm | **12.17 cm** | 17.92 cm |
| 首步位置误差 | 5.95 cm | **1.00 cm** | 0 |
| 姿态 MAE | 18.49° | **4.83°** | 6.81° |
| 首步姿态误差 | 26.95° | **0.47°** | 0 |
| 夹爪 MAE | 0.826 | **0.0103** | 0.012 |

- 微调模型全面跑赢静态基线；首步接近完美跟踪（1.00 cm / 0.47°）。
- 长时域位置误差仍有 12.17 cm（60 步开环累积漂移）。

### 7.4 RoboChallenge mock 接口

| 项 | 结果 |
| --- | --- |
| action_type | `leftpos`（Franka） |
| 动作形状 | **40 × 8**（自动套用 `TASKS_USE_LESS_CHUNK_SIZE=40`，夹爪二值化阈值 0.055） |
| 下发 | 全部成功 |
| 延迟 | 稳态 343 ms，冷启动 751 ms |

---

## 8. 结论

1. **推理正确性**：4090 上链路完整，同设备 bitwise 确定；数值逻辑与模型结构零改动。
2. **精度模式**：迁移/对齐必须使用 fp32 权重 + bf16 autocast；bf16 权重不满足文档阈值。
3. **性能**：fp32 约 283 ms/次（3.53 Hz），bf16 约 263 ms/次（3.80 Hz）；15fps 门槛未达，需 4× 左右优化。
4. **RoboChallenge 接口**：四机型（Franka/aloha/ARX5/UR5）动作形状与下发全部通过；mock 不打任务分。
5. **零样本**：base 在 LIBERO 与四组 RC 录制轨迹上均不可用（弱于静态基线）。
6. **微调有效**：`move_objects_into_box` 微调权重在对应录制 episode 上显著优于 base 与静态基线（首步 1 cm / 0.47°，长时域 12.17 cm / 4.83°），说明 **RoboChallenge 需要按任务微调**。
7. **官方分数**：仍需线上真机提交（`spirit_demo.py --user_token --run_id`）后由平台给出。

---

## 9. 复现与产物

### 9.1 产物

| 内容 | 路径 |
| --- | --- |
| 精度参考 dump | `/home/spirit-run/results/ref_cuda_fp32.npz` 等 |
| 精度比对 | `compare_determinism.json`、`compare_fp32_vs_bf16.json` |
| 性能 | `perf_cuda_fp32.json`、`perf_cuda_bf16.json` |
| LIBERO | `libero_spatial_t{0,1,2}_fp32.json` |
| 离线动作打分 | `offline_action_eval_stride2.json`、`offline_movebox_{base,finetuned}.json` |
| mock 接口 | `spirit_mock_{franka,aloha,arx5,ur5,movebox_finetuned}.json` |

### 9.2 主要命令

```bash
# 精度参考（GPU）
python precision_test.py --ckpt /home/Spirit-v1.5 --backbone /home/qwen3vl-4b-small \
  --device cuda --dump results/ref_cuda_fp32.npz
python compare_precision.py --ref results/ref_cuda_fp32.npz --test <other>.npz

# 性能
python benchmark_perf.py --ckpt /home/Spirit-v1.5 --backbone /home/qwen3vl-4b-small \
  --device cuda --warmup 5 --iters 30 --json results/perf_cuda_fp32.json

# LIBERO
LIBERO_ROOT=/home/spirit-run/LIBERO python libero_eval.py \
  --ckpt /home/Spirit-v1.5 --backbone /home/qwen3vl-4b-small \
  --suite libero_spatial --task-id 0 --init-ids 0 --max-steps 120

# RoboChallenge mock
cd RoboChallengeInference/mock_server
MOCK_ROBOT_TAG=franka MOCK_RECORD_DATA_DIR=../20250919/press_three_buttons/data/episode_FFFFFF \
  python mock_robot_server.py
cd ..
SPIRIT_ROOT=/home/spirit-run/spirit-v1.5 python test_spirit.py \
  --checkpoint /home/Spirit-v1.5 --backbone /home/qwen3vl-4b-small --single_task press_three_buttons

# 离线动作打分（base / 微调）
SPIRIT_ROOT=/home/spirit-run/spirit-v1.5 python offline_action_eval.py \
  --checkpoint /home/Spirit-v1.5 --backbone /home/qwen3vl-4b-small \
  --tag franka --task move_objects_into_box \
  --record_dir /home/spirit-run/RoboChallengeInference/20250919/move_objects_into_box/data/episode_000000
```

> 说明：`precision_test.py`、`compare_precision.py`、`benchmark_perf.py`、`libero_eval.py` 及 RC 侧的 `spirit_policy.py`、`spirit_demo.py`、`test_spirit.py`、`offline_action_eval.py` 目前位于工作区，**尚未提交**（按项目决定暂缓）。本报告只记录测试方法与结果。

### 9.3 已知限制

- 离线动作打分仅取少量采样帧/单个 episode，且公开 episode 可能与微调训练集重叠，分数可能偏乐观；官方分数以真机提交为准。
- LIBERO 为零样本跨域结果，不构成能力结论。
- bf16 权重模式未达精度阈值，量产使用前需重标定/重训或继续保持 fp32 权重 + bf16 autocast。
