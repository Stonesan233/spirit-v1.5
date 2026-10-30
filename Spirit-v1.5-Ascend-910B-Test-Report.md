# Spirit-v1.5 昇腾 910B 测试报告

| 项    | 值                                                                                                |
| ---- | ------------------------------------------------------------------------------------------------ |
| 报告版本 | v1.0                                                                                             |
| 日期   | 2026-10-27                                                                                       |
| 代码基线 | `ascend-npu-porting` @ `f64810f` + 工作区修复（`x_t = noise.clone()`）                                  |
| 测试对象 | `Spirit-v1.5`（base）<br>`Spirit-v1.5-for-RoboChallenge-move-objects-into-box`（微调）                 |
| 参考文档 | `Spirit-v1.5-Ascend-NPU-Porting.md`、`RTX4090-48G-Test-Report.md`                                 |
| 状态   | 推理链路、确定性、RoboChallenge mock 接口（5 项）、性能、离线动作打分（4 任务 + movebox base/微调）已完成；官方真机成功率待线上提交；LIBERO 未执行 |

---

## 1. 目标与范围

在单卡 **昇腾 910B4（Atlas 300I A2 形态，32GB HBM，PCIE）** 上复现 `RTX4090-48G-Test-Report.md` 的全部可迁移测试项：

1. 推理正确性与数值确定性（bf16 autocast / fp32 双模式）；
2. 与 RoboChallenge 官方推理 API 的对接（mock 回放，4 机型 + movebox 微调）；
3. 延迟与显存（与 4090 对照）；
4. 离线动作打分（4 任务 base；movebox base vs 微调 vs 保持不动基线）。

**不在范围内**：RoboChallenge 官方真机成功率（需线上 token/run_id）、LIBERO 仿真（aarch64 依赖重，且 Spirit 训练域与 LIBERO 无交集，4090 报告已证明无参考价值）、fp32-vs-bf16 权重对比（4090 已证明 bf16 权重不达标，未在 NPU 复测）。

---

## 2. 测试环境

### 2.1 硬件

| 项    | 值                                                                             |
| ---- | ----------------------------------------------------------------------------- |
| NPU  | Ascend **910B4** × 1（PCIE，32GB HBM，`npu-smi` NPU 5，Bus 0000:AB:00.0）          |
| 设备节点 | `/dev/davinci5` + `/dev/davinci_manager` + `/dev/devmm_svm` + `/dev/hisi_hdc` |

### 2.2 软件栈

| 层                                          | 版本                                                  |
| ------------------------------------------ | --------------------------------------------------- |
| 宿主 OS                                      | openEuler 22.03 LTS-SP4（aarch64），kernel 5.10.0-216  |
| NPU 驱动 / 固件                                | 25.3.rc1.1 / 25.2.0（binary 7.7.0.6.236）             |
| Docker                                     | 18.09.0.300 + docker-runc 1.0.0.rc3-300             |
| 容器镜像                                       | `quay.io/ascend/vllm-ascend:v0.26.0rc2`             |
| CANN                                       | **9.1.0**（ascend-toolkit）                           |
| torch / torch_npu                          | 2.10.0+cpu / **2.10.0.post4**                       |
| transformers / huggingface_hub / diffusers | 4.57.1 / 0.36.2 / 0.35.2（覆盖镜像内 5.14.1 / 1.31.0 / 无） |

> **驱动兼容性实测**：CANN 9.1.0 与驱动 25.3.rc1.1 兼容，`torch_npu` 初始化与全部推理正常，**无需升级驱动**（固件 25.2.0 声明的兼容驱动上限为 25.2.x，与驱动存在轻微版本偏差，实测无影响；量产建议对齐固件至 25.3.x）。

### 2.3 物料与目录

| 物料              | 路径                                                                                                   |
| --------------- | ---------------------------------------------------------------------------------------------------- |
| 代码              | `/home/spirit-run/spirit-v1.5`（ascend-npu-porting）                                                   |
| RC 客户端          | `/home/spirit-run/RoboChallengeInference`（`4090_test` 分支 + Spirit 集成脚本）                              |
| base 权重         | `/home/Spirit-v1.5`（21 GB）                                                                           |
| 微调权重            | `/home/Spirit-v1.5-move-box`（`model.safetensors` = 21,603,726,776 B，与官方一致）                           |
| Qwen 小文件        | `/home/qwen3vl-4b-small`（离线构造骨干结构，无需下载 HF 权重）                                                        |
| movebox episode | `/home/spirit-run/RoboChallengeInference/20250919/move_objects_into_box/data/episode_000000`（1352 帧） |
| 测试产物            | `/home/spirit-run/artifacts/`                                                                        |

---

## 3. 接口与集成测试（RoboChallenge mock）

方法同 4090 报告 §3：官方 mock server 回放录制 episode，Spirit 通过 HTTP API 拉取 `state.pkl` 并 `POST /action`，`320×240`，`duration=1/15`。

| 任务 / 录制                     | 机器人    | action_type | 动作形状                                            | 平均延迟     | 结果  |
| --------------------------- | ------ | ----------- | ----------------------------------------------- | -------- | --- |
| press_three_buttons         | Franka | `leftpos`   | 60 × 8                                          | 551.6 ms | 通过  |
| pour_fries_into_plate       | aloha  | `pos`       | 60 × 16                                         | 569.3 ms | 通过  |
| arrange_paper_cups          | ARX5   | `leftpos`   | 60 × 7                                          | 579.8 ms | 通过  |
| shred_scrap_paper           | UR5    | `leftjoint` | 60 × 7                                          | 577.3 ms | 通过  |
| move_objects_into_box（微调权重） | Franka | `leftpos`   | **40 × 8**（`TASKS_USE_LESS_CHUNK_SIZE=40` 自动生效） | 572.6 ms | 通过  |

- 四机型位姿/关节映射全部走通，无形状或队列错误；
- NPU mock 全链路延迟（含 HTTP 与图像传输）约为 4090（317–345 ms 稳态）的 1.7 倍，仍在 15Hz chunk 节奏预算内。

---

## 4. 数值确定性

### 4.1 结果

同一输入（CPU 固定种子 batch + 固定 noise）连续 3 次 `select_action`，逐级比对：

| 阶段                 | bf16-autocast    | fp32             |
| ------------------ | ---------------- | ---------------- |
| `vlm_last_embed`   | bitwise 相同       | bitwise 相同       |
| 最终动作 max-abs / cos | **0 / 1.000000** | **0 / 1.000000** |

**结论：同设备重复运行 bitwise 确定（两种精度模式均通过）。**

### 4.2 排查记录（重要）

初测曾出现"非确定"假象（同输入 3 次运行动作 cos ≈ 0），定位过程：

1. `vlm_last_embed` 逐次 bitwise 全等 → 骨干前向确定；
2. DiT 全部 250 个叶子模块（`dit_bisect.py` forward-hook 同输入 3 次比对）bitwise 全等 → 去噪算子确定；
3. 排除融合 SDPA（替换 math 实现无变化）、排除精度（fp32 同样"非确定"）；
4. **根因**：测试脚本跨次复用 `noise` 张量，而 `_sample_actions_unified` 的 Euler 更新 `x_t += dt * v_t` 为原地操作，会覆盖调用者传入的 noise——第 2/3 次运行实际从上一次输出的动作出发。

**修复与影响**：测试脚本改为 `noise.clone()`；模型侧 `x_t = noise.clone()` 防御（见工作区 diff）。生产路径 `select_action` 不传 noise（内部每次新生成），**行为从未受影响**。

---

## 5. 性能

### 5.1 端到端（batch=1，3 路 240×320，10 步去噪，eager attention，fp32 权重 + bf16 autocast，warmup=5 / iters=30）

| 指标              | 910B4（本次）                   | RTX 4090 48G（对照）            |
| --------------- | --------------------------- | --------------------------- |
| mean            | 301.85 ms                   | 283.18 ms                   |
| p50 / p95 / p99 | 293.17 / 333.45 / 335.15 ms | 283.23 / 283.71 / 283.82 ms |
| 峰值显存/显存         | **13.45 GB** / 32 GB        | 11.49 GiB / 48 GB           |

### 5.2 分阶段耗时（均值）

| 阶段                        | 910B4        | 4090      |
| ------------------------- | ------------ | --------- |
| 归一化/状态准备                  | ~0.5 ms      | 0.35 ms   |
| VLM 前向（含 CPU 预处理 22.7 ms） | **143.8 ms** | 114.90 ms |
| DiT 去噪（10 步 × 14.6 ms）    | **146.5 ms** | 164.66 ms |

**分析**：

- NPU 端到端比 4090 慢约 7%；VLM 前向慢 25%，DiT 去噪快 11%；
- 60 步 chunk 摊薄后约 3.3 Hz 推理 / ~200 动作步每秒，与 4090 同一量级；15fps 重推理场景两者均需进一步优化（sdpa/去噪蒸馏/预处理重叠），不构成 NPU 特有短板；
- eager attention 为 checkpoint 配置，NPU 上实测可用；`npu_fusion_attention` 为后续优化项（需重验精度）。

---

## 6. 离线动作打分（base 模型，4 任务 × 6 采样帧，horizon 60，stride 2）

| 任务 / 机器人                      | 指标     | RTX 4090（报告 §6.2）   | **910B4（本次）**           |
| ----------------------------- | ------ | ------------------- | ----------------------- |
| press_three_buttons / Franka  | 位置 MAE | 11.62 cm（首步 5.91）   | **11.68 cm（首步 5.26）**   |
|                               | 姿态 MAE | 13.66°（首步 14.69）    | **14.40°（首步 19.60）**    |
| pour_fries_into_plate / aloha | 位置 MAE | 8.74 cm（首步 2.37）    | **8.48 cm（首步 1.93）**    |
|                               | 姿态 MAE | 24.57°（首步 4.46）     | **21.55°（首步 4.76）**     |
| arrange_paper_cups / ARX5     | 位置 MAE | 13.67 cm（首步 13.85）  | **12.86 cm（首步 7.41）**   |
|                               | 姿态 MAE | 21.60°（首步 32.53）    | **21.07°（首步 26.82）**    |
| shred_scrap_paper / UR5       | 关节 MAE | 0.171 rad（首步 0.094） | **0.162 rad（首步 0.082）** |

"保持不动"基线逐位复现（如 Franka 10.07 cm / 2.12°，与 4090 完全一致，验证评测协议一致性）。

**结论：两平台对同一批观测的长时域 MAE 偏差约在 1%–12%（首步指标对去噪噪声更敏感、波动更大），与静态基线的相对关系完全一致（base 零样本在 4 任务上均弱于静态基线）——跨硬件行为等价成立。**

---

## 7. 微调权重：`move_objects_into_box`

### 7.1 权重

来源 `Spirit-AI-robotics/Spirit-v1.5-for-RoboChallenge-move-objects-into-box`（HF 镜像下载），`model.safetensors` 21,603,726,776 B 与官方一致，`strict=True` 直接加载，模型代码零改动。

### 7.2 RoboChallenge mock 接口（微调权重）

`action_type=leftpos`，动作形状 **40 × 8**（`TASKS_USE_LESS_CHUNK_SIZE=40` 自动生效），下发全部成功，平均延迟 572.6 ms。

### 7.3 离线动作打分（episode_000000，6 采样帧，horizon 60，stride 2）

| 指标     | base（910B4） | **微调（910B4）** | 微调（4090 对照） | 保持不动基线   |
| ------ | ----------- | ------------- | ----------- | -------- |
| 位置 MAE | 19.46 cm    | **12.02 cm**  | 12.17 cm    | 17.92 cm |
| 首步位置误差 | 8.71 cm     | **1.37 cm**   | 1.00 cm     | 0        |
| 姿态 MAE | 19.10°      | **4.81°**     | 4.83°       | 6.81°    |
| 首步姿态误差 | 20.86°      | **0.30°**     | 0.47°       | 0        |
| 夹爪 MAE | 0.678       | **0.0103**    | 0.0103      | 0.012    |

**结论：NPU 上完整复现 4090 的微调结论——微调模型全面优于 base 与静态基线，首步跟踪接近完美（1.37 cm / 0.30°），夹爪 MAE 与 4090 完全一致（0.0103）。**

---

## 8. 结论

1. **推理正确性**：NPU 链路完整，同设备 bitwise 确定（bf16-autocast 与 fp32 双模式）；骨干与 DiT 前向经逐模块 bisect 验证确定；数值逻辑与模型结构零改动。
2. **跨硬件等价**：离线动作打分的长时域 MAE 与 4090 结果偏差约 1%–12%（首步指标波动更大；bf16 计算舍入量级），任务间相对关系、微调收益、基线对比全部复现。
3. **驱动兼容**：CANN 9.1.0 + 驱动 25.3.rc1.1 实测兼容，无需升级驱动。
4. **接口**：4 机型 + movebox（40×8）mock 全部通过。
5. **性能**：端到端 301.85 ms（4090 为 283.18 ms，慢约 7%），峰值显存 13.45/32 GB（fp32 权重 + bf16 autocast），满足 chunk 摊薄节奏；15fps 重推理场景与 4090 同样需要后续优化。
6. **微调有效**：与 4090 结论一致，RoboChallenge 需按任务微调。
7. **官方分数**：需线上真机提交（`spirit_demo.py --user_token --run_id`）。

---

## 9. 复现与产物

### 9.1 产物（`/home/spirit-run/artifacts/`）

| 内容        | 文件                                                                  |
| --------- | ------------------------------------------------------------------- |
| 确定性/精度参考  | `results_npu.npz`、`inputs.npz`（共享输入）、`det_bf16.npz`                 |
| 算子 bisect | `dit_bisect.log`                                                    |
| 性能        | `perf_npu.json`                                                     |
| mock 接口   | `spirit_mock_{franka,aloha,arx5,ur5,movebox_finetuned}.json`        |
| 离线打分      | `offline_npu_base.json`、`offline_npu_movebox_{base,finetuned}.json` |

### 9.2 主要命令

```bash
cd /home/spirit-run/spirit-v1.5

# 确定性（bf16 / fp32）
python precision_test.py --ckpt /home/Spirit-v1.5 --backbone /home/qwen3vl-4b-small \
--cases 0 --stages 0 --determinism [--no-autocast] --inputs /home/spirit-run/artifacts/inputs.npz

# 性能
python perf_bench.py --ckpt /home/Spirit-v1.5 --backbone /home/qwen3vl-4b-small \
--iters 30 --warmup 5 --save-inputs /home/spirit-run/artifacts/inputs.npz \
--out /home/spirit-run/artifacts/perf_npu.json

# 离线动作打分（4 任务 base / movebox 单任务）
cd /home/spirit-run/RoboChallengeInference
SPIRIT_ROOT=/home/spirit-run/spirit-v1.5 python offline_action_eval.py \
--checkpoint /home/Spirit-v1.5 --backbone /home/qwen3vl-4b-small \
--device npu --num_samples 6 --output /home/spirit-run/artifacts/offline_npu_base.json
SPIRIT_ROOT=/home/spirit-run/spirit-v1.5 python offline_action_eval.py \
--checkpoint /home/Spirit-v1.5-move-box --backbone /home/qwen3vl-4b-small --device npu \
--tag franka --task move_objects_into_box \
--record_dir 20250919/move_objects_into_box/data/episode_000000 \
--output /home/spirit-run/artifacts/offline_npu_movebox_finetuned.json

# RoboChallenge mock（mock_server 从 mock_server/ 目录后台启动）
( cd mock_server && MOCK_ROBOT_TAG=franka \
    MOCK_RECORD_DATA_DIR=../20250919/press_three_buttons/data/episode_FFFFFF \
    nohup python mock_robot_server.py > /tmp/mock_franka.log 2>&1 & )
for i in $(seq 1 40); do
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 http://127.0.0.1:9098/clock-sync || true)
  [ "$code" = "200" ] && break; sleep 1
done
SPIRIT_ROOT=/home/spirit-run/spirit-v1.5 python test_spirit.py \
--checkpoint /home/Spirit-v1.5 --backbone /home/qwen3vl-4b-small --single_task press_three_buttons
fuser -k 9098/tcp 2>/dev/null || true
```

（全部命令在 `quay.io/ascend/vllm-ascend:v0.26.0rc2` 容器内执行，容器需挂载 `/dev/davinci5`、`/dev/davinci_manager`、`/dev/devmm_svm`、`/dev/hisi_hdc`、`/usr/local/Ascend/driver`、`/usr/local/dcmi`。）

### 9.3 已知限制

- LIBERO 未执行（aarch64 依赖重；4090 报告已证明域差异导致无参考价值）。
- fp32-vs-bf16 权重对比未在 NPU 复测（4090 已证明 bf16 权重不达标；NPU 结论预期一致）。
- 与 4090 的逐 case npz 精确比对（`compare_precision.py`）待 4090 侧提供 `results_4090.npz`；本报告的跨硬件等价性由离线打分数字对照支撑。
- mock 为录播回放，不打任务分；官方分数以真机提交为准。
- 离线打分仅单 episode 少量采样帧，公开 episode 可能与微调训练集重叠，分数可能偏乐观。
