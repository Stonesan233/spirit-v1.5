# NPU 测试脚本与代码变更合入指南

| 项    | 值                                                                                    |
| ---- | ------------------------------------------------------------------------------------ |
| 文档版本 | v1.0                                                                                 |
| 日期   | 2026-10-27                                                                           |
| 合入基线 | `Stonesan233/spirit-v1.5` @ `ascend-npu-porting` 分支 `f64810f`                        |
| 产出依据 | `Spirit-v1.5-Ascend-910B-Test-Report.md`（全部 NPU 测试由以下代码版本产出）                          |
| 验证环境 | 910B4（Atlas 300I A2, 32G）+ CANN 9.1.0 + torch_npu 2.10.0.post4 + transformers 4.57.1 |

---

## 1. 变更总览

| #   | 文件                                                                  | 状态            | 性质          | 用途                                |
| --- | ------------------------------------------------------------------- | ------------- | ----------- | --------------------------------- |
| 1   | `model/modeling_spirit_vla.py`                                      | **修改**（+3/−1） | 生产代码（防御性修复） | 消除 Euler 更新原地覆盖调用者 noise 的隐患      |
| 2   | `perf_bench.py`                                                     | **新增**        | 测试脚本        | 分阶段性能基准（延迟/吞吐/显存）                 |
| 3   | `precision_test.py`                                                 | **新增**        | 测试脚本        | 数值确定性检查 + 固定输入/噪声精度采样             |
| 4   | `compare_precision.py`                                              | **新增**        | 测试脚本        | 跨设备精度比对（物理量误差 + 判定）               |
| 5   | `dit_bisect.py`                                                     | **新增**        | 诊断脚本        | 算子级确定性 bisect（forward-hook 逐模块比对） |
| 6   | `robochallenge/test_examples/Spirit-v1.5-Ascend-910B-Test-Report.md` | **新增**        | 文档          | NPU 测试报告                          |

**不应合入**：`assets/rc_results.png` 的本地改动（LFS smudge 伪差异，保持仓库原样即可）、`.vscode/`。

---

## 2. 生产代码修改：`model/modeling_spirit_vla.py`

**唯一的生产代码改动，+3/−1，位于 `_sample_actions_unified` 内（约 667 行处）：**

```diff
vlm_last_embed = self.proj_vlm_output(vlm_last_embed)
dt = -1.0 / self.config.num_steps
dt = torch.tensor(dt, dtype=torch.float32, device=device)
- x_t = noise
+ # Do not alias the caller's noise tensor: the Euler update below is
+ # in-place (x_t += dt * v_t) and would otherwise silently overwrite it.
+ x_t = noise.clone()
time = torch.tensor(1.0, dtype=torch.float32, device=device)
```

**背景**：`x_t = noise` 使 `x_t` 与调用者传入的 `noise` 张量共享存储，后续 Euler 更新
`x_t += dt * v_t` 为原地写，会静默覆盖调用者的 noise。生产路径（`RoboChallengeExecutor` →
`select_action` 不传 noise）内部每次新生成张量，**行为从未受影响**；此修复只保护显式传
noise 的调用方（本测试套件即是）。已验证：mock / 离线打分 / smoke 在含此修复的代码上
结果与不含修复的 4090 结果一致。

---

## 3. 新增文件说明

### 3.1 `perf_bench.py` — 分阶段性能基准

```text
用法（容器内）:
python perf_bench.py --ckpt /home/Spirit-v1.5 --backbone /home/qwen3vl-4b-small \
--iters 30 --warmup 5 [--bf16] [--save-inputs inputs.npz] [--inputs inputs.npz] \
[--out perf_npu.json]
```

- 设备自适应（npu → cuda → cpu），计时前后强制 `torch.npu.synchronize()`；
- 对 `normalize_inputs` / `preprocess_rb_batch` / `_encode_vision` / `_sample_actions_unified` /
`_denoise_step` 挂计时 wrapper（注意：nn.Module 子模块需包其 `forward`，不能直接覆盖属性名）；
- 输出 total 与各阶段的 mean/p50/p95/p99/min/max + 峰值显存，JSON 落盘；
- `--save-inputs` 生成固定种子的共享输入 npz，供跨设备使用（4090 侧 `--inputs` 消费同一文件）。

### 3.2 `precision_test.py` — 确定性 + 精度采样

```text
用法:
python precision_test.py --ckpt ... --backbone ... [--inputs inputs.npz] \
[--determinism] [--no-autocast] [--dit-eager] [--bf16] \
[--cases 15 --stages 3 --out results_npu.npz]
```

- **`--determinism`**：同输入同 noise 连跑 3 次，输出 pairwise max-abs / 动作 cos /
`vlm_last_embed` 比对 / 逐步 `v_t` 比对（判别发散来源的关键工具）；
- **noise 处理**：CPU 固定种子生成，每次运行传 `noise.clone()`（配合 §2 的模型修复）；
- `--cases N`：N 个确定性扰动 case（固定种子），保存最终动作 + 前 K 个 case 的
`vlm_last_embed` / 逐步 `v_t`（fp16）供 `compare_precision.py` 跨设备比对；
- `--no-autocast`：纯 fp32 模式（排查精度相关性）；`--dit-eager`：全局把
`F.scaled_dot_product_attention` 替换为显式 math 实现（排查融合注意力）。

> 注意：本版本参数与 4090 工作区早期版本（`--device/--dump`）不同，报告 `§9.2` 命令以本版本为准。

### 3.3 `compare_precision.py` — 跨设备精度比对

```text
用法:
python compare_precision.py results_ref.npz results_test.npz
```

- 逐 case / 汇总的 max-abs、mean-abs、rel-L2、cos；
- **物理量误差**：按 Franka 编码（`[dxyz, drotvec, gripper]`）直接给出每步末端位置差（m）、
旋转差（rad，四元数角距离）、夹爪差；
- 阈值判定：cos > 0.995、位置 < 1cm、旋转 < 0.05rad、夹爪 < 0.05，输出 PASS/FAIL；
- 自动比对 stage 张量（`vlm_last_embed` / 逐步 `v_t`）若两侧 npz 均含。

### 3.4 `dit_bisect.py` — 算子级确定性 bisect

- 对 DiT 内全部 Linear/LayerNorm/GEGLU/Attention/AdaLayerNorm/TimestepEncoder 等叶子模块
挂 forward hook，同输入重复 3 次去噪，报告首个输出不一致的模块；
- 排查"非确定"假象时使用（结论：250 个模块 bitwise 全等，问题在测试脚本而非算子）；
- 平时无需运行，作为诊断工具保留。

---

## 4. 合并后验证清单

```bash
# 1. 冒烟（含离线 backbone 构造）
python npu_smoke_test.py --ckpt /home/Spirit-v1.5 --backbone /home/qwen3vl-4b-small

# 2. 确定性（应输出 PASS，pairwise = 0）
python precision_test.py --ckpt ... --backbone ... --cases 0 --stages 0 \
--determinism --inputs inputs.npz

# 3. 性能（对照报告 §5: mean ≈ 302ms）
python perf_bench.py --ckpt ... --backbone ... --iters 30 --warmup 5 \
--save-inputs inputs.npz --out perf_npu.json

# 4. mock 集成（对照报告 §3；从 RoboChallengeInference 根目录执行）
cd /home/spirit-run/RoboChallengeInference
( cd mock_server && MOCK_ROBOT_TAG=franka \
    MOCK_RECORD_DATA_DIR=../20250919/press_three_buttons/data/episode_FFFFFF \
    nohup python mock_robot_server.py > /tmp/mock_franka.log 2>&1 & )
for i in $(seq 1 40); do
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 http://127.0.0.1:9098/clock-sync || true)
  [ "$code" = "200" ] && break; sleep 1
done
SPIRIT_ROOT=<spirit路径> python test_spirit.py \
--checkpoint /home/Spirit-v1.5 --backbone /home/qwen3vl-4b-small \
--single_task press_three_buttons
fuser -k 9098/tcp 2>/dev/null || true

# 5. 离线打分（对照报告 §6/§7）
SPIRIT_ROOT=<spirit路径> python offline_action_eval.py \
--checkpoint /home/Spirit-v1.5-move-box --backbone /home/qwen3vl-4b-small --device npu \
--tag franka --task move_objects_into_box \
--record_dir 20250919/move_objects_into_box/data/episode_000000
```

---

## 5. 注意事项

1. `x_t = noise.clone()` 对生产行为无影响（生产路径 noise=None 内部生成），但**必须与
`precision_test.py --determinism` 配合使用**——旧版脚本复用 noise 会复现假"非确定"；
2. `precision_test.py` 依赖 `perf_bench.py` 的 `build_batch/pick_device/sync`（import 关系，
两个文件需同目录）；
3. 容器内依赖：transformers 4.57.1 + huggingface_hub 0.36.2 + diffusers 0.35.2
（`--no-deps` 安装，详见 `Spirit-v1.5-Ascend-NPU-Porting.md` §3.3）；
4. 离线 backbone 构造依赖本地 Qwen 小文件目录（`--backbone` 传入），离线环境禁止触发
HF 下载（脚本已设 `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`）；
5. `dit_bisect.py` 硬编码了 `/home/Spirit-v1.5` 等路径，仅在测试机使用，不作为通用工具维护。
