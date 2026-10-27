# Spirit-v1.5 昇腾 NPU 部署与测试指导

| 项    | 值                                                                        |
| ---- | ------------------------------------------------------------------------ |
| 文档版本 | v1.0                                                                     |
| 日期   | 2026-10-27                                                               |
| 适用硬件 | Atlas 300I A2（Ascend 910B4 PCIE，32GB HBM）单卡；其他 910B 型号需自行验证              |
| 实测栈  | CANN 9.1.0 + torch_npu 2.10.0.post4 + torch 2.10.0 + transformers 4.57.1 |
| 关联文档 | `Spirit-v1.5-Ascend-NPU-Porting.md`（代码改动说明）、`Spirit-v1.5-Ascend-910B-Test-Report.md`（测试报告） |

> 本文档是从零到"benchmark + 精度 + RoboChallenge 集成测试全部拉起"的完整操作手册，
> 所有命令均在 910B4 实测通过。

---

## 1. 前置检查（宿主机）

```bash
# 1.1 NPU 驱动/固件（确认卡在位、健康）
npu-smi info
# 期望: Health=OK，记下 NPU 序号（本文以 NPU 5 / /dev/davinci5 为例，PCIE 卡序号可能非 0）

# 1.2 驱动版本（与容器内 CANN 的兼容性实测: 25.3.rc1.1 + CANN 9.1.0 可用）
cat /usr/local/Ascend/driver/version.info # package_version=25.3.rc1.1
cat /usr/local/Ascend/firmware/version.info # package_version=25.2.0

# 1.3 磁盘空间（权重 21G + 微调 21.6G + 镜像 18G + 数据集 12G）
df -h /home
```

> 驱动兼容性：无需为 CANN 9.1.0 升级驱动。固件兼容声明上限略低于驱动版本，
> 实测无影响；量产环境建议将固件对齐到驱动同代（25.3.x）。

---

## 2. 安装 Docker（宿主机无 Docker 时）

openEuler 22.03 aarch64 rpm（版本必须匹配宿主 OS）：

```bash
rpm -ivh docker-runc-1.0.0.rc3-300.oe2203.aarch64.rpm
rpm -ivh docker-engine-18.09.0-300.oe2203.aarch64.rpm
systemctl enable --now docker
docker version # Client/Server 18.09.0
```

> `docker-compose` / `docker-client-java` 缺 python3/maven 依赖，跑容器用不到，可跳过。

---

## 3. 物料清单（离线环境提前备齐，放 `/home`）

| 物料              | 目标路径                                                     | 获取方式                                                                                    |
| --------------- | -------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| vllm-ascend 镜像  | `/home/v0.26.0rc2.tar.gz`                                | `docker pull quay.io/ascend/vllm-ascend:v0.26.0rc2 && docker save` 或直接取 `docker save` 包 |
| base 权重         | `/home/Spirit-v1.5/`                                     | HF `Spirit-AI-robotics/Spirit-v1.5`（21GB，config.json + model.safetensors）               |
| 微调权重            | `/home/Spirit-v1.5-move-box/`                            | HF `Spirit-AI-robotics/Spirit-v1.5-for-RoboChallenge-move-objects-into-box`（21.6GB）     |
| Qwen3-VL 小文件    | `/home/qwen3vl-4b-small/`                                | 见 3.1（**不要下载权重**，几 MB 小文件即可）                                                            |
| Python wheels   | `/home/wheels/`                                          | 见 3.2（3 个 wheel）                                                                        |
| Spirit 代码       | `/home/spirit-run/spirit-v1.5/`                          | `ascend-npu-porting` 分支 + 合入指南中的变更                                                      |
| RC 客户端          | `/home/spirit-run/RoboChallengeInference/`               | `Stonesan233/RoboChallengeInference` @ `4090_test` 分支                                   |
| movebox episode | `.../20250919/move_objects_into_box/data/episode_000000` | 数据集 `RoboChallenge/task_table30_move_objects_into_box` 抽取                               |

### 3.1 Qwen3-VL-4B-Instruct 小文件（离线构造骨干用，约 12MB）

```bash
export HF_HUB_DOWNLOAD_TIMEOUT=10000
export HF_ENDPOINT=https://hf-mirror.com
hf download Qwen/Qwen3-VL-4B-Instruct "*.json" "merges.txt" "*.jinja" "*.model" \
--local-dir /home/qwen3vl-4b-small
```

> **坑**：新版 `hf` CLI 的 patterns 是**位置参数**；`--include` 与位置参数混用会被忽略
> （只下到 1 个文件）。**不需要下载 Qwen3-VL 的权重**——Spirit checkpoint 已含全部
> `qwen.*`，`load_state_dict` 会整体覆盖，模型代码用本地目录按 config 构造结构。

### 3.2 Python wheels（覆盖镜像内版本，约 17MB）

```bash
pip download transformers==4.57.1 --no-deps -d wheels/
pip download huggingface_hub==0.36.2 --no-deps -d wheels/
pip download diffusers==0.35.2 --no-deps -d wheels/
```

> 版本原因：镜像内置 transformers 5.14.1 / hub 1.31.0，Spirit 代码基于 4.57（要求 hub<1.0）；
> 镜像内**没有 diffusers**（DiT 头依赖）。三个 wheel 必须 `--no-deps` 一次装齐。

---

## 4. 加载镜像

```bash
docker load -i /home/v0.26.0rc2.tar.gz
docker images # quay.io/ascend/vllm-ascend:v0.26.0rc2 (~17.8GB)
```

---

## 5. 创建容器（标准模板）

```bash
DEVS=""
for d in /dev/davinci[0-9]* /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc; do
[ -e "$d" ] && DEVS="$DEVS --device $d"
done
echo "devices:$DEVS" # 本例: --device /dev/davinci5 + manager + devmm_svm + hisi_hdc

docker run --rm --name spirit_work $DEVS \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
-v /usr/local/dcmi:/usr/local/dcmi:ro \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
-v /home/spirit-run:/home/spirit-run \
-v /home/Spirit-v1.5:/home/Spirit-v1.5:ro \
-v /home/Spirit-v1.5-move-box:/home/Spirit-v1.5-move-box:ro \
-v /home/qwen3vl-4b-small:/home/qwen3vl-4b-small:ro \
-v /home/wheels:/home/wheels:ro \
-w /home/spirit-run/spirit-v1.5 \
--entrypoint bash quay.io/ascend/vllm-ascend:v0.26.0rc2 -c "<命令>"
```

要点：

- 设备节点按实际存在的 glob 挂载（PCIE 卡可能是 `davinci5` 而非 `davinci0`）；
- 权重/小文件/wheels 全部 `:ro` 只读挂载；工作输出写到 `/home/spirit-run`（宿主目录）；
- **容器是 `--rm` 的，`pip install` 不落盘**——每次启动重新装 wheel（约 40s）；
环境固化：去掉 `--rm` 跑一次安装后 `docker commit <id> spirit-npu:v1`。

---

## 6. 容器内环境配置与验证

```bash
# 6.1 安装依赖（一次命令，顺序无关）
pip install --no-deps /home/wheels/huggingface_hub-0.36.2-py3-none-any.whl \
/home/wheels/transformers-4.57.1-py3-none-any.whl \
/home/wheels/diffusers-0.35.2-py3-none-any.whl

# 6.2 验证栈
python -c "import transformers, diffusers, huggingface_hub as h; \
print(h.__version__, transformers.__version__, diffusers.__version__)"
# 期望: 0.36.2 4.57.1 0.35.2

# 6.3 NPU 可用性
python -c "import torch, torch_npu; print(torch.npu.device_count(), torch.npu.get_device_name(0))"
# 期望: 1 Ascend910B4（首次导入约 10-30s）
```

---

## 7. 冒烟测试

```bash
python npu_smoke_test.py --ckpt /home/Spirit-v1.5 --backbone /home/qwen3vl-4b-small
```

期望输出：

```
action shape: (1, 60, 14)
finite: True
```

（首次运行含 CPU 随机初始化骨干 + 21GB 权重加载，约 2-3 分钟；无害警告
`NPUCachingAllocator ... 32 padding size` 可忽略。）

---

## 8. 性能基准

```bash
python perf_bench.py --ckpt /home/Spirit-v1.5 --backbone /home/qwen3vl-4b-small \
--iters 30 --warmup 5 \
--save-inputs /home/spirit-run/artifacts/inputs.npz \
--out /home/spirit-run/artifacts/perf_npu.json
```

- `--save-inputs` 生成固定种子共享输入（跨设备比对时 4090 侧用 `--inputs` 消费同一文件）；
- 实测参考（910B4）：端到端 mean 301.85ms / p99 335ms，VLM 前向 ~144ms，
去噪 10 步 ~146ms，峰值显存 13.45GB；
- 计时已内置 `torch.npu.synchronize()`，不要用裸 `time.time()` 直接测。

---

## 9. 确定性与精度测试

```bash
# 9.1 确定性（同输入 3 次，期望 pairwise max_abs_diff = 0，PASS）
python precision_test.py --ckpt /home/Spirit-v1.5 --backbone /home/qwen3vl-4b-small \
--cases 0 --stages 0 --determinism --inputs /home/spirit-run/artifacts/inputs.npz \
--out /home/spirit-run/artifacts/det_bf16.npz

# 9.2 fp32 模式确定性
python precision_test.py --ckpt ... --cases 0 --stages 0 --determinism --no-autocast \
--inputs ... --out /home/spirit-run/artifacts/det_fp32.npz

# 9.3 精度采样（15 case，含 vlm_last_embed / 逐步 v_t dump）
python precision_test.py --ckpt ... --backbone ... --cases 15 --stages 3 --determinism \
--inputs /home/spirit-run/artifacts/inputs.npz --out /home/spirit-run/artifacts/results_npu.npz
```

诊断工具（确定性异常时使用）：

```bash
# 算子级 bisect：250 个模块同输入 3 次比对，定位首个发散模块
python dit_bisect.py
```

跨设备比对（4090 侧产出 `results_4090.npz` 后）：

```bash
python compare_precision.py results_4090.npz results_npu.npz
# 输出逐 case max/rel-L2/cos + 物理量误差（位置<1cm / 旋转<0.05rad / 夹爪<0.05）+ PASS/FAIL
```

---

## 10. RoboChallenge 集成测试

### 10.1 mock 接口（4 机型 + movebox）

```bash
cd /home/spirit-run/RoboChallengeInference
for pair in franka:press_three_buttons aloha:pour_fries_into_plate \
arx5:arrange_paper_cups ur5:shred_scrap_paper; do
tag=${pair%%:*}; task=${pair##*:}
ep=$(ls -d 20250919/$task/data/episode_* | head -1)
(cd mock_server && MOCK_ROBOT_TAG=$tag MOCK_RECORD_DATA_DIR=../$ep \
python mock_robot_server.py > /tmp/mock_$tag.log 2>&1 &)
sleep 3
SPIRIT_ROOT=/home/spirit-run/spirit-v1.5 python test_spirit.py \
--checkpoint /home/Spirit-v1.5 --backbone /home/qwen3vl-4b-small \
--single_task $task --output /home/spirit-run/artifacts/spirit_mock_$tag.json
fuser -k 9098/tcp 2>/dev/null || true
done
```

- fastapi/uvicorn/httpx/loguru 镜像自带，无需额外安装；
- movebox 用微调权重：`--checkpoint /home/Spirit-v1.5-move-box --single_task move_objects_into_box`
（动作形状自动变为 40×8，`TASKS_USE_LESS_CHUNK_SIZE=40` 生效）。

### 10.2 离线动作打分（精度对照）

```bash
# base 全任务
SPIRIT_ROOT=/home/spirit-run/spirit-v1.5 python offline_action_eval.py \
--checkpoint /home/Spirit-v1.5 --backbone /home/qwen3vl-4b-small --device npu \
--num_samples 6 --output /home/spirit-run/artifacts/offline_npu_base.json

# movebox 单任务（微调 vs base，均含保持不动基线）
SPIRIT_ROOT=/home/spirit-run/spirit-v1.5 python offline_action_eval.py \
--checkpoint /home/Spirit-v1.5-move-box --backbone /home/qwen3vl-4b-small --device npu \
--tag franka --task move_objects_into_box \
--record_dir 20250919/move_objects_into_box/data/episode_000000 \
--output /home/spirit-run/artifacts/offline_npu_movebox_finetuned.json
```

实测参考（910B4，详见测试报告 §6/§7）：base 4 任务位置 MAE 8.48–12.86cm，
movebox 微调 12.02cm / 首步 1.37cm（4090: 12.17cm / 1.00cm）。

### 10.3 线上真机评测（需出网条件）

```bash
cd /home/spirit-run/spirit-v1.5
export TASK_NAME=move_objects_into_box
export ROBOCHALLENGE_JOB_ID=<平台上创建的 job collection id>
export USER_TOKEN=<平台用户 token>
export CKPT_PATH=/home/Spirit-v1.5-move-box
python -m robochallenge.run_robochallenge \
--single_task $TASK_NAME --robochallenge_job_id $ROBOCHALLENGE_JOB_ID \
--ckpt_path $CKPT_PATH --backbone /home/qwen3vl-4b-small \
--user_token $USER_TOKEN --used_chunk_size 60
```

**前提**：推理机必须能访问 `http://api.robochallenge.cn`（公网）。纯内网环境的替代方案：
经有出网能力的跳板建 `ssh -R` 反向隧道，并把 `robochallenge/robot/interface_client.py`
的 `base_url` 改为环境变量可配置（指向 `http://127.0.0.1:<端口>`）。此环节未在本环境验证。

---

## 11. 常见问题与排查

| 现象                                             | 原因 / 处理                                                                                 |
| ---------------------------------------------- | --------------------------------------------------------------------------------------- |
| `Unable to find image ... locally` 后卡住         | 容器内无外网，docker 尝试 pull 挂起——确认镜像已 `docker load`                                           |
| `ModuleNotFoundError: wandb`                   | 已在代码中改为可选导入；若仍出现，确认部署的是最新代码                                                             |
| `LocalEntryNotFoundError: huggingface.co`      | backbone 未用本地目录覆盖——`from_pretrained`/executor 必须传 `--backbone /home/qwen3vl-4b-small`   |
| `TypeError: cannot assign ... as child module` | 测试脚本不能直接覆盖 nn.Module 子模块属性，需包装其 `forward`（perf_bench 已修复）                               |
| 同输入两次推理结果不同                                    | 检查是否复用了 noise 张量（模型已改 `x_t = noise.clone()`，测试侧也要 `noise.clone()`）；用 `dit_bisect.py` 排查 |
| pip 装完重启容器后丢失                                  | `--rm` 容器无持久化，属预期；固化用 `docker commit`                                                   |
| 设备节点 404                                       | PCIE 卡可能是 `/dev/davinci5`，用 glob 挂载全部存在的 `davinci[0-9]*`                                |
| wheel 安装后 import 冲突                            | 三个 wheel 必须 `--no-deps` 一次装齐（transformers 4.57.1 强依赖 hub<1.0，镜像内置 1.x）                  |
