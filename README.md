# DuVLA

DuVLA 是一个以冻结 [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
为视觉语言骨干、使用 Flow Matching 生成机器人动作的研究项目。它将双相机图像、语言指令和机器人状态
接入统一策略；训练时缓存骨干特征，推理时在线编码并在每次执行两步动作后重新观察。
设计目标是在单张 8GiB 显存的消费级 NVIDIA GPU 上训练策略，**不是全参数训练 Qwen**。

当前对外主模型为 **V3.31**。同一 checkpoint 在 LIBERO 四套、40 个任务、2000 个官方初始状态上
取得 **1906/2000（95.30%）** 的闭环开发测评成功率。源码仓库不附带策略权重；
使用已有本地权重，或按下文流程自行训练。

[安装](#安装ubuntu) · [测评](#测评-v331) · [从头训练](#从头训练-v331) ·
[复现细节](docs/reproduction.md) · [贡献指南](CONTRIBUTING.md)

## 结果

<!-- DUVLA_RESULTS_BEGIN -->
| 模型 | LIBERO-10 | Spatial | Object | Goal | 总计 |
| --- | ---: | ---: | ---: | ---: | ---: |
| **V3.31** | 457/500 | 479/500 | 484/500 | 486/500 | **1906/2000（95.30%）** |
<!-- DUVLA_RESULTS_END -->

每套 10 个任务、每任务 50 个官方初始状态。统一策略使用自然语言，不用 benchmark task index
选择专家。协议为 20 Hz 仿真控制、双相机 128×128、5 条 Flow 候选、每轮执行前 2 步、
seed 23、OSMesa 渲染；四套环境步数上限依次为 520/280/280/300。20 Hz **不是推理帧率**。
这些初始状态已用于开发分析，结果不是独立盲测、未见任务或真实机器人泛化。

## 模型

```text
双相机 RGB + 语言 ──> 冻结 Qwen3-VL-2B ──> 空间网格/多层语义/有序语言 token
                                                        │
8D 机器人状态 ────────────────────────────────> 跨模态融合与双相机交互
                                                        │
                                        历史上下文 + Flow Matching 动作专家
                                                        │
                                    5 条候选 × 8 步 × 7D 动作；中值聚合
                                                        │
                                         执行 2 步，再观察并重新规划
```

策略使用 layer 14 的 8×8 空间网格、layer 12/14/18/final 的语义摘要与有序语言 token；
动作专家宽度 768，交替堆叠 cross-attention 与因果 self-attention。
7D 动作包含 3D 平移、3D 旋转和 1D 夹爪；策略为统一模型，不靠任务编号路由。

| 参数口径 | 数量 |
| --- | ---: |
| 冻结 Qwen 骨干 | 约 2.13B |
| V3.31 policy checkpoint 全部参数 | 134,394,924 |
| 当前训练且参与推理的策略参数 | 101,895,440 |

其余 checkpoint 参数用于保持历史模型拓扑并支持严格加载，不代表当前启用了独立动作残差专家。
V3.31 **没有** Outcome Verifier、Recovery 或 Qwen LoRA；完整推理系统也不能称为“只有 101.9M 参数”。

## 系统要求与资产

安装说明以 **Ubuntu Linux x86_64** 为准，推荐 Python 3.12、NVIDIA CUDA GPU 和 OSMesa。
开发设备为 RTX 5060 Laptop（8GiB 显存）；不同设备的吞吐和显存峰值需自行核验。

| 用途 | 必要资产 |
| --- | --- |
| 测评 V3.31 | 策略 `policy.pt` 与 `train_manifest.json`、[Qwen 骨干](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)、[LIBERO 模拟器/资产](https://github.com/Lifelong-Robot-Learning/LIBERO) |
| 从头训练 | 上述 Qwen 与 LIBERO 源码、[LIBERO 官方示范 HDF5](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets) 的 Spatial/Object/Goal/10 四套、约 210GiB 的本地特征缓存空间 |

模型和数据均**不会**在 `import duvla` 或安装时自动下载。
只做测评无需示范 HDF5 或训练特征缓存。

## 安装（Ubuntu）

先安装 Python 3.12（含 `venv`）、Git 和与 GPU 相配的 NVIDIA 驱动。以下命令在仓库根目录运行；
PyTorch 示例对应本项目已运行的 CUDA 12.8 组合。不同驱动/平台请先查看
[PyTorch 官方安装说明](https://docs.pytorch.org/get-started/locally/)。

LIBERO 需要 OSMesa；Ubuntu 上可检查 `ldconfig -p | grep libOSMesa`，缺失时由系统管理员安装：

```bash
sudo apt-get update
sudo apt-get install -y libosmesa6 libgl1
```

创建隔离环境并安装项目与模拟器 Python 依赖：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install 'pip==26.1.2' 'setuptools==80.10.2' 'wheel==0.47.0'
# 先装 CUDA 版 PyTorch；后续依赖应保留这个版本。
python -m pip install 'torch==2.11.0+cu128' 'torchvision==0.26.0+cu128' \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements/verified-runtime.txt
python -m pip install -e '.[train,qwen,eval,dev]'

# LIBERO 源码不由 DuVLA 的 PyPI 依赖自动安装。
export DUVLA_LIBERO_ROOT="$HOME/projects/LIBERO"
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$DUVLA_LIBERO_ROOT"
python -m pip install --no-deps -e "$DUVLA_LIBERO_ROOT"
export MUJOCO_GL=osmesa
# 第一次导入若提示配置 LIBERO 资产路径，选择源码内默认路径可回答 n。
python -c 'import libero.libero'

python -m pip check
python -c 'import duvla, torch; assert torch.cuda.is_available(); print(duvla.__version__, torch.__version__, torch.cuda.get_device_name(0))'
python scripts/check_libero_eval_env.py
```

已有 `LIBERO` 目录时先检查，不要直接克隆覆盖。不要在此环境安装 LIBERO 上游整份旧
`requirements.txt`；其旧版 Transformers/NumPy 会覆盖本项目的固定组合。
`requirements/verified-runtime.txt`记录已运行的主要直接依赖，不是跨机器完整锁文件；
上游 LIBERO 源码与本项目开发所用的本地安装存在待审计差异。
`pip check`和导入通过只证明依赖接线，不等于闭环仿真通过。

## 测评 V3.31

准备本地 `policy.pt`、`train_manifest.json` 和 Qwen 骨干；路径由使用者指定。
评测脚本沿用历史文件名 `evaluate_duvla_v2_1.py`，但会按 checkpoint 加载 V3.31 策略。

```bash
export DUVLA_QWEN_PATH=/path/to/Qwen3-VL-2B-Instruct
export DUVLA_MODEL_DIR=/path/to/duvla-v3.31
export MUJOCO_GL=osmesa
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=2
python scripts/check_libero_eval_env.py

set -e
for suite in libero_10 libero_spatial libero_object libero_goal; do
  python scripts/evaluate_duvla_v2_1.py \
    --checkpoint "$DUVLA_MODEL_DIR/policy.pt" \
    --train-manifest "$DUVLA_MODEL_DIR/train_manifest.json" \
    --suite "$suite" --task-indices 0-9 \
    --all-init-states --init-state-start 0 --init-state-count 50 \
    --fps 20 --camera-size 128 --flow-seed 23 --flow-samples 5 --action-steps 2 \
    --trace-dir "outputs/eval2000/$suite" --resume --device cuda
done
python scripts/aggregate_duvla_v2_1_eval.py \
  --root outputs/eval2000 --expected-episodes 2000 --output outputs/eval2000/aggregate.json
```

聚合器会拒绝不完整结果。默认不保存视频；换权重或协议必须换结果目录，不能混用 `--resume`。
项目 checkpoint 通过 DuVLA 加载器使用，不能直接交给 `AutoModel.from_pretrained`。
更详细的协议与结果文件见[复现指南](docs/reproduction.md)。

## 从头训练 V3.31

训练数据采用官方 LIBERO 四套 HDF5：40 任务 × 每任务 50 条示范，共 2000 条。
全部示范参与训练，归一化只根据训练数据计算；不把评测初始状态、奖励或轨迹用于梯度训练。
数据契约在该来源上审计为 `obs[t] → actions[t+1:t+9]`，共有 336,575 个有效训练起点；
其他数据源不得直接沿用这个时序偏移。

训练过程是 **缓存冻结 Qwen 特征 → 基础策略从头 30E → V3.31 联合适配 30E**。
脚本名中的 V3.29 是内部父阶段，不是另一个发布模型。训练前准备充足磁盘空间
（特征/侧车缓存合计约 210GiB，另需 checkpoint 和系统余量），并在仓库根目录配置：

```bash
export DUVLA_QWEN_PATH=/path/to/Qwen3-VL-2B-Instruct
export DUVLA_DATA_ROOT=/path/to/libero_official_hdf5
export DUVLA_CACHE_DIR="$PWD/cache"
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
```

`DUVLA_DATA_ROOT` 下须包含 `libero_spatial/`、`libero_object/`、`libero_goal/`、`libero_10/`。
正式训练命令：

```bash
# 1. 生成基础特征、语言侧车和训练侧增强侧车。
python scripts/cache_official_libero_hdf5_features.py \
  --dataset-root "$DUVLA_DATA_ROOT" --output "$DUVLA_CACHE_DIR/base" --batch-size 4
python scripts/cache_duvla_v3_29.py \
  --dataset-root "$DUVLA_DATA_ROOT" --base-cache "$DUVLA_CACHE_DIR/base" \
  --output "$DUVLA_CACHE_DIR/language"
python scripts/cache_duvla_v3_31.py \
  --dataset-root "$DUVLA_DATA_ROOT" --base-cache "$DUVLA_CACHE_DIR/base" \
  --language-sidecar "$DUVLA_CACHE_DIR/language" --output "$DUVLA_CACHE_DIR/augment"

# 2. 从头训练父策略 30E。
python scripts/train_duvla_v3_29.py \
  --base-cache "$DUVLA_CACHE_DIR/base" --sidecar "$DUVLA_CACHE_DIR/language" \
  --output outputs/base30e --batch-size 192 --epochs 30 --seed 17

# 3. 用本次父策略的实际 SHA256 训练 V3.31 30E。
export DUVLA_PARENT="$PWD/outputs/base30e/duvla-v3.29-30e.pt"
export DUVLA_PARENT_SHA256="$(sha256sum "$DUVLA_PARENT" | cut -d ' ' -f 1)"
python scripts/train_duvla_v3_31.py \
  --base-cache "$DUVLA_CACHE_DIR/base" --language-sidecar "$DUVLA_CACHE_DIR/language" \
  --augment-sidecar "$DUVLA_CACHE_DIR/augment" \
  --parent "$DUVLA_PARENT" --parent-sha256 "$DUVLA_PARENT_SHA256" \
  --output outputs/main30e --batch-size 144 --epochs 30 --seed 17
```

两个阶段均保存 checkpoint 和 `loss_curve.jsonl/csv/svg`。只检查接线时可在训练命令追加
`--smoke-updates 1 --batch-size 2`，并换用独立输出目录；这不代表训练完成或模型有效。
第二阶段可先用 `--check-inputs-only` 校验来源。中断续训需保持输入、代码和参数不变并追加
`--resume`。自己的训练权重测评时，将上节的 `policy.pt` 路径改为
`outputs/main30e/duvla-v3.31-30e.pt`，manifest 改为 `$DUVLA_CACHE_DIR/base/manifest.json`。

## 仓库结构

训练与测评使用同一个源码仓库。大型示范数据、Qwen 权重和特征缓存通过路径参数接入，
不进入 Git 提交。

```text
duvla/
├── src/duvla/          # 数据、策略和测评共享代码
├── scripts/            # 缓存、训练与LIBERO评测入口
├── tests/              # 数据/动作/加载契约测试
├── requirements/       # 已运行的主要依赖版本
├── docs/reproduction.md
├── pyproject.toml
└── LICENSE
```

GitHub 仓库保存源码、测试与使用说明；训练数据、特征缓存、checkpoint、逐集测评记录和
研究日志均留在本地。策略权重及归一化 manifest 单独管理；Qwen 骨干、LIBERO 数据与模拟
资产由使用者从各自来源获取。

## 复现边界

- README 只介绍当前主模型 V3.31；其他研究版本不作为本仓库的使用入口。
- 训练采用 2000 条 LIBERO 官方示范的全量 refit，**没有**示范 validation；上表的
  2000 集属于开发测评，不是独立验证集。
- 已对公开入口做安装/缓存/单更新接线检查，但尚未在原生 Ubuntu 的全新机器上完成
  全量 30E+30E 训练或 2000 集重跑。安装成功、`--help`和单更新冒烟不等于结果复现。
- 固定版本依赖与复现步骤见[复现说明](docs/reproduction.md)；完整实验记录保存在本地研究目录，
  不纳入 Git 提交。

## 致谢与许可

依赖[Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)、
[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)及PyTorch等开源项目。
项目自有代码采用[MIT](LICENSE)。
第三方代码、基座权重、数据和模拟资产许可独立适用。
来源与待核验项见[第三方说明](THIRD_PARTY_NOTICES.md)。
