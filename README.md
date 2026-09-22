# DuVLA

DuVLA 是一个以冻结 [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
为视觉语言骨干、使用 Flow Matching 生成机器人动作的研究项目。它将双相机图像、语言指令和机器人状态
接入统一策略；训练时缓存骨干特征，推理时在线编码并在每次执行两步动作后重新观察。
Qwen 骨干保持冻结，策略训练面向单张 8GiB 显存的消费级 NVIDIA GPU。

当前对外主模型为 **V3.31**。同一 checkpoint 在 LIBERO 四套、40 个任务、2000 个官方初始状态上
取得 **1906/2000（95.30%）** 的闭环开发测评成功率。测评使用单独准备的策略权重，
也可以按下文流程自行训练。

[下载 V3.31 模型权重](https://huggingface.co/doggodman/duvla-v3.31)。
作者与维护者：[goddogman](https://github.com/goddogman)。
Hugging Face 发布账号为 [doggodman](https://huggingface.co/doggodman)，由同一作者维护。

[安装](#安装ubuntu) · [测评](#测评-v331) · [从头训练](#从头训练-v331) ·
[复现细节](docs/reproduction.md) · [引用](CITATION.cff) · [贡献指南](CONTRIBUTING.md)

## 结果

<!-- DUVLA_RESULTS_BEGIN -->
| 模型 | LIBERO-10 | Spatial | Object | Goal | 总计 |
| --- | ---: | ---: | ---: | ---: | ---: |
| **V3.31** | 457/500 | 479/500 | 484/500 | 486/500 | **1906/2000（95.30%）** |
<!-- DUVLA_RESULTS_END -->

每套 10 个任务、每任务 50 个官方初始状态。统一策略使用自然语言，不用 benchmark task index
选择专家。协议为 20 Hz 仿真控制、双相机 128×128、5 条 Flow 候选、每轮执行前 2 步、
seed 23、OSMesa 渲染；四套环境步数上限依次为 520/280/280/300。

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

参数表分别统计冻结的 Qwen 骨干、策略 checkpoint，以及 V3.31 参与训练和推理的策略路径。

## 系统要求与资产

安装说明以 **Ubuntu Linux x86_64** 为准，推荐 Python 3.12、NVIDIA CUDA GPU 和 OSMesa。
开发设备为 RTX 5060 Laptop（8GiB 显存）；不同设备的吞吐和显存峰值需自行核验。

| 用途 | 必要资产 |
| --- | --- |
| 测评 V3.31 | [DuVLA 权重](https://huggingface.co/doggodman/duvla-v3.31)中的 `policy.pt` 与 `train_manifest.json`、[Qwen 骨干](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)、[LIBERO 模拟器/资产](https://github.com/Lifelong-Robot-Learning/LIBERO) |
| 从头训练 | 上述 Qwen 与 LIBERO 源码、[LIBERO 官方示范 HDF5](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets) 的 Spatial/Object/Goal/10 四套、约 210GiB 的本地特征缓存空间 |

模型和数据均**不会**在 `import duvla` 或安装时自动下载。
只做测评无需示范 HDF5 或训练特征缓存。

## 安装（Ubuntu）

先安装 Python 3.12（含 `venv`）、Git 和与 GPU 相配的 NVIDIA 驱动。获取源码：

```bash
git clone https://github.com/goddogman/duvla.git
cd duvla
```

以下命令在仓库根目录运行；
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
git -C "$DUVLA_LIBERO_ROOT" checkout --detach 8f1084e3132a39270c3a13ebe37270a43ece2a01
python -m pip install --no-deps -e "$DUVLA_LIBERO_ROOT"
export MUJOCO_GL=osmesa
# 第一次导入若提示配置 LIBERO 资产路径，选择源码内默认路径可回答 n。
python -c 'import libero.libero'

python -m pip check
python -c 'import duvla, torch; assert torch.cuda.is_available(); print(duvla.__version__, torch.__version__, torch.cuda.get_device_name(0))'
python scripts/check_libero_eval_env.py
```

已有 `LIBERO` 目录时先检查其位置。LIBERO 源码采用 `--no-deps` 安装，以保留上面的
依赖版本组合；主要依赖版本见 `requirements/verified-runtime.txt`。
DuVLA 的初始状态加载器兼容新版 PyTorch，环境检查会读取一份官方初始状态。

## 测评 V3.31

从 [Hugging Face 模型仓库](https://huggingface.co/doggodman/duvla-v3.31)下载 V3.31 策略文件，
并准备本地 Qwen 骨干：

```bash
hf download doggodman/duvla-v3.31 \
  policy.pt model_config.json train_manifest.json \
  --revision 5f5003cbf8c2f9febf2d88fd6b1e84107bd07689 \
  --local-dir weights/duvla-v3.31

hf download Qwen/Qwen3-VL-2B-Instruct \
  --revision 89644892e4d85e24eaac8bacfd4f463576704203 \
  --local-dir models/Qwen3-VL-2B-Instruct
```

首次使用先检查策略文件，可在 CPU 上完成，无需加载 Qwen 或启动仿真：

```bash
python scripts/check_duvla_model.py --model-dir weights/duvla-v3.31
```

V3.31 表示策略版本，Python 包版本独立维护；模型、代码和上游版本对应关系见
[固定版本与文件校验](docs/reproduction.md#固定版本与文件校验)。

评测脚本沿用历史文件名 `evaluate_duvla_v2_1.py`，但会按 checkpoint 加载 V3.31 策略。

```bash
export DUVLA_QWEN_PATH="$PWD/models/Qwen3-VL-2B-Instruct"
export DUVLA_MODEL_DIR="$PWD/weights/duvla-v3.31"
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

聚合器会检查结果完整性。默认不保存视频；换权重或协议时使用新的结果目录。
DuVLA checkpoint 由项目评测脚本加载。详细协议见[复现说明](docs/reproduction.md)。

## 从头训练 V3.31

训练数据采用官方 LIBERO 四套 HDF5：40 任务 × 每任务 50 条示范，共 2000 条。
全部示范参与训练，归一化只根据训练数据计算；不把评测初始状态、奖励或轨迹用于梯度训练。
数据契约在该来源上审计为 `obs[t] → actions[t+1:t+9]`，共有 336,575 个有效训练起点；
其他数据源不得直接沿用这个时序偏移。

训练过程是 **缓存冻结 Qwen 特征 → 基础策略从头 30E → V3.31 联合适配 30E**。
其中 V3.29 是内部父策略训练阶段。训练前准备充足磁盘空间
（特征/侧车缓存合计约 210GiB，另需 checkpoint 和系统余量），并在仓库根目录配置：

```bash
export DUVLA_QWEN_PATH=/path/to/Qwen3-VL-2B-Instruct
export DUVLA_DATA_ROOT=/path/to/libero_official_hdf5
export DUVLA_CACHE_DIR="$PWD/cache"
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
```

`DUVLA_DATA_ROOT` 下须包含 `libero_spatial/`、`libero_object/`、`libero_goal/`、`libero_10/`。
仅训练时需要下载以下四套示范；指定快照直接提供 HDF5，无需解压：

```bash
hf download yifengzhu-hf/LIBERO-datasets --repo-type dataset \
  --revision f13aa24a3da8c43c7225569f28c562979fa0e35a \
  --include 'libero_spatial/*.hdf5' 'libero_object/*.hdf5' \
            'libero_goal/*.hdf5' 'libero_10/*.hdf5' \
  --local-dir "$DUVLA_DATA_ROOT"
```

目录应为 `libero_official_hdf5/libero_10/<task>_demo.hdf5` 等，每套 10 个文件。
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

## 致谢与许可

依赖[Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)、
[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)及PyTorch等开源项目。
项目自有代码采用[MIT](LICENSE)。
第三方代码、基座权重、数据和模拟资产许可独立适用。
来源见[第三方说明](THIRD_PARTY_NOTICES.md)。
