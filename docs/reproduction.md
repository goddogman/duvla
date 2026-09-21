# 复现说明

## 已验证范围

本项目在WSL2 Ubuntu、RTX 5060 Laptop 8GiB、Python 3.12.13、PyTorch 2.11.0+cu128上
运行当前唯一发布模型V3.31。关键依赖见`requirements/verified-runtime.txt`。
这是一份已运行版本记录，尚不是干净机器验证通过的完整安装锁。
不在现有训练/评测环境中直接升级依赖。

面向Ubuntu的虚拟环境安装顺序见[README安装](../README.md#安装ubuntu)：先装对应CUDA版PyTorch，
再安装`requirements/verified-runtime.txt`和`.[train,qwen,eval,dev]`，最后安装LIBERO源码并检查依赖。
`pip install --no-deps -e .`仅适用于已手工备齐全部依赖的环境，**不是从零安装命令**。
安装后使用`import duvla`。仓库可克隆到任意路径（含空格）；命令在仓库根目录执行，
所有模型、数据和输出位置由使用者通过参数/环境变量指定，不能复制作者的绝对路径。
旧研究脚本的`qwen_vla`导入已迁移；原始state-dict权重不因包改名而重训或改写。
项目包不附带CUDA PyTorch轮子、LIBERO源码、Qwen权重或训练数据；这些分别准备。
`qwen` optional extra已对齐当前4.57.6/0.36.2/0.22.2组合。
当前本机LIBERO是site-packages源码安装，其精确上游commit与本地补丁仍待补充审计，
因此不能声称普通`pip install libero`能完整复现本机环境。

## LIBERO环境安装

README给出[官方LIBERO源码](https://github.com/Lifelong-Robot-Learning/LIBERO)的安装命令。
`eval` extra会装MuJoCo、robosuite 1.4、bddl、Gym等模拟器Python依赖；
`requirements/verified-runtime.txt`固定本机已运行的主要版本。上游LIBERO旧`requirements.txt`
含`transformers==4.21.1`等冲突版本，不要直接安装到本环境。
在首次调用评测器前，再检查以下环境和资产：

```bash
export MUJOCO_GL=osmesa
ldconfig -p | grep libOSMesa
python -c 'from libero.libero import get_libero_path; print({key: get_libero_path(key) for key in ("bddl_files", "init_states", "assets")})'
python scripts/check_libero_eval_env.py
```

首次导入按LIBERO提示配置其路径。配置键实际是`init_states`，不是`init_files`；
其目录通常对应源码中的`libero/libero/init_files`。OSMesa不是pip包，缺少`libOSMesa.so`
时由系统管理员安装Ubuntu包`libosmesa6`；本项目不自动修改驱动或系统库。
这里是完整的**安装命令与静态环境检查**，不是全新机器成功复现的证明：
本机LIBERO精确源码revision/补丁及干净机器闭环仍需审计，
不能将`--help`或包导入成功等同于渲染/2000集复现通过。

## 下载分工

1. DuVLA HF model repo（待创建）：`policy.pt`、`train_manifest.json`及模型说明。
2. Qwen官方仓库：`Qwen/Qwen3-VL-2B-Instruct`，固定revision
   `89644892e4d85e24eaac8bacfd4f463576704203`。本地HF下载metadata记录此revision；
   发布前还应完成全部骨干文件hash清单校验，不仅依赖config.json的revision记录。
3. LIBERO代码与模拟资产、官方初始状态；仅评测无需下载训练示范。
4. 仅重新训练才需要官方HDF5、基础空间特征缓存、有序语言侧车与增强侧车。

Qwen按上游说明下载到本地后，设置`DUVLA_QWEN_PATH`。
模型默认`local_files_only=True`，不因导入项目自动联网或下载。

## 完整2000集测评

在源码根目录和已经验证的环境中运行，修改前两项本地路径：

```bash
export DUVLA_QWEN_PATH=/your/local/Qwen3-VL-2B-Instruct
export DUVLA_MODEL_DIR=/your/local/duvla-v3.31
export DUVLA_EVAL_DIR="$PWD/outputs/reproduction_full2000"
export PYTHONPATH="$PWD/src:$PWD/scripts"
export MUJOCO_GL=osmesa
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=2

python scripts/check_libero_eval_env.py
for suite in libero_10 libero_spatial libero_object libero_goal; do
  python scripts/evaluate_duvla_v2_1.py \
    --checkpoint "$DUVLA_MODEL_DIR/policy.pt" \
    --train-manifest "$DUVLA_MODEL_DIR/train_manifest.json" \
    --suite "$suite" --task-indices 0-9 \
    --all-init-states --init-state-start 0 --init-state-count 50 \
    --fps 20 --camera-size 128 --flow-seed 23 --flow-samples 5 --action-steps 2 \
    --trace-dir "$DUVLA_EVAL_DIR/$suite" --resume --device cuda || break
done
python scripts/aggregate_duvla_v2_1_eval.py \
  --root "$DUVLA_EVAL_DIR" --expected-episodes 2000 \
  --output "$DUVLA_EVAL_DIR/aggregate.json"
```

任一套失败后循环停止，聚合器会拒绝缺失/不完整结果。续跑使用同样命令和`--resume`。
不要混合不同权重、manifest或协议的结果目录。新导出checkpoint的文件SHA256与原始文件不同，
原始测评证据继续绑定原始hash；`provenance.json`给出映射及张量一致性验证。

`--train-manifest`在测评中用于归一化和缓存签名核验，不读取feature shard。
发布manifest中的`${LOCAL_HOME}`是脱敏来源标记，不是运行时数据目录，无需手动替换。
20Hz是仿真控制频率；GPU在线前向和CPU模拟可能使实际执行慢于实时。

## 训练复现路径

主模型不是一次30E从头训练：

`官方2000示范 → 基础Qwen特征与因果语言侧车 → V3.29从头30E → V3.31追加30E`

固定数据源：`yifengzhu-hf/LIBERO-datasets`，revision
`f13aa24a3da8c43c7225569f28c562979fa0e35a`；具体四套文件/帧数由训练侧source contract审计。
全部示范用于训练，336,575个有效起点，数据契约为本来源的下一动作对齐。

| 阶段 | 实际入口 | 预算/依赖 |
| --- | --- | --- |
| 基础特征 | `cache_official_libero_hdf5_features.py` | Qwen在线前向，batch4，约170.9GiB |
| 因果动作/语言侧车 | `cache_duvla_v3_29.py` | 约31.7GiB，与基础manifest绑定 |
| 父策略训练 | `train_duvla_v3_29.py` | batch192，30E，52,590更新 |
| 配对增强 | `cache_duvla_v3_31.py` | 每任务250行，共10,000行，约7.7GiB |
| 主模型联合适配 | `train_duvla_v3_31.py` | batch144，30E，70,140更新 |

这些脚本的`--help`可检查输入。历史campaign不纳入首发入口：其绝对路径和自动训练链
属于原工作站。V3.29只是V3.31的内部前置训练阶段，不单独发布旧版本模型。
从头复现时按README构建一组新的、彼此绑定的缓存和父策略；不能把旧权重随意接到新缓存。
第二阶段`--parent-sha256`明确指定自己的完整30E父权重哈希；未传时仍锁定历史父策略。
同时检查父阶段预算、动作语义、数据签名和语言侧车hash，不使用“跳过验证”开关。
`--check-inputs-only`只作CPU来源校验，不等于完整分片遍历或GPU训练。
原生Linux检查本地空间；WSL必须设置`DUVLA_WSL_HOST_DRIVE`到实际VHDX所在盘，
如PowerShell不在默认位置另设`DUVLA_POWERSHELL`，无法核验时仍会阻止正式写入。
这一宿主磁盘检查属于本地WSL运行注意事项，普通Ubuntu安装不设置这些变量。
原始缓存/权重/结果的hash不修改；公开推理包不是含optimizer的resume快照。

训练缓存建议本地按需构建；首发不要求上传约210GiB缓存。
保存每阶段loss曲线和实际样本曝光，而非只比较optimizer step数。

## 可运行的轻量验证

```bash
PYTHONPATH=src:scripts python -m pytest -q \
  tests/test_import.py tests/test_flow_contract.py tests/test_libero_contract.py \
  tests/test_qwen_backbone.py tests/test_public_release.py tests/test_training_portability.py \
  tests/test_duvla_v3_31.py tests/test_v3_29_implementation.py
```

单元测试和CPU checkpoint加载不等于重新完成2000集复现。
公开结果来自原实验文件，硬件/环境改变后的实测应写入新的结果目录。
