---
license: mit
language:
- zh
- en
base_model: Qwen/Qwen3-VL-2B-Instruct
tags:
- robotics
- vision-language-action
- libero
- flow-matching
---

# DuVLA {{VERSION}}

这是唯一发布模型V3.31的本地待发布模型卡。采用MIT；作者GitHub账号与源码地址待补。
项目采用自定义PyTorch加载器，不支持直接使用AutoModel.from_pretrained。

## 模型与硬件

双相机128×128和自然语言输入，冻结Qwen3-VL-2B提供layer14空间网格与多层语义，
语言桥接及双相机融合后输入Flow动作专家。每次生成5条8步7D动作，逐坐标中值，执行前2步。
机器人policy为134,394,924参数，主策略激活路径101,895,440参数；完整推理仍需要约2.13B的Qwen。
RTX 5060 Laptop 8GiB实机完成缓存策略训练及在线测评，不是8GB全量训练Qwen。

完整结果为1906/2000（95.30%）。此包版本见标题及bundle_manifest.json，不包含LoRA实验。

## 训练与结果

使用2000条LIBERO官方训练示范、336,575个因果有效起点，训练数据源和revision在manifest中。
V3.31经过基础策略从头30E、再全量30E联合适配及训练侧增强；前置阶段在训练代码中标识为V3.29。
训练不使用评测初始状态、成功信号或奖励。

| 版本 | LIBERO-10 | Spatial | Object | Goal | 总计 |
| --- | ---: | ---: | ---: | ---: | ---: |
| V3.31 | 457/500 | 479/500 | 484/500 | 486/500 | 1906/2000 |

测评：单模型、自然语言、无task-index、40任务×50状态；20Hz仿真控制、OSMesa、
K5/action2/seed23、初始化稳定10步、时限520/280/280/300。20Hz不等于实测推理频率。
这是已用于研究迭代的完整开发测评，不是盲测或未见任务泛化。
evaluation.json保留2000条小型逐集结果，不包含视频或模拟器隐藏状态。

## 使用

policy.pt含全部机器人策略。train_manifest.json含必要归一化。
仅推理不需要训练缓存/HDF5，也不需要其他旧policy；Qwen和LIBERO模拟资产需单独获取。
Qwen固定revision为89644892e4d85e24eaac8bacfd4f463576704203。
设置DUVLA_QWEN_PATH后使用源码仓库的scripts/evaluate_duvla_v2_1.py；完整命令见源码docs/reproduction.md。
原始及发布SHA256见provenance.json。发布版只规范私人路径，全部权重张量已逐项核验相等。
当前仍使用可信PyTorch checkpoint并通过weights_only=True加载，尚未转换safetensors。

## 限制与许可

没有真实机器人安全验证、未见任务或完整LIBERO-Plus泛化结论。
不适合直接控制真实机器人；需要独立控制、安全与动作语义校验。
DuVLA自有代码和策略采用MIT；Qwen上游标注Apache-2.0，LIBERO代码为MIT，
不因此推断全部模拟资产/示范的再分发许可。此包不含Qwen权重、原始数据或模拟资产。
