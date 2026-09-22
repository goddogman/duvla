# 第三方来源

本文件是来源清单，不替代第三方LICENSE，也不是DuVLA的许可证。

| 来源 | 用途 | 许可与使用说明 |
| --- | --- | --- |
| Qwen/Qwen3-VL-2B-Instruct | V3.31 的冻结视觉语言骨干 | 官方model card标注Apache-2.0；单独下载，不包含在策略权重包 |
| Lifelong-Robot-Learning/LIBERO | 模拟器任务、测评 | 源码LICENSE为MIT，保留上游版权；从上游仓库获取 |
| yifengzhu-hf/LIBERO-datasets | 官方训练HDF5快照 | revision已记录；数据/模拟资产具体再分发许可需单独核验，不随本包上传 |
| PyTorch / Transformers / robosuite / MuJoCo | V3.31 运行依赖 | 由使用者按各上游许可独立安装，本包不打包其源码或环境 |
| SmolVLA / LeRobot | Flow Matching 动作专家设计与 LIBERO 接口参考 | 参考实现见上游链接；DuVLA 的相关实现位于 `src/duvla/models/flow_matching_vla.py` 和 `src/duvla/evaluation/libero_contract.py`；V3.31 运行入口不依赖 LeRobot 包 |

官方来源：

- https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct
- https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/LICENSE
- https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets
- https://github.com/huggingface/lerobot
- https://github.com/huggingface/lerobot/blob/main/LICENSE

项目自有代码采用MIT（见LICENSE）；第三方代码、Qwen、数据和模拟资产仍遵循其上游许可。
设计参考、外部运行依赖与复制/改写代码的许可义务分别适用；第三方文件及其改写版本中的
版权和许可声明应随相应文件保留。本项目的 MIT 声明不改变这些上游声明。
