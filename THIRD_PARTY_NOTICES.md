# 第三方来源与待核验事项

本文件是来源清单，不替代第三方LICENSE，也不是DuVLA的许可证。

| 来源 | 用途 | 当前核验状态 |
| --- | --- | --- |
| Qwen/Qwen3-VL-2B-Instruct | 冻结视觉语言骨干、可选LoRA基座 | 官方model card标注Apache-2.0；单独下载，不包含在发布包 |
| Lifelong-Robot-Learning/LIBERO | 模拟器任务、测评 | 源码LICENSE为MIT，保留上游版权；本机安装来源/补丁仍待锁定 |
| yifengzhu-hf/LIBERO-datasets | 官方训练HDF5快照 | revision已记录；数据/模拟资产具体再分发许可需单独核验，不随本包上传 |
| LIBERO-Plus | 独立鲁棒性研究 | 不包含在首发包，不把资产页许可扩展为所有代码/数据许可 |
| PyTorch / Transformers / PEFT / robosuite / MuJoCo | 运行依赖 | 由使用者按上游许可独立安装，本包不打包其源码或环境 |
| SmolVLA / LeRobot等参考工作 | 研究设计与历史实验参考 | 不宣称整仓源码已完成逐文件来源审计；发布前核实复制/改写部分并补版权声明 |

官方来源：

- https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct
- https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/LICENSE
- https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets
- https://github.com/sylvestf/LIBERO-plus

项目自有代码采用MIT（见LICENSE）；第三方代码、Qwen、数据和模拟资产仍遵循其上游许可。
禁止用“参考实现”代替应保留的第三方许可证；不添加虚构的作者、论文引用或来源声明。
