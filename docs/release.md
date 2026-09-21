# GitHub / Hugging Face 发布方案

## 发布定位

GitHub仓库名、项目目录与Python包名统一为`duvla`，项目展示名为DuVLA。
用户可将仓库克隆到任意目录；无需复刻作者的用户名和本机路径。
HF只发布`<账号>/duvla-v3.31`，完整结果1906/2000。其余版本/LoRA均为本地实验，不上传。
本次只生成本地候选，不创建、登录或上传远程仓库。

**维护方式更新**：当前仓库同时保留本地研究资料和公开代码，用`.gitignore`区分提交范围，
不搬移历史脚本、DPV4或学习日志。以后直接在本仓库审查、暂存并提交公开内容即可；
下述源码导出只是可选快照，不需要维护两份源码。首次提交前必须检查暂存区，
因为忽略规则不会自动排除已经暂存/跟踪的文件。

## 资产分工

| 资产 | 去向 | 原因 |
| --- | --- | --- |
| 当前代码/测试/复现说明/小型结果证据 | GitHub | 可读、可审计、体积小 |
| V3.31机器人policy（原文件约512.81MiB） | HF model repo | 实际主发布权重 |
| manifest/归一化/model config/2000集计分/曲线 | 跟随HF权重 | 推理与结果溯源必需或有价值 |
| Qwen约4.1GiB | 引用官方HF仓库和固定revision | 不重复托管骨干 |
| 基础Qwen缓存约171GiB | 本地保留 | 训练用，推理不需要 |
| V3.29语言/动作侧车约32GiB | 本地保留 | 依赖基础缓存签名 |
| V3.31增强侧车约7.7GiB | 本地保留 | 重新训练时使用 |
| 官方HDF5、LIBERO/Plus模拟资产 | 引用上游获取方式 | 不自动二次上传 |
| optimizer/resume/失败版本/视频/私有日志/环境 | 本地保留 | 不属于首发最小包 |
| DPV4/DS4VLA | 独立路线，后续另发 | 不混入当前DuVLA主模型声明 |

缓存若日后确需共享，单独建HF **dataset repo**，补dataset card、来源许可、Qwen revision、
特征层/相机/时序、manifest和逐shard哈希。当前总量约210GiB，不默认上传。
GitHub不使用Git LFS携带这些训练资产。

## 本地构建

使用已配置PyTorch的隔离环境，在研究仓库根目录运行。每次使用不存在的新目标目录，
导出器拒绝覆盖，失败产物保留以便审计，不自动删除。

```bash
python scripts/prepare_public_release.py model \
  --checkpoint outputs/duvla_v3_31/candidate/duvla-v3.31-30e.pt \
  --train-manifest /your/cache/duvla_v3_27_official2000_highres/manifest.json \
  --evaluation-root outputs/duvla_v3_31/candidate_30e_full2000 \
  --destination outputs/public_release/huggingface/duvla-v3.31

python scripts/prepare_public_release.py source \
  --destination outputs/public_release/github
```

导出器只接受V3.31，先检查四套500唯一状态、
原checkpoint SHA256及主要协议，再规范副本内私人路径、保存模型卡/归一化/结果/曲线。
导出后逐项验证模型state dict与LoRA张量相等。原始权重与原始测评文件均不变。
不会把optimizer/resume或上游Qwen复制到模型包。

源码导出采用`release/source_manifest.json`，同时收集当前入口依赖的本地Python脚本。
共享`src`保留加载所需的兼容实现，历史campaign、旧实验配置、私人日志不进入公开快照。
本机研究目录已更名为`duvla`；旧目录兼容链接已取消，不进入公开快照。
历史checkpoint、缓存manifest、结果和私人学习记录的来源路径/hash保持原样。
历史campaign仍是本机研究入口，不属于跨机器公开接口；公开脚本使用自己的路径参数。

## README / AGENTS / 研究日志

README面向使用者：项目定位、架构、完整结果、快速入口及真实限制。
AGENTS.md面向贡献者和编码助手：数据/测评/变更/发布契约。
原详细研究规则完整保存在忽略的AGENTS.local.md，根AGENTS要求本地工作时继续读取。
学习日志`docs/learning_log.md`完整保留并追加本次整理记录，不默认带入公开快照。
公开快照的研究证据是`release/results.json`和模型包的逐集计分，后续可审核后另发历史研究档案。

## 尚未完成的发布门禁

1. 作者/组织、GitHub和HF目标账号尚未指定；当前没有远程仓库和初始Git提交。
2. 用户已选择MIT，自有代码/策略使用根LICENSE；暂署名DuVLA contributors，
   发布时改成实际GitHub账号。第三方来源及必要版权声明仍需核验，不被MIT自动覆盖。
3. 旧研究文件的暂存条目已备份后撤下，文件仍留本地。每次暂存前后都检查文件清单；
   不用`git add -f`把忽略的私有资料加回。源码导出不是提交的前提。
4. 骨干文件完整hash/来源、LIBERO安装来源commit与本地补丁待核验。
5. 干净目录导入、CPU加载已可验证；干净机器CUDA/OSMesa端到端复现尚未完成。
6. 当前主线已支持新父策略哈希与可配置WSL盘符；来源约束继续保留。见复现说明，
   不把本机独立源码目录冒烟等同于从零安装环境或重新跑完30E/2000集。
7. 公开副本需人工复核来源/署名、历史代码本地路径与隐私。允许列表不等于完成全面安全审计。

上述审核完成前，本包标为**发布候选**；MIT文件已提供，作者账号/第三方审计仍须完成。

## 最后上传（本次未执行）

先在GitHub创建空仓库；在当前duvla仓库审查公开文件、暂存后检查差异，再提交并推送。
已有Git元数据，无需重新初始化；本轮没有实际暂存新内容、提交或推送。
HF先创建private model repo，上传模型包，检查下载加载后再切public。
HF的README.md是model card；源码README与模型卡分开维护。
使用独立发布环境或已兼容的Hub CLI，先用`hf upload --help`核验版本；不为上传升级评测环境。

```bash
# 仅示例，在目标账号/许可确定且本地检查完成后执行。
hf upload YOUR_ACCOUNT/duvla-v3.31 ./outputs/public_release/huggingface/duvla-v3.31 . --repo-type model
```

上传后记录HF revision、Git commit/tag和全部文件hash，再替换README中的待发布状态。
不编造尚不存在的模型下载地址、论文、作者或榜单排名。

## 官方参考

- [HF模型卡](https://huggingface.co/docs/hub/model-cards)
- [HF上传文件](https://huggingface.co/docs/huggingface_hub/guides/upload)
- [HF数据集上传](https://huggingface.co/docs/hub/datasets-adding)
- [Qwen模型来源](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
- [LIBERO源码](https://github.com/Lifelong-Robot-Learning/LIBERO)
