# 贡献边界与源码来源

上游：shibing624/MedicalGPT，原项目 remote 仍指向上游；保留 LICENSE、CITATION.cff 和原始训练文件。`training/` 中 PT/SFT/DPO/PPO/RM/GRPO/ORPO/OPD 的主体不是本次从零新增。

本地已有个人修改（以原仓库 git status 为线索，不等同于已审核的完整 diff）：医疗数据构造、候选/偏好数据构造、SFT 数据混合、RM/PPO 启动脚本，以及部分 training 脚本。

本次新增：`src/medical_alignment/`、`src/alignment/`、`scripts/audit_alignment.py`、`scripts/train_safe_dpo.py`、独立配置/示例/测试与实验协议。本次修改：旧候选构造脚本改为从 GLM_API_KEYS / DEEPSEEK_API_KEYS 环境变量读取；损坏的关键词评测由审阅标注聚合入口替换；旧 PPO launcher 保留 cache 且要求独立 validation 路径。

不能声称：从零开发 MedicalGPT 全框架；已在本地重现简历所有多卡训练与 AAR 数字；已得到临床安全保证；已经核验了 RLOO 的训练成绩（历史 `ppo_training.py` 实际使用 RLOOTrainer；本次移至 `rloo_training.py` 并保留兼容入口，但日志仍缺失）。
