# 团队协作进度日志

本文件是全组统一的接龙式进度文档。所有组员都只在文末追加，不覆盖、不删除他人内容。

## 目的

- 降低组员之间的沟通成本
- 保证任何人都能快速接手上一位同学的工作
- 避免“知道做了什么，但不知道为什么、改了哪里、结果如何”

## 强制更新规则

- 每完成一段可汇报工作，就追加一条记录
- 时间必须写绝对时间，统一格式：`YYYY-MM-DD HH:MM`
- 必须写清楚文件路径、实验目录、结果结论
- 结论和问题都要写，不能只写“已完成”
- 发现前文有误时，新增一条“勘误/修正”，不要直接改写旧记录
- 新条目统一追加到文末，按时间自然接龙

## 记录模板

将下面模板复制到文末直接填写：

```md
## [姓名] | [YYYY-MM-DD HH:MM]

### 本次工作
- 

### 为什么这样做
- 

### 修改/涉及文件
- 

### 实验或运行信息
- 机器：
- 命令：
- 输出目录：

### 结果与结论
- 

### 风险 / 遗留问题
- 

### 建议下一步
- 
```

## 填写标准

- “本次工作”写动作，不写空话
- “为什么这样做”写动机，不写“老师要求”
- “修改/涉及文件”必须写仓库内相对路径
- “结果与结论”必须包含可复用信息，例如准确率、失败模式、异常现象
- “建议下一步”要写到下一个同学可以直接开做

## 示例

## Codex | 2026-05-17 18:00

### 本次工作
- 跑通 `s800_think` 的显式 `think` 监督
- 补齐仓库中文文档与协作规范

### 为什么这样做
- 需要先验证“监督 `<think>` 内容本身”这条主线是否成立
- 后续会有同学分工造数据和做评测，仓库必须先变成可协作状态

### 修改/涉及文件
- `scripts/generate_dataset.py`
- `scripts/train_lora_local.py`
- `scripts/inspect_think_samples.py`
- `docs/PROJECT_PROGRESS.md`

### 实验或运行信息
- 机器：AutoDL RTX 4090 24GB
- 命令：`python scripts/train_lora_local.py --dataset-tag s800_think --run-tag think_pilot`
- 输出目录：`runs/20260517_002714_think_pilot`

### 结果与结论
- 微调后白话题输出白话 `think`
- 微调后文言题输出文言 `think`
- `test_s800_think` 规则评测从 `55.0%` 提升到 `100%`

### 风险 / 遗留问题
- 当前数据规模仍小，暂时不能代表跨分布泛化
- 更细的 `think` 质量评测脚本仍待补齐

### 建议下一步
- 继续扩充数据集版本，同时保持命名与 schema 稳定
- 让评测同学基于统一目录和字段规范补深度指标

## Codex | 2026-05-17 20:35

### 本次工作
- 在 AutoDL 上补跑一条全链路 smoke 训练参考
- 启动一条正式训练参考 run，供后续组员直接复用
- 验证远端仓库新架构、虚拟环境、数据生成与校验链路可用

### 为什么这样做
- 仅有“脚本能启动”还不够，组员更需要看到完整训练产物长什么样
- smoke run 用来证明仓库现在能从数据到 checkpoint 全链路闭环
- formal reference run 用来提供标准参数模板，避免后续同学直接照搬不合理的 smoke 评测频率

### 修改/涉及文件
- `docs/PROJECT_PROGRESS.md`
- `docs/TEAM_SYNC_LOG.md`
- 远端实验目录：`runs/20260517_195355_train_s800_think_qwen3.5-0.8b_smoke_ref`

### 实验或运行信息
- 机器：AutoDL RTX 4090 24GB
- smoke 命令：`.venv/bin/python scripts/train_lora_local.py --dataset-tag s800_think --run-tag smoke_ref --max-length 512 --num-train-epochs 0.05 --gradient-accumulation-steps 8 --save-steps 1 --eval-steps 1 --logging-steps 1`
- smoke 输出目录：`runs/20260517_195355_train_s800_think_qwen3.5-0.8b_smoke_ref`
- formal 命令：`.venv/bin/python scripts/train_lora_local.py --dataset-tag s800_think --run-tag reference_v1 --max-length 1024 --learning-rate 1e-4 --train-batch-size 1 --eval-batch-size 1 --gradient-accumulation-steps 16 --num-train-epochs 1.0 --save-steps 25 --eval-steps 25 --logging-steps 5`
- formal 输出目录：待运行完成后补记

### 结果与结论
- smoke 训练已完整跑通
- smoke run 证明以下链路都正常：
  - 数据读取
  - Swift 模板编码
  - LoRA 挂载
  - CUDA 训练
  - checkpoint 保存
  - run_config / last_checkpoint 追溯文件生成
- smoke 最终结果：
  - `train_runtime = 445.4s`
  - `train_loss = 1.652`
  - `eval_loss = 1.083`
  - `eval_token_acc = 0.7705`
- smoke 同时暴露出一个重要经验：
  - `eval_steps=1` 虽然适合教学观察，但会显著拖慢总时长，不能作为正式默认配置

### 风险 / 遗留问题
- formal reference run 仍在进行中，完成后需要补充最终 run 目录与结果摘要
- 远端命令应统一显式使用 `.venv/bin/python`，不要依赖交互 shell 的 PATH

### 建议下一步
- 等 formal reference run 完成后，把其输出目录、核心指标和推荐使用场景补进 README 与进度文档
- 后续组员新增训练实验时，优先参考 formal run，不要直接照抄 smoke 参数

## Codex | 2026-05-17 22:21

### 本次工作
- 补记 `reference_v1` 正式参考 run 的最终结果
- 将仓库规范文档改名为更直接的规则文档 `REPOSITORY_RULES.md`

### 为什么这样做
- 队友需要直接可执行的现行规则，而不是继续看带有 `plan` 语义的文档名
- 正式参考 run 已完成，需要把它补进文档，作为后续训练模板

### 修改/涉及文件
- `README.md`
- `docs/README.md`
- `docs/PROJECT_PROGRESS.md`
- `docs/TEAM_SYNC_LOG.md`
- `docs/PHASE1_EXECUTION_CHECKLIST.md`
- `docs/REPOSITORY_RULES.md`

### 实验或运行信息
- 机器：AutoDL RTX 4090 24GB
- formal 输出目录：`runs/20260517_203504_train_s800_think_qwen3.5-0.8b_reference_v1`

### 结果与结论
- `reference_v1` 已正常完成
- 最终 checkpoint：
  - `checkpoint-40`
- 最终训练结果：
  - `train_runtime = 1322.5324s`
  - `train_loss = 0.27809329`
- 最终验证结果：
  - `eval_loss = 0.00075078`
  - `eval_token_acc = 1.0`
- 后续正式训练应优先参考这条 run，而不是 `smoke_ref`

### 风险 / 遗留问题
- 当前文档与仓库结构已经适合组员接手，但更深入的评测维度仍待后续组员扩展

### 建议下一步
- 后续若新增数据方向，先保证 `dataset_tag`、`xxx_think` 命名和 schema 规则不被破坏
- 新实验默认沿用 `reference_v1` 参数，再按实验目标做局部改动
