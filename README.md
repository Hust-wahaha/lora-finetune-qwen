# 中文数学应用题 LoRA 微调实验仓库

> 华中科技大学人工智能与自动化学院 2023 级课程设计项目  
> 研究主题：面向低参数中文模型的数学应用题数据集构建与 LoRA 微调

本仓库当前聚焦一条明确主线：基于自建中文数学应用题数据集，对 `Qwen/Qwen3.5-0.8B` 进行 LoRA 微调，并比较现代汉语题面、文言题面、结构化解题过程监督与显式 `think` 监督对模型行为的影响。

## 当前状态

- 当前可见输出监督主线：
  - `modern_question -> structured_think + answer`
  - `classical_question -> structured_think + answer`
- 当前主线模型：`runs/20260515_210429/checkpoints/checkpoint-40`
- 已完成一次更可信的全量评测：
  - Rule-based baseline：`55.0%`
  - Rule-based finetuned：`96.25%`
- 已补充显式 `think` 监督主线：
  - `modern_question -> <think>modern_think</think> + answer`
  - `classical_question -> <think>classical_think</think> + answer`
- 当前基准数据集：`s800`
- 当前显式思维监督数据集：`s800_think`
- 已完成第一轮显式 `think` 监督实验：
  - checkpoint：`runs/20260517_002714_think_pilot/checkpoints/checkpoint-40`
  - `think` 风格对齐：成功
  - `test_s800_think` 规则评测：baseline `55.0%`，finetuned `100%`

说明：早期 `max_tokens=256` 的评测会被长 `<think>` 截断低估；当前应以 `max_tokens=512` 版本结果为准。后续正式口径改为“规则评测 + DeepSeek V4 Flash 复核错例”。

## 仓库结构

```text
.
├── data/
│   ├── final/                 # 训练/验证/测试集与数据摘要
│   └── interim/               # 对齐后的中间数据
├── docs/
│   ├── README.md              # 文档导航
│   ├── PROJECT_PROGRESS.md    # 主线科研进度与关键结论
│   └── TEAM_SYNC_LOG.md       # 小组协作接龙文档
├── src/                       # 公共路径、命名、schema 等可复用逻辑
├── runs/                      # 训练与评测产物
├── scripts/                   # 数据生成、训练、评测脚本
├── ms-swift/                  # 训练框架依赖
└── README.md
```

## 关键脚本

- `scripts/generate_dataset.py`
  - `python scripts/generate_dataset.py --variant visible`
  - `python scripts/generate_dataset.py --variant think --think-style by_view`
- `scripts/validate_dataset.py`
  - `python scripts/validate_dataset.py data/final/train_s800.jsonl`
  - `python scripts/validate_dataset.py data/final/train_s800_think.jsonl --expect-think-tags`
- `scripts/train_lora_local.py`
  - `python scripts/train_lora_local.py --dataset-tag s800`
  - `python scripts/train_lora_local.py --dataset-tag s800_think --run-tag think_pilot`
- `scripts/eval_compare_full.py`
  - 两阶段评测入口，先做规则抽取，再用 `DeepSeek V4 Flash` 复核错例或全量样本
- `scripts/inspect_think_samples.py`
  - 抽样查看 base / finetuned 模型在白话题、文言题上的 `think` 输出

## 推荐阅读与协作顺序

1. 先读 [docs/README.md](docs/README.md)
2. 再看 [docs/PROJECT_PROGRESS.md](docs/PROJECT_PROGRESS.md)
3. 所有组员协作更新统一写入 [docs/TEAM_SYNC_LOG.md](docs/TEAM_SYNC_LOG.md)

## 参考 Run

- `runs/20260517_195355_train_s800_think_qwen3.5-0.8b_smoke_ref`
  - 用途：教学型全链路样板
  - 特点：高频保存、高频评测，适合看目录结构和产物组成
- `runs/20260517_203504_train_s800_think_qwen3.5-0.8b_reference_v1`
  - 用途：正式训练参数参考
  - 特点：评测频率更合理，已完整跑通并正常收敛，适合后续组员直接仿照开新实验

说明：

- 不要直接照抄 `smoke_ref` 的 `eval_steps=1` 配置去跑正式实验
- AutoDL 上建议显式使用 `.venv/bin/python`，不要依赖交互 shell 的 PATH

## 为什么显式 Think 监督重要

之前的数据只监督 assistant 可见输出，Swift/Qwen 模板会自动补一个空的 `<think></think>` 区块，因此模型不会学到“中文/文言 think 内容本身”。现在的 `s800_think` 显式把目标思维文本放进 `<think>...</think>`，可以直接验证：

- 微调后 `think` 是否从英文转成中文
- 白话题面是否对应白话思维
- 文言题面是否对应文言思维
- 答案正确率是否仍然保持稳定

说明：从命名规范上，仓库后续统一使用 `xxx_think` 表示“显式监督的思维内容”。旧名称 `structured_cot` 仅视作历史兼容口径，后续规范字段名统一改为 `structured_think`。

## 最新 Think 结果

第一轮 think-supervised pilot 已经证明这条路线有效：

- 定性结果：
  - 白话题输出白话 `think`
  - 文言题输出文言 `think`
  - 不再出现默认英文 `think`
- 定量结果：
  - `test_s800_think` rule-based baseline：`44/80 = 55.0%`
  - `test_s800_think` rule-based finetuned：`80/80 = 100%`
  - finetuned `answer_marker_rate = 1.0`
  - finetuned `avg_chars = 50.96`

示例：

```text
Q: 小明有12颗糖，送给小红5颗，还剩几颗？
<think>
原来有12颗糖，送给小红5颗，所以用减法：12-5=7。
</think>

答案：7。
```

```text
Q: 明有糖12颗，遗红5颗，尚余几何？
<think>
初有12颗，遗5颗，故12-5=7。
</think>

答案：7。
```

## 运行环境

- AutoDL
- Python `3.12`
- PyTorch `2.8.0`
- CUDA `12.8`
- GPU：RTX 4090 24GB

## 说明

- `runs/`、`artifacts/` 属于实验产物目录，新增实验必须保证目录名可追溯。
- 新结论先写入 `docs/PROJECT_PROGRESS.md`，再决定是否进入 README。
- 组员之间不要私聊式同步实验细节，统一落到 `docs/TEAM_SYNC_LOG.md`。
- 正式汇报时优先区分三类指标：`Rule-based Accuracy`、`LLM-reviewed Accuracy`、`Answer Marker Rate`。
