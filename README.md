# LoRA Fine-Tuning for Chinese Math Word Problems

> 华中科技大学人工智能与自动化学院 2023 级课程设计项目  
> 研究主题：面向低参数中文模型的数学应用题数据集构建与 LoRA 微调

本仓库当前聚焦一条明确主线：基于自建中文数学应用题数据集，对 `Qwen/Qwen3.5-0.8B` 进行 LoRA 微调，并比较现代汉语题面、文言题面与结构化解题过程监督对模型行为的影响。

## Current Status

- 当前主线数据方向：
  - `modern_question -> structured_cot + answer`
  - `classical_question -> structured_cot + answer`
- 当前主线数据集：`s800`
- 当前主线模型：`runs/20260515_210429/checkpoints/checkpoint-40`
- 已完成一次更可信的全量评测：
  - baseline：`55.0%`
  - finetuned：`96.25%`

说明：早期 `max_tokens=256` 的评测会被长 `<think>` 截断低估；当前应以 `max_tokens=512` 版本结果为准。

## Repository Layout

```text
.
├── data/
│   ├── final/                 # 训练/验证/测试集与数据摘要
│   └── interim/               # 对齐后的中间数据
├── docs/
│   ├── README.md              # 文档导航
│   ├── PROJECT_PROGRESS.md    # 主线科研进度与关键结论
│   └── TEAM_SYNC_LOG.md       # 小组协作接龙文档
├── runs/                      # 训练与评测产物
├── scripts/                   # 数据生成、训练、评测脚本
├── ms-swift/                  # 训练框架依赖
└── README.md
```

## Key Scripts

- `scripts/generate_dataset.py`：生成当前 `s800` 数据集
- `scripts/train_lora_local.py`：启动当前主线 LoRA 训练
- `scripts/eval_compare_full.py`：baseline vs finetuned 全量对比评测
- `scripts/validate_dataset.py`：数据结构与字段校验

## Recommended Workflow

1. 先读 [docs/README.md](docs/README.md)
2. 再看 [docs/PROJECT_PROGRESS.md](docs/PROJECT_PROGRESS.md)
3. 所有组员协作更新统一写入 [docs/TEAM_SYNC_LOG.md](docs/TEAM_SYNC_LOG.md)

## Environment

- AutoDL
- Python `3.12`
- PyTorch `2.8.0`
- CUDA `12.8`
- GPU：RTX 4090 24GB

## Notes

- `runs/`、`artifacts/` 属于实验产物目录，新增实验必须保证目录名可追溯。
- 新结论先写入 `docs/PROJECT_PROGRESS.md`，再决定是否进入 README。
- 组员之间不要私聊式同步实验细节，统一落到 `docs/TEAM_SYNC_LOG.md`。
