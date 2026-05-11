# LoRA 微调 Qwen3.5-0.8B

> 华科人工智能与自动化学院 2023 级 Hust wahaha 小组 · 模式识别课程设计

基于 [MS-SWIFT](https://github.com/modelscope/ms-swift) 框架，使用 **LoRA** 对 **Qwen3.5-0.8B** 进行指令微调。

---

## 🏗 项目结构

```
lora-finetune-qwen/
├── tune_qwen.py          # 训练入口脚本
├── ms-swift/             # MS-SWIFT 训练框架
├── output/               # 模型输出（checkpoint、日志）
├── .gitignore
└── README.md
```

## 🚀 快速开始

### 环境（AutoDL）

| 组件 | 版本 |
|------|------|
| OS | Ubuntu 22.04 LTS |
| Python | 3.12.3 (Miniconda) |
| PyTorch | 2.8.0 + CUDA 12.8 |
| MS-SWIFT | 最新 |
| GPU | 按需开机挂载 |

### 安装依赖

```bash
cd qwen_ws
source .venv/bin/activate
pip install -e ms-swift/
```

### 开始训练

```bash
python tune_qwen.py
```

训练超参在 `tune_qwen.py` 中调整：

| 参数 | 值 |
|------|-----|
| 模型 | `Qwen/Qwen3.5-0.8B` |
| LoRA rank | 8 |
| LoRA alpha | 32 |
| 学习率 | 1e-4 |
| batch size | 1 × grad_accum 16 |
| 训练轮数 | 1 |
| 序列长度 | 2048 |

### 训练数据

- `AI-ModelScope/alpaca-gpt4-data-zh`（中文指令）
- `AI-ModelScope/alpaca-gpt4-data-en`（英文指令）
- `swift/self-cognition`（自认知）

## 👥 小组成员

| 姓名 | GitHub | 分工 |
|------|--------|------|
| uDeserve | [@uDeserve](https://github.com/uDeserve) | PM / 统筹 |
| 队友2 | - | 待分配 |
| 队友3 | - | 待分配 |
| 队友4 | - | 待分配 |
| 队友5 | - | 待分配 |

## 📝 更新日志

- **2026-05-11** 项目初始化，代码推送到 GitHub 组织仓库
