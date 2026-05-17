# 仓库规范化计划

## 目标

把当前“已经能跑通实验”的仓库整理成“多名组员可并行协作、结果可追溯、接手成本低”的科研工程仓库。重点不是重写全部代码，而是先把数据、训练、评测、文档、实验记录之间的边界理顺。

## 基本约束

- 不打断已经验证过的 `s800` / `s800_think` 主线实验
- 保留现有能复现结果的脚本，优先做增量整理
- 让后续负责造数据和深度评测的同学直接复用现有代码

## 第一阶段：目录职责与命名规范

### 目录职责

- `data/raw`
  - 外部来源数据
  - 原始公开数据集
  - 人工整理但未清洗的原始材料
- `data/interim`
  - 对齐后的中间数据
  - 增补了字段但还未定稿的数据
  - 风格转换、字段补全、结构重组后的版本
- `data/final`
  - 可直接进入训练或评测的数据版本
  - 必须具备稳定 schema
- `src`
  - 可复用核心逻辑
  - 不放一次性实验入口
- `scripts`
  - CLI 入口脚本
  - 只保留薄封装，不堆主要业务逻辑
- `runs`
  - 训练、评测、抽样、分析输出
- `docs`
  - 项目说明、计划、进度、组员同步、数据格式说明

### 命名规范

- 数据集名称统一使用 `dataset_tag`
- `dataset_tag` 需要显式编码四类信息：
  - 数据来源
  - 任务风格
  - 数据规模
  - 版本号

推荐格式：

```text
{source}_{task}_{size}_{version}
```

示例：

- `synth_structured_800_v1`
- `math23k_structured_2000_v1`
- `math23k_think_2000_v2`
- `mixcn_think_5000_v3`

补充规则：

- `source`
  - 表示来源或来源组合
  - 例如 `synth`、`math23k`、`mixcn`
- `task`
  - 表示监督任务
  - 例如 `structuredthink`、`think`、`shortthink`
- `size`
  - 表示最终可训练样本规模
  - 必须写绝对数值，不写 `small` / `large`
- `version`
  - 表示同一数据方向下的数据迭代版本
  - 必须从 `v1` 开始单调递增

### 训练 run 命名规范

训练目录统一格式：

```text
{timestamp}_train_{dataset_tag}_{model_tag}_{exp_tag}
```

示例：

- `20260517_002714_train_synth_think_800_v1_qwen3.5-0.8b_think-pilot`
- `20260518_103000_train_math23k_structured_2000_v1_qwen3.5-0.8b_baseline`

其中：

- `timestamp`
  - 绝对时间，精确到秒
- `dataset_tag`
  - 必须与训练输入数据一致
- `model_tag`
  - 例如 `qwen3.5-0.8b`
- `exp_tag`
  - 表示实验意图，如 `baseline`、`think-pilot`、`ablation-no-classical`

### 评测 run 命名规范

评测目录统一格式：

```text
{timestamp}_eval_{dataset_tag}_{model_tag}_{eval_tag}
```

示例：

- `20260517_113430_eval_synth_think_800_v1_qwen3.5-0.8b_rule`
- `20260518_141500_eval_math23k_think_2000_v2_qwen3.5-0.8b_llmreview`

### 抽样与案例分析 run 命名规范

抽样目录统一格式：

```text
{timestamp}_inspect_{dataset_tag}_{model_tag}_{inspect_tag}
```

示例：

- `20260517_113359_inspect_synth_think_800_v1_qwen3.5-0.8b_checkpoint40`

## 第二阶段：脚本边界重构

- 把当前可复用逻辑从 `scripts/` 抽到 `src/`
- `src/` 建议拆成：
  - `src/datasets/`
  - `src/training/`
  - `src/evaluation/`
  - `src/analysis/`
  - `src/common/`
- 当前建议抽出的公共逻辑：
  - 数据构造
  - prompt 渲染
  - 训练参数解析
  - 答案抽取
  - 评测汇总
- `scripts/` 只保留入口，不再堆主要业务逻辑
- 所有入口脚本统一支持：
  - `--dataset-tag`
  - `--run-tag`
  - `--output-dir` 或默认输出规则

## 第三阶段：实验追溯规范

- 每次训练必须保存：
  - `run_config.json`
  - 数据集路径
  - checkpoint 路径
  - 关键训练参数
  - 核心评测结果引用
- 每次评测必须保存：
  - 输入模型
  - 输入测试集
  - 生成参数
  - summary json
  - prediction 文件
- README 和 `PROJECT_PROGRESS.md` 只写高层结论
- 详细过程与具体文件路径统一落在：
  - `runs/`
  - `TEAM_SYNC_LOG.md`

## 第四阶段：面向组员协作的可扩展性

### 为数据构造同学预留

- 数据源适配层
- 风格转换规则层
- schema 校验层
- 数据版本登记说明

### 为深度评测同学预留

- think 长度指标
- think 质量指标
- 超短窗口评测
- LLM-as-judge 评测适配器

### 需要补齐的文档

- 数据格式说明
- 新增实验的最小操作流程
- 结果记录模板
- 常见命名示例

## 交付物

1. 目录职责重整，不改主实验结果
2. `src/` 初版落地
3. 训练 / 评测 / 抽样入口参数风格统一
4. 数据 schema 文档
5. 组员协作说明与新增实验模板

## 推荐执行顺序

1. 先整理目录与命名，不碰算法逻辑
2. 再抽 `src/` 公共逻辑
3. 再补数据 / 评测接口文档
4. 最后让其他同学在新结构上继续扩展
