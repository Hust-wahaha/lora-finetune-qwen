# 第二阶段执行与协作规范

本文档是项目第二阶段的统一执行入口。目标不是让每位同学理解全部细节，而是让每个人只看与自己直接相关的任务，也不会做出无法合并进主仓的工作。

## 一、当前项目阶段

当前主线已经完成两件关键事情：

- `s800 / s800_think` 已证明显式 `think` 监督可行
- 当前 0.8B 主线代码已经具备数据生成、LoRA 训练、规则评测与 `DeepSeek` 复核的基本闭环

现阶段不再回答“think 监督能不能起作用”，而是进入第二阶段：

- 扩充更有公信力、分难度的数据集
- 系统比较 `modern_think / classical_think / structured_think`
- 扩展到 `2B / 4B` 模型
- 统一 Benchmark 协议，保证后续所有结果可横向比较

## 二、全组统一限制条件

以下限制是强制性的，所有人都必须遵守。

### 1. 不改主字段命名

- 所有显式思维监督字段统一使用 `xxx_think`
- 不新增 `xxx_cot`
- `structured_cot` 只允许作为历史兼容字段存在

### 2. 不破坏现有主线脚本默认行为

当前主线脚本是：

- `scripts/generate_dataset.py`
- `scripts/validate_dataset.py`
- `scripts/train_lora_local.py`
- `scripts/eval_compare_full.py`
- `scripts/inspect_think_samples.py`

新增能力优先通过以下方式实现：

- 增加新脚本
- 增加可选参数
- 保持旧默认值不变

不要直接改坏现有 `s800 / s800_think` 训练与评测闭环。

### 3. 数据集必须可追溯

- train / val / test 必须显式区分
- 新数据必须写清 `source`、`split`、`family`
- 不允许把 few-shot 示例题、模板原题直接泄漏到测试集

### 4. Benchmark 口径不得私自改动

当前统一基线评测入口是 `scripts/eval_compare_full.py`。

未经统一确认，不要私自修改以下核心口径：

- `max_tokens`
- 答案抽取规则
- baseline / finetuned 对照方式
- `DeepSeek` 复核模式

如需扩展指标，请新开脚本或加开关，不要覆盖现有口径。

### 5. 所有结果必须落文档

- 正式结论写入 `docs/PROJECT_PROGRESS.md`
- 个人推进、实验目录、遗留问题写入 `docs/TEAM_SYNC_LOG.md`

## 三、每位同学的任务

## 汤子墨

### 任务目标

收敛 `structured_think` 风格，解决第一轮结构化思维文本风格发散的问题。

### 具体交付物

- 一组 `7-8` 条高质量 few-shot 样例
- 一版新的 `structured_think` 数据生成规范
- 一版重生成后的结构化数据集

### 必须遵守

- 不修改 `modern_think`、`classical_think` 字段定义
- 结构化输出必须继续服务于 `structured_think`
- 风格优化优先追求“精简、稳定、可校验”，不要追求花哨表达

### 与代码的衔接要求

- 新数据必须能被 `scripts/validate_dataset.py` 校验通过
- 最终训练样本仍需兼容当前 `messages` 格式
- 如果新增生成脚本，不能破坏现有 `generate_dataset.py`

## 白泽鑫

### 任务目标

把当前仅支持 `Qwen/Qwen3.5-0.8B` 的训练/评测主线扩展到 `2B / 4B`。

### 具体交付物

- 可配置模型规模的训练脚本
- 可配置模型规模的评测脚本
- 至少一条 `2B` 和一条 `4B` 的完整参考 run

### 必须遵守

- 当前 `0.8B` 默认行为必须保留
- 不允许把现有脚本改成“只能跑 2B/4B”
- 所有新增参数都要有默认值，且默认仍指向 `0.8B`

### 与代码的衔接要求

当前 `scripts/train_lora_local.py` 和 `scripts/eval_compare_full.py` 都写死了 `0.8B` 模型标识。你的任务是把它们改成“可扩展”，而不是另起一套脱离主线的新工程。

## 颜艺晨

### 任务目标

构建统一 Benchmark 协议，并在新数据到位后负责标准评测。

### 具体交付物

- ✅ `scripts/eval_five_metrics.py` — 三指标 + max_tokens 扫描评测脚本（已完成）
- ✅ `src/common/style_detect.py` — 指标判定纯函数集合（已完成）
- ✅ 2×3 折线网格可视化图（`metrics/metrics_sweep.pdf/png`）（已完成）
- 待执行：新数据（gsm-1k 等）到位后的正式评测结果

### 当前冻结的三指标

| 指标名 | 衡量内容 |
|---|---|
| `answer_accuracy` | 从输出末尾抽取阿拉伯数字，与 gold 字符串做精确匹配 |
| `cot_completeness_rate` | `<think>…</think>` 标签是否成对出现（推理段未被截断） |
| `generation_completion_rate` | 在前者基础上，整段以「N。」（数字+句号）收尾，整条没被截 |

> 注：脚本名 `eval_five_metrics` 是历史命名，初版为 5 指标，后精简为当前 3 个；指标实现统一在 `src/common/style_detect.py`，不在脚本内重复定义正则。

### 必须遵守

- `eval_compare_full.py` 是基线，不要直接推翻；二者并行：前者负责 baseline/finetuned 对照与 DeepSeek 复核，后者负责 max_tokens 扫描与完整性分析
- 新指标以”新增开关”方式接入（`style_detect.py` 中的 `style_label` / `english_leak_count` 等函数已备用）
- 不在 `scripts/` 里重复定义正则或判定逻辑，统一走 `src/common/style_detect.py`

### 与代码的衔接要求

当前仓库已有的稳定评测资产：

- `eval_compare_full.py`：规则答案抽取 + DeepSeek 复核 + baseline/finetuned 对照
- `eval_five_metrics.py`：三指标 × max_tokens × 多模型 扫描 + PDF/PNG 可视化
- `src/common/style_detect.py`：`is_cot_complete` / `is_generation_complete` / `extract_final_answer` / `style_label` / `english_leak_count`

后续如需加”思维风格命中率”或”英文漏出率”统计，直接调用 `style_detect.py` 的现有函数即可，无需重写。

## 刘佳为

### 任务目标

扩展中学与高中层数据，重点补“过程较长、适合 think 压缩研究”的题。

### 具体交付物

- 一版中学难度数据
- 一版高中难度数据
- 一版“思路不难但步骤较长”的压力测试数据

### 必须遵守

- 目标是测试压缩边界，不是单纯追求难题
- 题目必须有稳定标准答案
- 优先保证三类 `think` 字段能够被稳定构造

### 与代码的衔接要求

- 数据字段先满足 `DATA_SCHEMA.md` 的最低要求
- 进入主仓前必须过 `validate_dataset.py`
- 数据集标签、目录名、版本号按 `REPOSITORY_RULES.md` 执行

## Zhuoya Wang_with_codex

### 任务目标

负责总线维护、任务整合、冲突裁决与最终汇报口径统一。

### 具体职责

- 冻结第二阶段实验矩阵
- 审核新数据是否满足主线 schema 与 split 纪律
- 审核训练与评测脚本是否破坏兼容性
- 统一 README、进度文档与最终汇报叙事

## 四、协作注意事项

### 1. 不要各自维护私有口径

凡是影响全组可比性的内容，都必须统一：

- 数据字段名
- 训练 run 命名
- 评测命令
- 指标定义

### 2. 不要把“能跑通”当成“可合并”

提交前至少自查三件事：

- 是否破坏旧主线
- 是否写清输入输出路径
- 是否补了文档

### 3. 不要跳过追溯信息

每次新增实验，至少要记录：

- 命令
- 数据集标签
- 输出目录
- 结论

### 4. 不要私自覆盖他人工作

- 涉及公共脚本时，优先加参数，不直接改默认逻辑
- 涉及文档时，只追加、补充、重构，不抹掉历史结论

## 五、建议执行顺序

1. 汤子墨先稳定 `structured_think` 风格
2. 刘佳为并行补中学 / 高中 / 长过程数据
3. 白泽鑫把 `0.8B -> 2B / 4B` 模型轴打通
4. 颜艺晨在新数据与新模型到位后统一跑 Benchmark
5. Zhuoya Wang_with_codex 负责收敛口径并汇总结果

## 六、这份文档如何使用

- 普通组员：只看“自己的任务”与“全组统一限制条件”
- 负责整合的人：重点看“限制条件”“协作注意事项”“建议执行顺序”
- 新加入同学：先读 `README.md`，再读本文档，再读 `DATA_SCHEMA.md`
