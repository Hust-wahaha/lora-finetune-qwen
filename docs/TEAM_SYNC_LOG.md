# Team Sync Log

本文件是全组统一协作文档。所有组员都只在这个文件末尾追加，不覆盖、不改写他人内容。

## Purpose

- 降低队友之间的沟通成本
- 保证任何人都能快速接手上一位同学的工作
- 避免“只知道做了什么，不知道为什么、改了哪里、结论是什么”

## Mandatory Update Rules

- 每次完成一段可汇报工作，就追加一条记录
- 记录必须写绝对时间，格式统一：`YYYY-MM-DD HH:MM`
- 记录必须写清楚文件路径、实验目录、结果结论
- 结论和问题都要写，不能只写“已完成”
- 禁止删除他人条目；若发现错误，新增一条“勘误/修正”说明
- 新条目统一追加到文末，按时间自然接龙

## Entry Template

复制下面模板，直接接在文末：

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

## Writing Standard

- “本次工作”写动作，不写空话  
  例：`补跑 s800 全量 baseline vs finetuned 评测`

- “为什么这样做”写动机  
  例：`之前 256 token 评测存在长 think 截断，结果可能被低估`

- “修改/涉及文件”必须给相对路径  
  例：`scripts/eval_compare_full.py`、`docs/PROJECT_PROGRESS.md`

- “结果与结论”必须包含可复用信息  
  例：准确率、实验目录、异常现象、解释判断

- “建议下一步”要让下一个同学可以直接开做

## Example Entry

## Codex | 2026-05-16 15:20

### 本次工作
- 补跑 `s800` 全量 baseline vs finetuned 正式评测
- 修正文档结构并新增团队协作文档

### 为什么这样做
- 之前 `max_tokens=256` 的评测会被长 `<think>` 截断，不能作为最终判断
- 组内需要统一的接龙式进度文档，降低同步成本

### 修改/涉及文件
- `scripts/eval_compare_full.py`
- `docs/PROJECT_PROGRESS.md`
- `docs/TEAM_SYNC_LOG.md`
- `README.md`

### 实验或运行信息
- 机器：AutoDL RTX 4090 24GB
- 命令：`python scripts/eval_compare_full.py --checkpoint runs/20260515_210429/checkpoints/checkpoint-40 --max-tokens 512 --tag compare_s800_full_mt512`
- 输出目录：`runs/20260516_150218_compare_s800_full_mt512`

### 结果与结论
- baseline：`55.0%`
- finetuned：`96.25%`
- `s800` 在自有分布上已经证明方向有效，当前短板转为输出控制而非算术能力

### 风险 / 遗留问题
- 模型仍然普遍保留 `<think>` 长输出
- classical 视图仍弱于 modern 视图

### 建议下一步
- 做下一版数据集时优先增强 classical 鲁棒性
- 同时优化目标格式，降低对长 `<think>` 的依赖

## Codex | 2026-05-16 16:30

### 本次工作
- 将正式评测管线升级为“规则评测 + DeepSeek V4 Flash 复核错例”两阶段结构
- 对 `runs/20260516_150218_compare_s800_full_mt512` 的现有预测结果做了一轮完整错例复核

### 为什么这样做
- baseline 长回答中存在大量“模型其实答对，但规则提取失败”的情况
- 仅靠数值抽取会严重低估 baseline 真实答题能力，也会影响后续数据方向判断

### 修改/涉及文件
- `scripts/eval_compare_full.py`
- `README.md`
- `docs/PROJECT_PROGRESS.md`
- `docs/TEAM_SYNC_LOG.md`

### 实验或运行信息
- 机器：AutoDL RTX 4090 24GB
- 命令：`python scripts/eval_compare_full.py --existing-run-dir runs/20260516_full_llm_review --llm-review-mode mismatches`
- 输出目录：`runs/20260516_full_llm_review`

### 结果与结论
- baseline：规则口径 `55.0%`，LLM 复核后 `90.0%`
- finetuned：规则口径 `96.25%`，LLM 复核后 `100%`
- baseline 共复核 `36` 条规则错例，其中 `28` 条被改判为正确
- finetuned 共复核 `3` 条规则错例，`3` 条全部被改判为正确
- 说明 baseline 绝对值此前被明显低估，而 finetuned 在当前分布上已经接近饱和

### 风险 / 遗留问题
- baseline 的 classical 视图仍明显弱于 modern 视图
- 即使 finetuned 已基本全对，输出仍偏长，`<think>` 依赖依然存在

### 建议下一步
- 下一轮数据设计继续重点补强 classical 鲁棒性
- 评测汇报时必须同时给出 `Rule-based Accuracy` 和 `LLM-reviewed Final Accuracy`
- 下一轮训练要把“输出收尾稳定性”和“减少长 think 依赖”纳入目标
