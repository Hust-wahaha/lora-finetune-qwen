"""
三指标 + max_tokens 扫描评测脚本。

目的：在同一份测试集（默认 `data/final/test_s800_think.jsonl`）上，对若干个 LoRA
微调模型（可选追加 baseline）做 `max_tokens` 列表扫描，每个组合算 3 个指标：

1. cot_completeness_rate       —— `<think>...</think>` 是否闭合
2. generation_completion_rate  —— 整段以「N。」（阿拉伯数字+句号）收尾，「答案：」前缀可有可无；
                                  亦即整条没被截。微调后实测主流输出 `…</think>\n\n78。`
                                  与训练目标格式 `…\n\n答案：78。` 都接受
3. answer_accuracy             —— 从整段末尾抽出阿拉伯数字并与 gold 字符串相等

模型在 CLI 用 `--model name:adapter_path` 传入，可重复。`--include-baseline` 自动追加
`baseline:none`（不加 LoRA）。早期 `name:adapter:expected_style` 三段式仍兼容，第三段
（modern/classical/none 等）被忽略。

对应课设主线诉求："设极小 max_tokens，观察 CoT 完整性与最终推理准确率"。

命令行示例
==========

完整扫描（两组 LoRA + baseline，5 个 max_tokens 档位）

AutoDL:
    uv run python scripts/eval_five_metrics.py \\
        --model modern_lora:runs/<run>/checkpoints/checkpoint-XX \\
        --model classical_lora:runs/<run>/checkpoints/checkpoint-YY \\
        --include-baseline \\
        --max-tokens-list 32,64,128,256,512 \\
        --run-tag window_sweep

Windows:
    .\\.venv\\Scripts\\python.exe scripts\\eval_five_metrics.py `
        --dataset-tag s800_think `
        --model "m2m:runs\\20260520_184554_train_s800_m2m_qwen3.5-0.8b_m2m_v1\\checkpoints\\checkpoint-40" `
        --model "c2m:runs\\20260520_192446_train_s800_c2m_qwen3.5-0.8b_c2m_v1\\checkpoints\\checkpoint-40" `
        --include-baseline `
        --max-tokens-list 16,32,64,128,256,512 `
        --max-batch-size 4 `
        --run-tag m2m_vs_c2m_mixed

本地 smoke（限 8 条样本，仅两个窗口）

    uv run python scripts/eval_five_metrics.py \\
        --model smoke:runs/<smoke_run>/checkpoints/checkpoint-16 \\
        --include-baseline \\
        --max-tokens-list 64,256 --limit 8 --run-tag smoke

只重新渲染 markdown（summary.json 已存在时，跳过推理与指标重算）

    uv run python scripts/eval_five_metrics.py \\
        --render-only --existing-run-dir runs/<eval_run>

    # Windows:
    .\\.venv\\Scripts\\python.exe scripts\\eval_five_metrics.py `
        --render-only --existing-run-dir runs\\20260521_120000_eval_s800_think_qwen3.5-0.8b_window_sweep

对已有 eval run 用新的 max_tokens_list 重新切分并更新指标（不重新推理）

    # 新列表里所有值必须 <= 原始推理窗口最大值；已有对应 prediction 文件时直接复用，
    # 没有时从最大窗口 prediction 按 token 边界截断生成。
    uv run python scripts/eval_five_metrics.py \\
        --reslice --existing-run-dir runs/<eval_run> \\
        --max-tokens-list 64,128,256,512,1024

    # Windows:
    .\\.venv\\Scripts\\python.exe scripts\\eval_five_metrics.py `
        --reslice `
        --existing-run-dir runs\\20260611_153614_eval_gsm1k_mixed_qwen3.5-0.8b_gsm1k_6models `
        --max-tokens-list 64,128,256,512,1024,2048

结果保存
========

run 目录由 `make_run_dir(stage='eval', ...)` 生成，路径形如
`runs/{timestamp}_eval_{dataset_tag}_qwen3.5-0.8b_{run_tag}/`，固定五个子目录。
本脚本只写其中两个：

- `predictions/{name}_mt{max_tokens}.json` —— 每个模型每个窗口一份原始模型输出
  （id/view/gold/response），N 模型 × M 窗口共 N×M 个文件
- `metrics/summary.json`                    —— 主结果，结构为
  `results[model_name][str(max_tokens)] = {3 个一级指标 + by_view + detailed[]}`
- `metrics/summary.md`                      —— 由 summary.json 渲染出的中文可读汇总
- `metrics/metrics_sweep.pdf` / `.png`      —— 2×3 网格图（行=题面风格，列=指标），
  每格 N 条线对比 N 个模型随 max_tokens 的变化；PDF 给论文/汇报用，PNG 快速预览

控制台输出
==========

只 print 一行 JSON headline 供日志抓取，**不打印指标对比表**（图直接落盘）：

    {"run_dir": "runs/...", "models": ["modern_lora", "baseline"],
     "max_tokens_list": [32, 64, 128, 256, 512],
     "summary_md": "runs/.../metrics/summary.md",
     "summary_figures": ["runs/.../metrics/metrics_sweep.pdf",
                         "runs/.../metrics/metrics_sweep.png"]}

要看指标值请读 `metrics/summary.json` 或 `metrics/summary.md`；要看趋势图直接打开
`metrics/metrics_sweep.pdf`。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.common.naming import DEFAULT_MODEL_TAG, dataset_file, make_run_dir
from src.common.paths import ROOT
from src.common.style_detect import extract_final_answer, is_cot_complete, is_generation_complete

MODEL_ID = 'Qwen/Qwen3.5-0.8B'
SYSTEM = '你是一个擅长中文数学应用题的助手，要求输出简洁、准确、可验证，并将最终答案输出在字符\'答案：\'后面'
BASELINE_SPEC = 'baseline:none'
_LEGACY_STYLE_SUFFIXES = {'modern', 'classical', 'none', 'unknown', 'mixed'}


def _system_from_run_config(adapter: str | None) -> str:
    """还原训练时 ms-swift 实际使用的 system prompt；找不到则返回默认值。

    checkpoint 路径约定：.../runs/<run>/checkpoints/checkpoint-XX
    run_config.json 在 checkpoints/ 的兄弟目录 metrics/ 里。

    注意：run_config.json 的 `system` 字段记录的是 `--system` CLI 默认值（传给
    get_template 的 default_system），但 ms-swift 在处理有显式 system 消息的记录时
    会优先用数据里 messages[0] 的内容，default_system 不会被触发。
    因此此函数优先从 train_dataset 的第一条记录读 messages[0]（训练时实际用的），
    若读不到则回退到 run_config.json 的 system 字段，最后回退到默认 SYSTEM。
    baseline（adapter=None）直接返回默认 SYSTEM。
    """
    if adapter is None:
        return SYSTEM
    run_config_path = Path(adapter).resolve().parent.parent / 'metrics' / 'run_config.json'
    if run_config_path.is_file():
        try:
            cfg = json.loads(run_config_path.read_text(encoding='utf-8'))
            train_dataset = cfg.get('train_dataset')
            if train_dataset:
                ds_path = Path(train_dataset)
                if not ds_path.is_absolute():
                    ds_path = ROOT / ds_path
                if ds_path.is_file():
                    first_line = ds_path.open(encoding='utf-8').readline().strip()
                    if first_line:
                        row = json.loads(first_line)
                        msgs = row.get('messages', [])
                        if msgs and msgs[0].get('role') == 'system':
                            return msgs[0]['content']
            return cfg.get('system', SYSTEM)
        except Exception:
            pass
    return SYSTEM


def enable_thinking_for(adapter: str | None) -> bool:
    """baseline（无 LoRA adapter）关 thinking，否则保持开启。

    背景：Qwen3.5 原始模型在 thinking 模式下对中文数学题非常容易陷入死循环
    （反复内省、自我怀疑、重新展开同一推理路径），消耗 max_tokens 却不收敛；
    关掉 thinking 后改用「答案：」风格的直接回答模板，反而更稳定。微调过的
    LoRA checkpoint 已经被训练成「在 <think> 块内紧凑产出」，保持 True。

    `parse_model_spec` 已把 'none'/'' 规约成 `None`，这里用 `bool(adapter)`
    双重保险：None 与空串都视为 baseline，落到 False。
    """
    return bool(adapter)


def parse_model_spec(raw: str) -> dict:
    # 兼容老的 `name:adapter:expected_style` 三段式：发现尾段是 legacy 风格枚举值就剥掉，
    # 但只有在剥完后仍剩至少一个 `:`（即原本就是三段、不是 `name:none` 这种二段）才剥，
    # 避免把 `baseline:none:none` 误缩成 `baseline:none` 又再次被当作"adapter='none'"。
    head, sep, tail = raw.rpartition(':')
    if sep and tail.lower() in _LEGACY_STYLE_SUFFIXES and ':' in head:
        raw = head
    name, sep, adapter = raw.partition(':')
    if not sep:
        raise ValueError(f'--model expects name:adapter (or legacy name:adapter:style), got {raw!r}')
    if not name:
        raise ValueError(f'empty model name in {raw!r}')
    adapter_value = None if adapter.lower() in {'', 'none'} else adapter
    return {'name': name, 'adapter': adapter_value}


def parse_max_tokens_list(raw: str) -> list[int]:
    values = [int(x.strip()) for x in raw.split(',') if x.strip()]
    if not values:
        raise ValueError('--max-tokens-list cannot be empty')
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--dataset-tag', type=str, default='s800_think')
    parser.add_argument('--test-file', type=Path, default=None)
    parser.add_argument(
        '--model', dest='models', action='append', default=[],
        help='Repeatable. Format: name:adapter_path_or_none (legacy 3-part still accepted)',
    )
    parser.add_argument('--include-baseline', action='store_true')
    parser.add_argument(
        '--max-tokens-list', type=str, default='32,64,128,256,512',
        help='Comma-separated ints, e.g. 32,64,128,256,512',
    )
    parser.add_argument('--max-batch-size', type=int, default=8)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top-p', type=float, default=1.0)
    parser.add_argument('--top-k', type=int, default=20)
    parser.add_argument('--repetition-penalty', type=float, default=1.0)
    parser.add_argument('--run-tag', type=str, default='five_metrics')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--existing-run-dir', type=Path, default=None)
    parser.add_argument(
        '--render-only', action='store_true',
        help='Skip inference + metric recompute, just regenerate summary.md from an existing summary.json. Requires --existing-run-dir.',
    )
    parser.add_argument(
        '--reslice', action='store_true',
        help=(
            'Re-slice an existing run with a new --max-tokens-list without re-running inference. '
            'Requires --existing-run-dir. New values must be <= the largest prediction window '
            'already on disk. Existing per-mt prediction files are reused; missing ones are '
            'generated by token-level truncation from the largest available file. '
            'Updates summary.json, summary.md, and figures in-place.'
        ),
    )
    return parser.parse_args()


def build_engine(
    adapter: str | None,
    max_batch_size: int,
    system: str = SYSTEM,
    enable_thinking: bool = True,
):
    from peft import PeftModel
    from swift import get_model_processor, get_template
    from swift.infer_engine import TransformersEngine

    model, tokenizer = get_model_processor(MODEL_ID)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    template_type = model.model_meta.template
    template = get_template(
        tokenizer,
        template_type=template_type,
        default_system=system,
        enable_thinking=enable_thinking,
    )
    return TransformersEngine(model, template=template, max_batch_size=max_batch_size), tokenizer


def truncate_by_tokens(text: str, max_tokens: int, tokenizer) -> str:
    """把 text 截断到 max_tokens 个 token，在 token 边界上切，再 decode 回字符串。

    `<think>` / `</think>` 等特殊 token 用 skip_special_tokens=False 保留，
    确保截断后的字符串与模型真实用 max_tokens=N 推理的输出等价（greedily）。
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=False)


def load_rows(test_file: Path, limit: int | None) -> list[dict]:
    rows = [json.loads(line) for line in test_file.read_text(encoding='utf-8').splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def run_inference(
    name: str,
    rows: list[dict],
    max_tokens: int,
    args: argparse.Namespace,
    run_dir: Path,
    engine,
) -> list[dict]:
    from swift.infer_engine import InferRequest, RequestConfig
    requests = [InferRequest(messages=[{'role': 'user', 'content': row['messages'][1]['content']}]) for row in rows]
    request_config = RequestConfig(
        max_tokens=max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )
    responses = engine.infer(requests, request_config)
    output_rows = []
    for row, response in zip(rows, responses):
        output_rows.append({
            'id': row['id'],
            'base_id': row['base_id'],
            'family': row['family'],
            'view': row['view'],
            'query': row['messages'][1]['content'],
            'gold_answer': row['answer'],
            'response': response.choices[0].message.content,
        })
    output_path = run_dir / 'predictions' / f'{name}_mt{max_tokens}.json'
    output_path.write_text(json.dumps(output_rows, ensure_ascii=False, indent=2), encoding='utf-8')
    return output_rows


def compute_metrics(rows: list[dict]) -> dict:
    n = len(rows) or 1
    detailed = []
    counters = {'cot_complete': 0, 'generation_complete': 0, 'answer_correct': 0}
    by_view: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        resp = row['response']
        cot_complete = is_cot_complete(resp)
        gen_complete = is_generation_complete(resp)
        pred_answer = extract_final_answer(resp)
        answer_correct = pred_answer == row['gold_answer']

        counters['cot_complete'] += int(cot_complete)
        counters['generation_complete'] += int(gen_complete)
        counters['answer_correct'] += int(answer_correct)

        view = row['view']
        view_bucket = by_view[view]
        view_bucket['total'] += 1
        view_bucket['cot_complete'] += int(cot_complete)
        view_bucket['generation_complete'] += int(gen_complete)
        view_bucket['answer_correct'] += int(answer_correct)

        detailed.append({
            'id': row['id'],
            'view': row['view'],
            'family': row['family'],
            'gold_answer': row['gold_answer'],
            'pred_answer': pred_answer,
            'answer_correct': answer_correct,
            'cot_complete': cot_complete,
            'generation_complete': gen_complete,
            'response_chars': len(resp),
        })

    by_view_out = {}
    for view, bucket in by_view.items():
        v_total = bucket['total'] or 1
        by_view_out[view] = {
            'total': bucket['total'],
            'cot_completeness_rate': bucket['cot_complete'] / v_total,
            'generation_completion_rate': bucket['generation_complete'] / v_total,
            'answer_accuracy': bucket['answer_correct'] / v_total,
        }

    return {
        'count': len(rows),
        'cot_completeness_rate': counters['cot_complete'] / n,
        'generation_completion_rate': counters['generation_complete'] / n,
        'answer_accuracy': counters['answer_correct'] / n,
        'by_view': by_view_out,
        'detailed': detailed,
    }


def collect_models(args: argparse.Namespace) -> list[dict]:
    specs = list(args.models)
    if args.include_baseline and not any(s.startswith('baseline:') for s in specs):
        specs.append(BASELINE_SPEC)
    if not specs:
        raise ValueError('At least one --model is required (or pass --include-baseline)')
    return [parse_model_spec(s) for s in specs]


def load_existing(run_dir: Path, model_name: str, max_tokens: int) -> list[dict]:
    path = run_dir / 'predictions' / f'{model_name}_mt{max_tokens}.json'
    return json.loads(path.read_text(encoding='utf-8'))


def _fmt_pct(value) -> str:
    if value is None:
        return '-'
    return f'{value * 100:.2f}%'


def render_markdown_summary(summary: dict) -> str:
    results = summary.get('results', {})
    model_specs = summary.get('models', [])
    model_names = [m['name'] for m in model_specs] or list(results.keys())
    max_tokens_list = summary.get('decode_max_tokens_list', [])
    views = ['modern', 'classical']
    view_zh = {'modern': '白话题面', 'classical': '文言题面'}

    lines: list[str] = []
    lines.append(f'# 评测汇总 - {Path(summary.get("run_dir", "")).name}')
    lines.append('')

    # 基本信息
    lines.append('## 基本信息')
    lines.append('')
    lines.append('| 项 | 值 |')
    lines.append('|---|---|')
    lines.append(f'| 数据集标签 | `{summary.get("dataset_tag")}` |')
    lines.append(f'| 测试集文件 | `{summary.get("test_file")}` |')
    # 顶层 count 已经去掉，从 results 任取一项的 per-result count（同一份测试集
    # 各模型各 max_tokens 都是同一规模，挑哪个都行；取不到则 '-' 占位）。
    _any_model_results = next(iter(results.values()), {})
    _first_result = next(iter(_any_model_results.values()), {})
    lines.append(f'| 测试样本数 | {_first_result.get("count", "-")} |')
    lines.append(f'| 生成长度上限（tokens） | {", ".join(str(x) for x in max_tokens_list)} |')
    lines.append('')

    # 指标说明
    lines.append('## 三个评测指标说明')
    lines.append('')
    lines.append('| 指标 | 衡量的是什么 |')
    lines.append('|---|---|')
    lines.append('| 答案准确率 | 从输出末尾抽取出的阿拉伯数字是否等于标准答案。优先识别「答案：N」（句号可省），其次识别裸 `N。`（句号必备） |')
    lines.append('| 推理过程完整率 | 输出里 `<think>…</think>` 标签是否成对出现（推理段没被截断） |')
    lines.append('| 整体回答完整率 | 在前者基础上，整段以「答案：N」或裸「N。」收尾，即整段没被截 |')
    lines.append('')
    lines.append('> **关于 baseline 的特别说明**：baseline 评测时关闭了 thinking 模式（避免原始 Qwen3.5')
    lines.append('> 在中文数学题上陷入死循环），其输出始终是 `<think>\\n\\n</think>\\n\\n…答案：N` 结构，')
    lines.append('> 即 `<think>` 块为空但 tag 已对——因此 baseline 的「**推理过程完整率**」恒为 100%，')
    lines.append('> 只反映 tag 是否成对，不反映其推理质量；要看 baseline 的真实能力请直接看「答案准确率」')
    lines.append('> 与「整体回答完整率」。')
    lines.append('')

    # 整体表现
    lines.append('## 整体表现（白话题与文言题合并统计）')
    lines.append('')
    lines.append('> 在同一份测试集上，每个模型在每档生成长度上限下的成绩。')
    lines.append('> 表头中的"生成长度上限"指模型解码时允许的最大 token 数，截断会导致推理过程不完整。')
    lines.append('')
    lines.append('| 生成长度上限 | 模型 | 答案准确率 | 推理过程完整率 | 整体回答完整率 |')
    lines.append('|---:|---|---:|---:|---:|')
    for mt in max_tokens_list:
        for name in model_names:
            metric = (results.get(name) or {}).get(str(mt))
            if not metric:
                continue
            lines.append(
                f'| {mt} | `{name}` | '
                f'{_fmt_pct(metric.get("answer_accuracy"))} | '
                f'{_fmt_pct(metric.get("cot_completeness_rate"))} | '
                f'{_fmt_pct(metric.get("generation_completion_rate"))} |'
            )
    lines.append('')

    # 分语言风格细看
    lines.append('## 按输入题面语言风格分别看')
    lines.append('')
    lines.append('> 把测试集按输入题面是白话还是文言拆成两组分别统计，可以看出模型对哪种输入风格更擅长。')
    lines.append('> 若一个模型在白话题面和文言题面上的表现接近，说明它对两种语言风格的泛化能力较强。')
    lines.append('')
    view_metric_specs = [
        ('答案准确率', 'answer_accuracy'),
        ('推理过程完整率', 'cot_completeness_rate'),
        ('整体回答完整率', 'generation_completion_rate'),
    ]
    for title, key in view_metric_specs:
        lines.append(f'### {title}')
        lines.append('')
        lines.append(f'| 生成长度上限 | 模型 | {view_zh["modern"]} | {view_zh["classical"]} |')
        lines.append('|---:|---|---:|---:|')
        for mt in max_tokens_list:
            for name in model_names:
                metric = (results.get(name) or {}).get(str(mt))
                if not metric:
                    continue
                by_view = metric.get('by_view', {}) or {}
                cells = [_fmt_pct((by_view.get(v) or {}).get(key)) for v in views]
                lines.append(f'| {mt} | `{name}` | {cells[0]} | {cells[1]} |')
        lines.append('')

    lines.append('---')
    lines.append('')
    lines.append('注释：')
    lines.append('- `baseline` 表示未做任何 LoRA 微调的原始模型，作为对照参照。')
    lines.append('- 百分号数字越高越好。')
    lines.append('- 表中"-"代表该项不适用或该 (模型, 生成长度上限) 组合无数据。')
    lines.append('')

    return '\n'.join(lines)


def write_markdown_summary(run_dir: Path, summary: dict) -> Path:
    md_path = run_dir / 'metrics' / 'summary.md'
    md_path.write_text(render_markdown_summary(summary), encoding='utf-8')
    return md_path


# Wong 8 色色盲安全色板（去掉极浅的 #F0E442 黄色），最多支持 7 条模型曲线。
# 参考 Wong, B. "Points of view: Color blindness." Nature Methods 8, 441 (2011).
_FIGURE_PALETTE: tuple[str, ...] = (
    '#000000',  # black
    '#E69F00',  # orange
    '#56B4E9',  # sky blue
    '#009E73',  # bluish green
    '#0072B2',  # deep blue
    '#D55E00',  # vermillion
    '#CC79A7',  # reddish purple
)

_FIGURE_METRIC_SPECS = (
    ('answer_accuracy', '答案准确率\nAnswer accuracy (%)'),
    ('cot_completeness_rate', '推理过程完整率\nCoT completeness (%)'),
    ('generation_completion_rate', '整体回答完整率\nGeneration completion (%)'),
)

_FIGURE_VIEW_ZH = {
    'modern': '白话题面 (modern view)',
    'classical': '文言题面 (classical view)',
}


def _select_xaxis_scale(values: list[int]) -> str:
    # max_tokens 列表可能是线性 (4,8,...,44) 也可能是几何 (32,64,128,256,512)。
    # 跨度 >10 倍就切对数 (base 2)，让两端数据点都可读。
    if len(values) < 2:
        return 'linear'
    return 'log' if (max(values) / max(min(values), 1)) > 10 else 'linear'


def render_summary_figures(summary: dict, run_dir: Path) -> list[Path]:
    """Nature-leaning 2×3 折线网格：行=题面风格，列=指标，线=模型。

    输出 `metrics/metrics_sweep.{pdf,png}`，PDF 矢量给汇报/论文用，PNG 600 DPI
    给 IDE/markdown 预览用。模型数 ≤7 时色板足够区分；超过 7 个会循环用色。
    """
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    results = summary.get('results') or {}
    model_specs = summary.get('models') or []
    model_names = [m['name'] for m in model_specs] or list(results.keys())
    max_tokens_list = summary.get('decode_max_tokens_list') or []
    views = ['modern', 'classical']

    if not model_names or not max_tokens_list:
        return []

    # rcParams 一次性套上：CJK 字体回退链 + 论文级排版。
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'Arial', 'DejaVu Sans'],
        'axes.unicode_minus': False,
        'font.size': 8,
        'axes.titlesize': 9,
        'axes.labelsize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.7,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'lines.linewidth': 1.4,
        'lines.markersize': 4.5,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.08,
    })

    x_scale = _select_xaxis_scale(max_tokens_list)
    # x 轴稀疏化：>=10 个点时每隔一个标注，避免最右端文字重叠。
    if len(max_tokens_list) >= 10:
        x_ticks = max_tokens_list[::2]
        if max_tokens_list[-1] not in x_ticks:
            x_ticks = x_ticks + [max_tokens_list[-1]]
    else:
        x_ticks = list(max_tokens_list)

    n_rows, n_cols = len(views), len(_FIGURE_METRIC_SPECS)
    # 留出顶部 0.92 留给 suptitle + legend；不要 constrained_layout 强行抢空间。
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(7.8, 5.0),
        sharex='col',
        sharey='row',
    )
    if n_rows == 1:
        axes = [axes]

    handles_for_legend: dict[str, plt.Line2D] = {}

    for r, view in enumerate(views):
        for c, (key, label) in enumerate(_FIGURE_METRIC_SPECS):
            ax = axes[r][c]
            for i, name in enumerate(model_names):
                xs, ys = [], []
                for mt in max_tokens_list:
                    metric = (results.get(name) or {}).get(str(mt))
                    if not metric:
                        continue
                    val = (metric.get('by_view', {}).get(view) or {}).get(key)
                    if val is None:
                        continue
                    xs.append(mt)
                    ys.append(val * 100.0)
                if not xs:
                    continue
                color = _FIGURE_PALETTE[i % len(_FIGURE_PALETTE)]
                line, = ax.plot(
                    xs, ys,
                    color=color, marker='o', markeredgewidth=0,
                    label=name,
                )
                handles_for_legend.setdefault(name, line)
            ax.set_ylim(-4, 104)
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.grid(True, axis='y', linewidth=0.4, alpha=0.35)
            ax.set_xscale(x_scale)
            ax.set_xticks(x_ticks)
            if x_scale == 'log':
                ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
                ax.minorticks_off()
            if r == 0:
                ax.set_title(label, pad=6)
            if r == n_rows - 1:
                ax.set_xlabel('生成长度上限 Max tokens')
            if c == 0:
                ax.set_ylabel(_FIGURE_VIEW_ZH[view])

    # 单图例放在 axes 上方、suptitle 下方；按 CLI 声明顺序保留模型顺序。
    ordered_handles = [handles_for_legend[n] for n in model_names if n in handles_for_legend]
    ordered_labels = [n for n in model_names if n in handles_for_legend]
    if ordered_handles:
        fig.legend(
            ordered_handles, ordered_labels,
            loc='upper center',
            bbox_to_anchor=(0.5, 0.945),
            ncol=min(7, len(ordered_handles)),
            frameon=False,
            columnspacing=1.6,
            handlelength=1.8,
        )

    fig.suptitle(
        f'指标随 max_tokens 的变化（数据集 {summary.get("dataset_tag", "?")}）',
        fontsize=10, y=0.99,
    )
    # baseline 关 thinking 后输出 `<think>\n\n</think>\n\n…`，CoT 完整率恒为 100%
    # 但不反映推理质量；图例里有 baseline 时在底部加一行小字脚注，避免读者误读。
    has_baseline = any(n.lower() == 'baseline' for n in ordered_labels)
    if has_baseline:
        fig.text(
            0.5, 0.02,
            '注：baseline 关闭 thinking，`<think>` 块为空但 tag 已对，'
            '其「推理过程完整率」恒为 100%，仅反映标签存在性，不反映推理质量。',
            ha='center', va='bottom', fontsize=6.5, style='italic', color='#444444',
        )
        bottom_margin = 0.13
    else:
        bottom_margin = 0.10

    # 顶部空出 legend (~y=0.945) + suptitle (~y=0.99) + 列标题；子图顶到 0.80。
    fig.subplots_adjust(top=0.80, bottom=bottom_margin, left=0.09, right=0.985,
                        wspace=0.12, hspace=0.22)

    out_dir = run_dir / 'metrics'
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / 'metrics_sweep.pdf'
    png_path = out_dir / 'metrics_sweep.png'
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=600)
    plt.close(fig)
    return [pdf_path, png_path]


def main() -> None:
    args = parse_args()
    root = args.root.resolve()

    if args.render_only:
        if not args.existing_run_dir:
            raise ValueError('--render-only requires --existing-run-dir')
        run_dir = args.existing_run_dir.resolve()
        summary_path = run_dir / 'metrics' / 'summary.json'
        if not summary_path.is_file():
            raise FileNotFoundError(
                f'--render-only 需要已有的评测产物，但找不到 {summary_path}。'
                f'确认 --existing-run-dir={run_dir} 路径正确，且该 run 已经跑过完整评测'
                f'（先去掉 --render-only 跑一次生成 metrics/summary.json，再用 --render-only 重渲染）。'
            )
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        md_path = write_markdown_summary(run_dir, summary)
        fig_paths = render_summary_figures(summary, run_dir)
        print(json.dumps({
            'rendered_markdown': str(md_path),
            'summary_figures': [str(p) for p in fig_paths],
        }, ensure_ascii=False))
        return

    if args.reslice:
        if not args.existing_run_dir:
            raise ValueError('--reslice requires --existing-run-dir')
        run_dir = args.existing_run_dir.resolve()
        summary_path = run_dir / 'metrics' / 'summary.json'
        if not summary_path.is_file():
            raise FileNotFoundError(f'找不到 {summary_path}，请先完整跑一次评测再用 --reslice')
        existing_summary = json.loads(summary_path.read_text(encoding='utf-8'))
        model_specs = existing_summary.get('models', [])
        if not model_specs:
            raise ValueError('existing summary.json 里没有 models 字段')
        orig_mt_list = existing_summary.get('decode_max_tokens_list', [])
        new_max_tokens_list = parse_max_tokens_list(args.max_tokens_list)

        # 找每个模型实际存在的最大 prediction 文件，作为截断来源
        best_mt_per_model: dict[str, int] = {}
        for spec in model_specs:
            name = spec['name']
            available = [
                mt for mt in orig_mt_list
                if (run_dir / 'predictions' / f'{name}_mt{mt}.json').is_file()
            ]
            if not available:
                raise FileNotFoundError(
                    f'模型 {name!r} 在 {run_dir}/predictions/ 下找不到任何 prediction 文件'
                )
            best_mt_per_model[name] = max(available)

        # 校验：新列表里的值不能超过现有最大窗口
        for spec in model_specs:
            name = spec['name']
            best_mt = best_mt_per_model[name]
            too_large = [mt for mt in new_max_tokens_list if mt > best_mt]
            if too_large:
                raise ValueError(
                    f'模型 {name!r}：新 max_tokens 值 {too_large} 超过现有最大推理窗口 '
                    f'({best_mt})，无法从现有 prediction 截断生成。请重新跑推理。'
                )

        # 只加载 tokenizer，不加载模型权重，不需要 GPU
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

        results: dict[str, dict] = {}
        for spec in model_specs:
            name = spec['name']
            results[name] = {}
            best_mt = best_mt_per_model[name]
            full_rows = json.loads(
                (run_dir / 'predictions' / f'{name}_mt{best_mt}.json').read_text(encoding='utf-8')
            )
            for mt in new_max_tokens_list:
                pred_path = run_dir / 'predictions' / f'{name}_mt{mt}.json'
                if pred_path.is_file():
                    pred_rows = json.loads(pred_path.read_text(encoding='utf-8'))
                else:
                    pred_rows = [
                        {**row, 'response': truncate_by_tokens(row['response'], mt, tokenizer)}
                        for row in full_rows
                    ]
                    pred_path.write_text(json.dumps(pred_rows, ensure_ascii=False, indent=2), encoding='utf-8')
                results[name][str(mt)] = compute_metrics(pred_rows)

        tokenizer = None

        # 删除原列表里有、新列表里没有的 prediction 文件，避免旧窗口文件残留
        new_mt_set = set(new_max_tokens_list)
        for spec in model_specs:
            for mt in orig_mt_list:
                if mt not in new_mt_set:
                    stale = run_dir / 'predictions' / f'{spec["name"]}_mt{mt}.json'
                    if stale.is_file():
                        stale.unlink()

        new_summary = {**existing_summary, 'decode_max_tokens_list': new_max_tokens_list, 'results': results}
        summary_path.write_text(json.dumps(new_summary, ensure_ascii=False, indent=2), encoding='utf-8')
        md_path = write_markdown_summary(run_dir, new_summary)
        fig_paths = render_summary_figures(new_summary, run_dir)
        print(json.dumps({
            'run_dir': str(run_dir),
            'new_max_tokens_list': new_max_tokens_list,
            'summary_md': str(md_path),
            'summary_figures': [str(p) for p in fig_paths],
        }, ensure_ascii=False))
        return

    max_tokens_list = parse_max_tokens_list(args.max_tokens_list)
    model_specs = collect_models(args)

    if args.existing_run_dir:
        run_dir = args.existing_run_dir.resolve()
        test_file = args.test_file
    else:
        test_file = (args.test_file or dataset_file('test', args.dataset_tag)).resolve()
        rows = load_rows(test_file, args.limit)
        run_dir = make_run_dir(
            stage='eval',
            dataset_tag=args.dataset_tag,
            model_tag=DEFAULT_MODEL_TAG,
            suffix=args.run_tag,
            base_dir=root / 'runs',
        )

    (run_dir / 'metrics').mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}

    for spec in model_specs:
        name = spec['name']
        adapter = spec['adapter']
        results[name] = {}

        if args.existing_run_dir:
            for mt in max_tokens_list:
                pred_rows = load_existing(run_dir, name, mt)
                results[name][str(mt)] = compute_metrics(pred_rows)
        else:
            # 每个模型只推理一次（max_tokens 取列表最大值），再按 token 边界截断
            # 重算较小窗口的指标。greedily 解码下前 N token 与 max_tokens=N 单独推理等价。
            engine, tokenizer = build_engine(
                adapter, args.max_batch_size, _system_from_run_config(adapter),
                enable_thinking=enable_thinking_for(adapter),
            )
            max_mt = max(max_tokens_list)
            full_rows = run_inference(name, rows, max_mt, args, run_dir, engine)
            engine = None  # 推理完立即释放 GPU，后续截断在 CPU 做

            for mt in max_tokens_list:
                if mt == max_mt:
                    pred_rows = full_rows  # 已由 run_inference 落盘，无需重写
                else:
                    pred_rows = [
                        {**row, 'response': truncate_by_tokens(row['response'], mt, tokenizer)}
                        for row in full_rows
                    ]
                    out_path = run_dir / 'predictions' / f'{name}_mt{mt}.json'
                    out_path.write_text(json.dumps(pred_rows, ensure_ascii=False, indent=2), encoding='utf-8')
                results[name][str(mt)] = compute_metrics(pred_rows)

            tokenizer = None  # 释放词表内存

    # 顶层不再放单一的 `count` —— 各 (model, max_tokens) 组合的样本数已经在
    # `results[name][str(mt)]['count']` 里，per-result 才不会误导。
    summary = {
        'run_dir': str(run_dir),
        'dataset_tag': args.dataset_tag,
        'test_file': str(test_file) if test_file else None,
        'decode_max_tokens_list': max_tokens_list,
        'models': model_specs,
        'results': results,
    }
    summary_path = run_dir / 'metrics' / 'summary.json'
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    md_path = write_markdown_summary(run_dir, summary)
    fig_paths = render_summary_figures(summary, run_dir)

    headline = {
        'run_dir': summary['run_dir'],
        'models': [m['name'] for m in model_specs],
        'max_tokens_list': max_tokens_list,
        'summary_md': str(md_path),
        'summary_figures': [str(p) for p in fig_paths],
    }
    print(json.dumps(headline, ensure_ascii=False))


if __name__ == '__main__':
    if '--help' in sys.argv or '-h' in sys.argv:
        parse_args()
        raise SystemExit(0)
    main()
