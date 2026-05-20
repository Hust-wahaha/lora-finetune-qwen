"""
五指标 + max_tokens 扫描评测脚本。

目的：在同一份测试集（默认 `data/final/test_s800_think.jsonl`）上，对若干个 LoRA
微调模型（可选追加 baseline）做 `max_tokens` 列表扫描，每个组合算 5 个指标：

1. cot_completeness_rate       —— `<think>...</think>` 是否闭合
2. generation_completion_rate  —— 进一步要求出现「答案：」且以「。」结尾，亦即整条没被截
3. answer_accuracy             —— 规则抽数值并与 gold 字符串相等（与 eval_compare_full 同口径）
4. cot_quality_score           —— 可选；DeepSeek V4 Flash 对 CoT 内容 1-5 打分
5. style_faithfulness          —— modern/classical 命中率 + 期望风格命中率 + 英文漏出率

模型在 CLI 用 `--model name:adapter_path:expected_style` 三元组传入，可重复。
`--include-baseline` 自动追加 `baseline:none:none`（不加 LoRA）。

对应课设主线诉求："设极小 max_tokens，观察 CoT 完整性与最终推理准确率"。

命令行示例
==========

完整扫描（两组 LoRA + baseline，5 个 max_tokens 档位）：

    uv run python scripts/eval_five_metrics.py \\
        --model modern_lora:runs/<run>/checkpoints/checkpoint-XX:modern \\
        --model classical_lora:runs/<run>/checkpoints/checkpoint-YY:classical \\
        --include-baseline \\
        --max-tokens-list 32,64,128,256,512 \\
        --run-tag window_sweep

本地 smoke（限 8 条样本，仅两个窗口，跳过 LLM 复核）：

    uv run python scripts/eval_five_metrics.py \\
        --model smoke:runs/<smoke_run>/checkpoints/checkpoint-16:modern \\
        --include-baseline \\
        --max-tokens-list 64,256 --limit 8 --run-tag smoke

启用 DeepSeek CoT 质量评分（需环境变量 DEEPSEEK_API_KEY，可用 --existing-run-dir
复用已有推理结果，避免重跑）：

    DEEPSEEK_API_KEY=xxx uv run python scripts/eval_five_metrics.py \\
        --existing-run-dir runs/<eval_run> \\
        --model modern_lora:none:modern \\
        --llm-review-mode cot_quality \\
        --llm-review-max-cases 80

结果保存
========

run 目录由 `make_run_dir(stage='eval', ...)` 生成，路径形如
`runs/{timestamp}_eval_{dataset_tag}_qwen3.5-0.8b_{run_tag}/`，固定五个子目录。
本脚本只写其中两个：

- `predictions/{name}_mt{max_tokens}.json` —— 每个模型每个窗口一份原始模型输出
  （id/view/gold/response），N 模型 × M 窗口共 N×M 个文件
- `metrics/summary.json`                    —— 主结果，结构为
  `results[model_name][str(max_tokens)] = {5 个一级指标 + by_view + detailed[]}`
- `metrics/{name}_mt{max_tokens}_cot_quality.json` —— 仅启用 LLM 复核时生成，含
  每条样本的 1-5 分与简短理由

控制台输出
==========

只 print 一行 JSON headline 供日志抓取，**不打印指标对比表，也不出图**：

    {"run_dir": "runs/...", "models": ["modern_lora", "baseline"],
     "max_tokens_list": [32, 64, 128, 256, 512]}

要看指标值或画图请读 `metrics/summary.json`，后续可用独立脚本渲染。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.common.naming import DEFAULT_MODEL_TAG, dataset_file, make_run_dir
from src.common.paths import ROOT
from src.common.style_detect import (
    english_leak_count,
    extract_think_block,
    is_cot_complete,
    is_generation_complete,
    style_label,
    style_scores,
)

MODEL_ID = 'Qwen/Qwen3.5-0.8B'
SYSTEM = '你是一个擅长中文数学应用题的助手，请尽量给出简洁且正确的答案。'
DEFAULT_DEEPSEEK_ENDPOINT = 'https://api.deepseek.com/chat/completions'
BASELINE_SPEC = 'baseline:none:none'


def parse_model_spec(raw: str) -> dict:
    parts = raw.split(':')
    if len(parts) != 3:
        raise ValueError(f'--model expects name:adapter:expected_style, got {raw!r}')
    name, adapter, expected = parts
    if not name:
        raise ValueError(f'empty model name in {raw!r}')
    adapter_value = None if adapter.lower() in {'', 'none'} else adapter
    expected_value = expected.lower()
    if expected_value not in {'modern', 'classical', 'none', 'unknown', 'mixed'}:
        raise ValueError(f'expected_style must be modern/classical/none, got {expected!r}')
    return {'name': name, 'adapter': adapter_value, 'expected_style': expected_value}


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
        help='Repeatable. Format: name:adapter_path_or_none:expected_style',
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
    parser.add_argument('--llm-review-mode', choices=['none', 'cot_quality', 'all'], default='none')
    parser.add_argument('--llm-review-model', type=str, default='deepseek-v4-flash')
    parser.add_argument('--llm-review-api-key-env', type=str, default='DEEPSEEK_API_KEY')
    parser.add_argument('--llm-review-endpoint', type=str, default=DEFAULT_DEEPSEEK_ENDPOINT)
    parser.add_argument('--llm-review-max-cases', type=int, default=None)
    parser.add_argument('--llm-review-timeout', type=int, default=120)
    return parser.parse_args()


def extract_answer(text: str) -> str | None:
    patterns = [
        r'答案[：: ]*([0-9]+(?:\.[0-9]+)?)',
        r'故[^。\n=]*=\s*([0-9]+(?:\.[0-9]+)?)',
        r'=\s*([0-9]+(?:\.[0-9]+)?)。?\s*$',
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    numbers = re.findall(r'(?<![A-Za-z])([0-9]+(?:\.[0-9]+)?)(?![A-Za-z])', text)
    return numbers[-1] if numbers else None


def build_engine(adapter: str | None, max_batch_size: int):
    from peft import PeftModel
    from swift import get_model_processor, get_template
    from swift.infer_engine import TransformersEngine

    model, tokenizer = get_model_processor(MODEL_ID)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    template_type = model.model_meta.template
    template = get_template(tokenizer, template_type=template_type, default_system=SYSTEM)
    return TransformersEngine(model, template=template, max_batch_size=max_batch_size)


def load_rows(test_file: Path, limit: int | None) -> list[dict]:
    rows = [json.loads(line) for line in test_file.read_text(encoding='utf-8').splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def run_inference(
    name: str,
    adapter: str | None,
    rows: list[dict],
    max_tokens: int,
    args: argparse.Namespace,
    run_dir: Path,
    engine=None,
) -> list[dict]:
    from swift.infer_engine import InferRequest, RequestConfig
    if engine is None:
        engine = build_engine(adapter, args.max_batch_size)
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


def compute_metrics(rows: list[dict], expected_style: str) -> dict:
    n = len(rows) or 1
    detailed = []
    counters = {
        'cot_complete': 0,
        'generation_complete': 0,
        'answer_correct': 0,
        'modern_hit': 0,
        'classical_hit': 0,
        'style_match_expected': 0,
        'english_leak': 0,
    }
    by_view: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        resp = row['response']
        cot = extract_think_block(resp)
        cot_complete = is_cot_complete(resp)
        gen_complete = is_generation_complete(resp)
        pred_answer = extract_answer(resp)
        answer_correct = pred_answer == row['gold_answer']
        scores = style_scores(cot)
        label = style_label(cot)
        eng = english_leak_count(cot)

        counters['cot_complete'] += int(cot_complete)
        counters['generation_complete'] += int(gen_complete)
        counters['answer_correct'] += int(answer_correct)
        counters['modern_hit'] += int(scores['modern_hits'] > 0)
        counters['classical_hit'] += int(scores['classical_hits'] > 0)
        if expected_style in {'modern', 'classical'}:
            counters['style_match_expected'] += int(label == expected_style)
        counters['english_leak'] += int(eng > 0)

        view = row['view']
        view_bucket = by_view[view]
        view_bucket['total'] += 1
        view_bucket['cot_complete'] += int(cot_complete)
        view_bucket['generation_complete'] += int(gen_complete)
        view_bucket['answer_correct'] += int(answer_correct)
        view_bucket['style_match_expected'] += int(
            expected_style in {'modern', 'classical'} and label == expected_style
        )
        view_bucket['english_leak'] += int(eng > 0)

        detailed.append({
            'id': row['id'],
            'view': row['view'],
            'family': row['family'],
            'gold_answer': row['gold_answer'],
            'pred_answer': pred_answer,
            'answer_correct': answer_correct,
            'cot_complete': cot_complete,
            'generation_complete': gen_complete,
            'cot_chars': len(cot) if cot else 0,
            'response_chars': len(resp),
            'style_label': label,
            'modern_hits': scores['modern_hits'],
            'classical_hits': scores['classical_hits'],
            'english_leak_runs': eng,
            'cot_quality_score': None,
            'cot_quality_reason': None,
        })

    def _rate(key: str) -> float:
        return counters[key] / n

    by_view_out = {}
    for view, bucket in by_view.items():
        v_total = bucket['total'] or 1
        by_view_out[view] = {
            'total': bucket['total'],
            'cot_completeness_rate': bucket['cot_complete'] / v_total,
            'generation_completion_rate': bucket['generation_complete'] / v_total,
            'answer_accuracy': bucket['answer_correct'] / v_total,
            'style_faithfulness_rate': (
                bucket['style_match_expected'] / v_total
                if expected_style in {'modern', 'classical'} else None
            ),
            'english_leak_rate': bucket['english_leak'] / v_total,
        }

    return {
        'count': len(rows),
        'expected_style': expected_style,
        'cot_completeness_rate': _rate('cot_complete'),
        'generation_completion_rate': _rate('generation_complete'),
        'answer_accuracy': _rate('answer_correct'),
        'cot_quality_score': {'enabled': False},
        'style_faithfulness': {
            'modern_hit_rate': _rate('modern_hit'),
            'classical_hit_rate': _rate('classical_hit'),
            'faithfulness_rate': (
                _rate('style_match_expected') if expected_style in {'modern', 'classical'} else None
            ),
            'english_leak_rate': _rate('english_leak'),
        },
        'by_view': by_view_out,
        'detailed': detailed,
    }


def extract_first_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith('{') and text.endswith('}'):
        return json.loads(text)
    match = re.search(r'\{.*\}', text, re.S)
    if not match:
        raise ValueError(f'No JSON object found in review response: {text[:200]}')
    return json.loads(match.group(0))


def cot_quality_prompt(row: dict, cot: str | None) -> str:
    cot_text = cot if cot else '(empty)'
    return (
        '请只对下面这段 <think> 推理本身打 1-5 分。'
        '5=步骤完整、计算正确、无跳跃；4=略有冗余但结论正确；3=步骤不全或有小错；'
        '2=明显错误；1=空白或完全错误。'
        '只返回 JSON：{"score":1-5,"reason":"一句简短中文"}。\n\n'
        f'question: {row["query"]}\n'
        f'gold_answer: {row["gold_answer"]}\n'
        f'cot: {cot_text}\n'
    )


def deepseek_score(row: dict, cot: str | None, args: argparse.Namespace) -> dict:
    api_key = os.environ.get(args.llm_review_api_key_env)
    if not api_key:
        raise RuntimeError(f'Missing API key env: {args.llm_review_api_key_env}')
    payload = {
        'model': args.llm_review_model,
        'messages': [
            {'role': 'system', 'content': '你是严格的中文数学 CoT 评分员。不要解释，不要输出额外文本，只输出 JSON。'},
            {'role': 'user', 'content': cot_quality_prompt(row, cot)},
        ],
        'temperature': 0.0,
        'max_tokens': 256,
        'response_format': {'type': 'json_object'},
        'thinking': {'type': 'disabled'},
    }
    request = urllib.request.Request(
        args.llm_review_endpoint,
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'},
        method='POST',
    )
    try:
        with urllib.request.urlopen(request, timeout=args.llm_review_timeout) as response:
            body = json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode('utf-8', errors='replace')
        raise RuntimeError(f'DeepSeek score HTTP error {exc.code}: {detail}') from exc
    content = body['choices'][0]['message']['content']
    parsed = extract_first_json_object(content)
    score = int(parsed.get('score', 0))
    if score < 1 or score > 5:
        raise ValueError(f'Unexpected DeepSeek score: {parsed}')
    return {
        'score': score,
        'reason': str(parsed.get('reason', '')).strip(),
        'raw_response': content,
        'usage': body.get('usage', {}),
    }


def apply_cot_quality(
    metrics: dict,
    rows: list[dict],
    run_dir: Path,
    model_name: str,
    max_tokens: int,
    args: argparse.Namespace,
) -> None:
    if args.llm_review_mode == 'none':
        return
    detail_by_id = {d['id']: d for d in metrics['detailed']}
    review_rows = []
    reviewed = 0
    scores: list[int] = []
    distribution: dict[int, int] = defaultdict(int)
    for row in rows:
        detail = detail_by_id[row['id']]
        should_review = args.llm_review_mode == 'all' or (
            args.llm_review_mode == 'cot_quality' and detail['cot_complete']
        )
        if not should_review:
            continue
        if args.llm_review_max_cases is not None and reviewed >= args.llm_review_max_cases:
            break
        cot = extract_think_block(row['response'])
        result = deepseek_score(row, cot, args)
        reviewed += 1
        detail['cot_quality_score'] = result['score']
        detail['cot_quality_reason'] = result['reason']
        scores.append(result['score'])
        distribution[result['score']] += 1
        review_rows.append({
            'id': row['id'],
            'view': row['view'],
            'gold_answer': row['gold_answer'],
            'cot_complete': detail['cot_complete'],
            'score': result['score'],
            'reason': result['reason'],
            'usage': result['usage'],
        })

    mean_score = sum(scores) / len(scores) if scores else None
    metrics['cot_quality_score'] = {
        'enabled': True,
        'mode': args.llm_review_mode,
        'review_model': args.llm_review_model,
        'reviewed_cases': reviewed,
        'mean': mean_score,
        'distribution': dict(distribution),
    }
    review_path = run_dir / 'metrics' / f'{model_name}_mt{max_tokens}_cot_quality.json'
    review_path.write_text(json.dumps(review_rows, ensure_ascii=False, indent=2), encoding='utf-8')


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


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    max_tokens_list = parse_max_tokens_list(args.max_tokens_list)
    model_specs = collect_models(args)

    if args.existing_run_dir:
        run_dir = args.existing_run_dir.resolve()
        test_file = args.test_file
        rows_for_review: dict[tuple[str, int], list[dict]] = {}
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
        rows_for_review = {}

    (run_dir / 'metrics').mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}

    for spec in model_specs:
        name = spec['name']
        adapter = spec['adapter']
        expected = spec['expected_style']
        results[name] = {}
        engine = None
        for mt in max_tokens_list:
            if args.existing_run_dir:
                pred_rows = load_existing(run_dir, name, mt)
            else:
                if engine is None:
                    engine = build_engine(adapter, args.max_batch_size)
                pred_rows = run_inference(name, adapter, rows, mt, args, run_dir, engine=engine)
            rows_for_review[(name, mt)] = pred_rows
            metrics = compute_metrics(pred_rows, expected_style=expected)
            apply_cot_quality(metrics, pred_rows, run_dir, name, mt, args)
            results[name][str(mt)] = metrics
        engine = None  # free GPU before next model

    summary = {
        'run_dir': str(run_dir),
        'dataset_tag': args.dataset_tag,
        'test_file': str(test_file) if test_file else None,
        'count': len(next(iter(rows_for_review.values()))) if rows_for_review else 0,
        'decode_max_tokens_list': max_tokens_list,
        'llm_review_mode': args.llm_review_mode,
        'models': model_specs,
        'results': results,
    }
    summary_path = run_dir / 'metrics' / 'summary.json'
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    headline = {
        'run_dir': summary['run_dir'],
        'models': [m['name'] for m in model_specs],
        'max_tokens_list': max_tokens_list,
    }
    print(json.dumps(headline, ensure_ascii=False))


if __name__ == '__main__':
    if '--help' in sys.argv or '-h' in sys.argv:
        parse_args()
        raise SystemExit(0)
    main()
