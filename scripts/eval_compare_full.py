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

MODEL_ID = 'Qwen/Qwen3.5-0.8B'
SYSTEM = '你是一个擅长中文数学应用题的助手，请尽量给出简洁且正确的答案。'
DEFAULT_DEEPSEEK_ENDPOINT = 'https://api.deepseek.com/chat/completions'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--dataset-tag', type=str, default='s800')
    parser.add_argument('--test-file', type=Path, default=None)
    parser.add_argument('--checkpoint', type=Path, required=False)
    parser.add_argument('--existing-run-dir', type=Path, default=None)
    parser.add_argument('--max-tokens', type=int, default=512)
    parser.add_argument('--max-batch-size', type=int, default=8)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top-p', type=float, default=1.0)
    parser.add_argument('--top-k', type=int, default=20)
    parser.add_argument('--repetition-penalty', type=float, default=1.0)
    parser.add_argument('--run-tag', type=str, default='rule')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--llm-review-mode', choices=['none', 'mismatches', 'all'], default='none')
    parser.add_argument('--llm-review-model', type=str, default='deepseek-v4-flash')
    parser.add_argument('--llm-review-api-key-env', type=str, default='DEEPSEEK_API_KEY')
    parser.add_argument('--llm-review-endpoint', type=str, default=DEFAULT_DEEPSEEK_ENDPOINT)
    parser.add_argument('--llm-review-max-cases', type=int, default=None)
    parser.add_argument('--llm-review-timeout', type=int, default=120)
    return parser.parse_args()


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


def finish(metric: dict[str, int]) -> dict[str, float | int]:
    total = metric['total']
    return {
        'total': total,
        'correct': metric['correct'],
        'accuracy': metric['correct'] / total if total else 0.0,
        'has_answer_marker': metric['has_answer_marker'],
        'answer_marker_rate': metric['has_answer_marker'] / total if total else 0.0,
    }


def extract_first_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith('{') and text.endswith('}'):
        return json.loads(text)
    match = re.search(r'\{.*\}', text, re.S)
    if not match:
        raise ValueError(f'No JSON object found in review response: {text[:200]}')
    return json.loads(match.group(0))


def review_prompt(row: dict) -> str:
    return (
        '请做数学答案裁判。任务很简单：判断“assistant response”最终是否给出了与标准答案完全一致的数值答案。'
        '只看最终数值是否正确，不要求措辞一致。'
        '如果回答过程正确但没有明确最终答案，也按 incorrect 处理。'
        '只返回 JSON，对应字段为: '
        '{"verdict":"correct|incorrect","final_answer":"提取到的最终数值或空字符串","reason":"一句简短中文说明"}。\n\n'
        f'question: {row["query"]}\n'
        f'gold_answer: {row["gold_answer"]}\n'
        f'assistant_response: {row["response"]}\n'
    )


def deepseek_review(row: dict, args: argparse.Namespace) -> dict:
    api_key = os.environ.get(args.llm_review_api_key_env)
    if not api_key:
        raise RuntimeError(f'Missing API key env: {args.llm_review_api_key_env}')
    payload = {
        'model': args.llm_review_model,
        'messages': [
            {
                'role': 'system',
                'content': '你是严格的数学答案裁判。不要解释，不要输出额外文本，只输出 JSON。',
            },
            {
                'role': 'user',
                'content': review_prompt(row),
            },
        ],
        'temperature': 0.0,
        'max_tokens': 256,
        'response_format': {'type': 'json_object'},
        'thinking': {'type': 'disabled'},
    }
    request = urllib.request.Request(
        args.llm_review_endpoint,
        data=json.dumps(payload).encode('utf-8'),
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}',
        },
        method='POST',
    )
    try:
        with urllib.request.urlopen(request, timeout=args.llm_review_timeout) as response:
            body = json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode('utf-8', errors='replace')
        raise RuntimeError(f'DeepSeek review HTTP error {exc.code}: {detail}') from exc
    content = body['choices'][0]['message']['content']
    parsed = extract_first_json_object(content)
    verdict = parsed.get('verdict')
    if verdict not in {'correct', 'incorrect'}:
        raise ValueError(f'Unexpected DeepSeek verdict: {parsed}')
    return {
        'verdict': verdict,
        'final_answer': str(parsed.get('final_answer', '')),
        'reason': str(parsed.get('reason', '')).strip(),
        'raw_response': content,
        'usage': body.get('usage', {}),
    }


def run_model(
    name: str,
    adapter: str | None,
    requests: list[InferRequest],
    rows: list[dict],
    run_dir: Path,
    args: argparse.Namespace,
) -> list[dict]:
    engine = build_engine(adapter, args.max_batch_size)
    request_config = RequestConfig(
        max_tokens=args.max_tokens,
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
    output_path = run_dir / 'predictions' / f'{name}.json'
    output_path.write_text(json.dumps(output_rows, ensure_ascii=False, indent=2), encoding='utf-8')
    return output_rows


def summarize(name: str, rows: list[dict]) -> dict:
    overall = {'total': 0, 'correct': 0, 'has_answer_marker': 0}
    by_family = defaultdict(lambda: {'total': 0, 'correct': 0, 'has_answer_marker': 0})
    by_view = defaultdict(lambda: {'total': 0, 'correct': 0, 'has_answer_marker': 0})
    total_chars = 0
    total_lines = 0
    think_count = 0
    detailed = []
    for row in rows:
        pred_answer = extract_answer(row['response'])
        correct = pred_answer == row['gold_answer']
        has_answer_marker = '答案' in row['response']
        overall['total'] += 1
        overall['correct'] += int(correct)
        overall['has_answer_marker'] += int(has_answer_marker)
        by_family[row['family']]['total'] += 1
        by_family[row['family']]['correct'] += int(correct)
        by_family[row['family']]['has_answer_marker'] += int(has_answer_marker)
        by_view[row['view']]['total'] += 1
        by_view[row['view']]['correct'] += int(correct)
        by_view[row['view']]['has_answer_marker'] += int(has_answer_marker)
        total_chars += len(row['response'])
        total_lines += row['response'].count('\n') + 1
        think_count += int('<think>' in row['response'])
        detailed.append({
            'id': row['id'],
            'family': row['family'],
            'view': row['view'],
            'query': row['query'],
            'gold_answer': row['gold_answer'],
            'pred_answer': pred_answer,
            'rule_correct': correct,
            'response_chars': len(row['response']),
            'has_think': '<think>' in row['response'],
            'has_answer_marker': has_answer_marker,
            'reviewed_by_llm': False,
            'llm_verdict': None,
            'llm_final_answer': None,
            'llm_reason': None,
            'final_correct': correct,
        })
    total = overall['total'] or 1
    return {
        'model_name': name,
        'overall': {
            **finish(overall),
            'avg_chars': total_chars / total,
            'avg_lines': total_lines / total,
            'think_rate': think_count / total,
        },
        'by_family': {key: finish(value) for key, value in by_family.items()},
        'by_view': {key: finish(value) for key, value in by_view.items()},
        'detailed': detailed,
    }


def should_review(detail: dict, mode: str) -> bool:
    if mode == 'none':
        return False
    if mode == 'all':
        return True
    return not detail['rule_correct']


def apply_llm_review(summary: dict, rows: list[dict], run_dir: Path, args: argparse.Namespace) -> None:
    if args.llm_review_mode == 'none':
        summary['llm_review'] = {
            'enabled': False,
            'mode': 'none',
        }
        return

    detail_by_id = {detail['id']: detail for detail in summary['detailed']}
    review_rows = []
    reviewed_cases = 0
    changed_to_correct = 0
    changed_to_incorrect = 0
    final_correct = 0
    for row in rows:
        detail = detail_by_id[row['id']]
        if should_review(detail, args.llm_review_mode):
            if args.llm_review_max_cases is not None and reviewed_cases >= args.llm_review_max_cases:
                detail['final_correct'] = detail['rule_correct']
            else:
                review = deepseek_review(row, args)
                reviewed_cases += 1
                detail['reviewed_by_llm'] = True
                detail['llm_verdict'] = review['verdict']
                detail['llm_final_answer'] = review['final_answer']
                detail['llm_reason'] = review['reason']
                detail['final_correct'] = review['verdict'] == 'correct'
                if detail['final_correct'] and not detail['rule_correct']:
                    changed_to_correct += 1
                if (not detail['final_correct']) and detail['rule_correct']:
                    changed_to_incorrect += 1
                review_rows.append({
                    'id': row['id'],
                    'family': row['family'],
                    'view': row['view'],
                    'gold_answer': row['gold_answer'],
                    'rule_pred_answer': detail['pred_answer'],
                    'rule_correct': detail['rule_correct'],
                    'llm_verdict': review['verdict'],
                    'llm_final_answer': review['final_answer'],
                    'llm_reason': review['reason'],
                    'usage': review['usage'],
                })
        final_correct += int(detail['final_correct'])

    total = len(summary['detailed']) or 1
    summary['llm_review'] = {
        'enabled': True,
        'mode': args.llm_review_mode,
        'review_model': args.llm_review_model,
        'reviewed_cases': reviewed_cases,
        'changed_to_correct': changed_to_correct,
        'changed_to_incorrect': changed_to_incorrect,
        'final_correct': final_correct,
        'final_accuracy': final_correct / total,
    }
    review_path = run_dir / 'metrics' / f'{summary["model_name"]}_llm_review.json'
    review_path.write_text(json.dumps(review_rows, ensure_ascii=False, indent=2), encoding='utf-8')


def load_rows_from_test_file(test_file: Path, limit: int | None) -> list[dict]:
    rows = [json.loads(line) for line in test_file.read_text(encoding='utf-8').splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def load_existing_predictions(run_dir: Path) -> tuple[list[dict], list[dict]]:
    baseline_rows = json.loads((run_dir / 'predictions' / 'baseline.json').read_text(encoding='utf-8'))
    finetuned_rows = json.loads((run_dir / 'predictions' / 'finetuned.json').read_text(encoding='utf-8'))
    return baseline_rows, finetuned_rows


def print_result(summary: dict) -> dict:
    result = {
        'run_dir': summary['run_dir'],
        'baseline_rule_acc': summary['baseline']['overall']['accuracy'],
        'finetuned_rule_acc': summary['finetuned']['overall']['accuracy'],
    }
    if summary['baseline']['llm_review']['enabled']:
        result['baseline_final_acc'] = summary['baseline']['llm_review']['final_accuracy']
        result['finetuned_final_acc'] = summary['finetuned']['llm_review']['final_accuracy']
    return result


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if args.existing_run_dir:
        run_dir = args.existing_run_dir.resolve()
        baseline_rows, finetuned_rows = load_existing_predictions(run_dir)
        checkpoint = args.checkpoint or 'unknown'
        test_file = args.test_file or 'loaded_from_existing_predictions'
    else:
        if not args.checkpoint:
            raise ValueError('--checkpoint is required unless --existing-run-dir is provided')
        test_file = (args.test_file or dataset_file('test', args.dataset_tag)).resolve()
        rows = load_rows_from_test_file(test_file, args.limit)
        from swift.infer_engine import InferRequest
        requests = [InferRequest(messages=[{'role': 'user', 'content': row['messages'][1]['content']}]) for row in rows]
        run_dir = make_run_dir(
            stage='eval',
            dataset_tag=args.dataset_tag,
            model_tag=DEFAULT_MODEL_TAG,
            suffix=args.run_tag,
            base_dir=root / 'runs',
        )
        baseline_rows = run_model('baseline', None, requests, rows, run_dir, args)
        finetuned_rows = run_model('finetuned', str(args.checkpoint), requests, rows, run_dir, args)
        checkpoint = str(args.checkpoint)

    (run_dir / 'metrics').mkdir(parents=True, exist_ok=True)
    baseline_summary = summarize('baseline', baseline_rows)
    finetuned_summary = summarize('finetuned', finetuned_rows)
    apply_llm_review(baseline_summary, baseline_rows, run_dir, args)
    apply_llm_review(finetuned_summary, finetuned_rows, run_dir, args)

    summary = {
        'run_dir': str(run_dir),
        'checkpoint': str(checkpoint),
        'test_file': str(test_file),
        'count': len(baseline_rows),
        'decode_max_tokens': args.max_tokens,
        'llm_review_mode': args.llm_review_mode,
        'baseline': baseline_summary,
        'finetuned': finetuned_summary,
    }
    summary_path = run_dir / 'metrics' / 'compare_summary.json'
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(print_result(summary), ensure_ascii=False))


if __name__ == '__main__':
    if '--help' in sys.argv or '-h' in sys.argv:
        parse_args()
        raise SystemExit(0)
    main()
