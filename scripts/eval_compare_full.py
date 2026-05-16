import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from peft import PeftModel
from swift import get_model_processor, get_template
from swift.infer_engine import InferRequest, RequestConfig, TransformersEngine

MODEL_ID = 'Qwen/Qwen3.5-0.8B'
SYSTEM = '你是一个擅长中文数学应用题的助手，请尽量给出简洁且正确的答案。'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--test-file', type=Path, default=None)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--max-tokens', type=int, default=512)
    parser.add_argument('--max-batch-size', type=int, default=8)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top-p', type=float, default=1.0)
    parser.add_argument('--top-k', type=int, default=20)
    parser.add_argument('--repetition-penalty', type=float, default=1.0)
    parser.add_argument('--tag', type=str, default='compare_full')
    return parser.parse_args()


def build_engine(adapter: str | None, max_batch_size: int):
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
            'gold_answer': row['gold_answer'],
            'pred_answer': pred_answer,
            'correct': correct,
            'response_chars': len(row['response']),
            'has_think': '<think>' in row['response'],
            'has_answer_marker': has_answer_marker,
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


def run_model(name: str, adapter: str | None, requests: list[InferRequest], rows: list[dict], run_dir: Path, args: argparse.Namespace) -> list[dict]:
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


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    test_file = args.test_file or (root / 'data' / 'final' / 'test_s800.jsonl')
    rows = [json.loads(line) for line in test_file.read_text(encoding='utf-8').splitlines() if line.strip()]
    requests = [InferRequest(messages=[{'role': 'user', 'content': row['messages'][1]['content']}]) for row in rows]
    run_dir = root / 'runs' / (datetime.now().strftime('%Y%m%d_%H%M%S') + f'_{args.tag}')
    (run_dir / 'predictions').mkdir(parents=True, exist_ok=True)
    (run_dir / 'metrics').mkdir(parents=True, exist_ok=True)

    baseline_rows = run_model('baseline', None, requests, rows, run_dir, args)
    finetuned_rows = run_model('finetuned', str(args.checkpoint), requests, rows, run_dir, args)

    summary = {
        'checkpoint': str(args.checkpoint),
        'test_file': str(test_file),
        'count': len(rows),
        'decode_max_tokens': args.max_tokens,
        'baseline': summarize('baseline', baseline_rows),
        'finetuned': summarize('finetuned', finetuned_rows),
    }
    summary_path = run_dir / 'metrics' / 'compare_summary.json'
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({
        'run_dir': str(run_dir),
        'baseline_acc': summary['baseline']['overall']['accuracy'],
        'finetuned_acc': summary['finetuned']['overall']['accuracy'],
        'baseline_answer_rate': summary['baseline']['overall']['answer_marker_rate'],
        'finetuned_answer_rate': summary['finetuned']['overall']['answer_marker_rate'],
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
