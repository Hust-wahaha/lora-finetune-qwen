from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.common.naming import DEFAULT_MODEL_TAG, default_dataset_tag, make_run_dir

MODEL_ID = 'Qwen/Qwen3.5-0.8B'
SYSTEM = '你是一个擅长中文数学应用题的助手，请尽量给出简洁且正确的答案。'

SAMPLES = [
    {'id': 'modern_give', 'style': 'modern', 'query': '小明有12颗糖，送给小红5颗，还剩几颗？'},
    {'id': 'classical_give', 'style': 'classical', 'query': '明有糖12颗，遗红5颗，尚余几何？'},
    {'id': 'modern_price', 'style': 'modern', 'query': '一支钢笔9元，买4支需要多少钱？'},
    {'id': 'classical_price', 'style': 'classical', 'query': '一笔值9元，购4支，共费几何？'},
    {'id': 'modern_div', 'style': 'modern', 'query': '把24个苹果平均分给6个小朋友，每人分几个？'},
    {'id': 'classical_div', 'style': 'classical', 'query': '有苹果24个，均分6人，人得几何？'},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, default=None)
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument('--dataset-tag', type=str, default=default_dataset_tag('think'))
    parser.add_argument('--max-tokens', type=int, default=512)
    return parser.parse_args()


def build_engine(adapter: Path | None):
    from peft import PeftModel
    from swift import get_model_processor, get_template
    from swift.infer_engine import TransformersEngine

    model, tokenizer = get_model_processor(MODEL_ID)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    template_type = model.model_meta.template
    template = get_template(tokenizer, template_type=template_type, default_system=SYSTEM)
    return TransformersEngine(model, template=template, max_batch_size=len(SAMPLES))


def main() -> None:
    args = parse_args()
    run_dir = make_run_dir(
        stage='inspect',
        dataset_tag=args.dataset_tag,
        model_tag=DEFAULT_MODEL_TAG,
        suffix=args.tag or 'think-samples',
    )
    from swift.infer_engine import InferRequest, RequestConfig
    engine = build_engine(args.checkpoint)
    requests = [InferRequest(messages=[{'role': 'user', 'content': row['query']}]) for row in SAMPLES]
    request_config = RequestConfig(max_tokens=args.max_tokens, temperature=0.0, top_p=1.0, top_k=20)
    responses = engine.infer(requests, request_config)

    output_rows = []
    for row, response in zip(SAMPLES, responses):
        output_rows.append({
            'id': row['id'],
            'style': row['style'],
            'query': row['query'],
            'response': response.choices[0].message.content,
        })

    output_path = run_dir / 'predictions' / 'think_samples.json'
    output_path.write_text(json.dumps(output_rows, ensure_ascii=False, indent=2), encoding='utf-8')
    summary = {
        'checkpoint': str(args.checkpoint) if args.checkpoint else None,
        'output_file': str(output_path),
        'count': len(output_rows),
    }
    (run_dir / 'metrics' / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    if '--help' in sys.argv or '-h' in sys.argv:
        parse_args()
        raise SystemExit(0)
    main()
