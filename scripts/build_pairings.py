"""
从 data/final/{train,val,test}_s800_think.jsonl 派生 6 种 (输入题面 × CoT 风格) 组合。

6 个 tag = 输入(2) × CoT(3)：

| tag | user 输入字段        | assistant CoT 字段  |
| --- | ------------------- | ------------------ |
| m2m | modern_question     | modern_think       |
| m2c | modern_question     | classical_think    |
| m2s | modern_question     | structured_think   |
| c2m | classical_question  | modern_think       |
| c2c | classical_question  | classical_think    |
| c2s | classical_question  | structured_think   |

输出目录：data/pairings/{train,val,test}_s800_{tag}.jsonl + summary.json。

约定（与 src/common/schema.py 对齐）：
- 每条派生记录保留所有原始字段（modern_question / classical_question / modern_think /
  classical_think / structured_think / structured_cot / answer / family / split /
  source / base_id 等），保证 validate_dataset.py 的 CORE + TRAINING 字段齐全。
- 仅改写 id（追加 -{tag} 后缀）、view（仍代表输入题面来源）、messages（user 与
  assistant 内容按 tag 重组）、新增 input_view / cot_style 两个元数据字段。
- think_style 字段被覆盖为 modern / classical / structured 三选一，便于评测脚本分桶。
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

from src.common.paths import DATA_DIR, FINAL_DATA_DIR  # noqa: E402
from src.common.schema import THINK_ANSWER_BOUNDARY, THINK_TAG_PREFIX  # noqa: E402

SOURCE_DATASET_TAG = 's800_think'
SOURCE_SPLITS = ('train', 'val', 'test')
OUTPUT_DIR_NAME = 'pairings'

PAIRINGS = [
    {'tag': 'm2m', 'input_view': 'modern',    'cot_style': 'modern',     'cot_field': 'modern_think'},
    {'tag': 'm2c', 'input_view': 'modern',    'cot_style': 'classical',  'cot_field': 'classical_think'},
    {'tag': 'm2s', 'input_view': 'modern',    'cot_style': 'structured', 'cot_field': 'structured_think'},
    {'tag': 'c2m', 'input_view': 'classical', 'cot_style': 'modern',     'cot_field': 'modern_think'},
    {'tag': 'c2c', 'input_view': 'classical', 'cot_style': 'classical',  'cot_field': 'classical_think'},
    {'tag': 'c2s', 'input_view': 'classical', 'cot_style': 'structured', 'cot_field': 'structured_think'},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-tag', default=SOURCE_DATASET_TAG,
                        help='源数据集 tag，对应 data/final/{split}_{tag}.jsonl')
    parser.add_argument('--output-dir', type=Path, default=DATA_DIR / OUTPUT_DIR_NAME,
                        help='派生数据输出目录（默认 data/pairings/）')
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open('r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def assistant_with_think(cot_text: str, answer: str) -> str:
    return f'{THINK_TAG_PREFIX}{cot_text}{THINK_ANSWER_BOUNDARY}{answer}。'


def derive_record(src: dict, pairing: dict) -> dict:
    base_id = src.get('base_id') or src['id'].rsplit('-', 1)[0]
    cot_text = src[pairing['cot_field']]
    user_text = src['modern_question'] if pairing['input_view'] == 'modern' else src['classical_question']
    system_msg = src['messages'][0] if src.get('messages') else {
        'role': 'system',
        'content': '你是一个擅长中文数学应用题的助手，要求输出简洁、准确、可验证。',
    }
    new_record = dict(src)
    new_record['id'] = f"{base_id}-{pairing['tag']}"
    new_record['base_id'] = base_id
    new_record['view'] = pairing['input_view']
    new_record['input_view'] = pairing['input_view']
    new_record['cot_style'] = pairing['cot_style']
    new_record['think_style'] = pairing['cot_style']
    new_record['messages'] = [
        system_msg,
        {'role': 'user', 'content': user_text},
        {'role': 'assistant', 'content': assistant_with_think(cot_text, src['answer'])},
    ]
    return new_record


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir.resolve()

    summary: dict = {
        'source_tag': args.source_tag,
        'output_dir': str(out_dir),
        'pairings': {},
    }

    for pairing in PAIRINGS:
        tag = pairing['tag']
        per_tag_counts: dict[str, int] = {}
        for split in SOURCE_SPLITS:
            src_path = FINAL_DATA_DIR / f'{split}_{args.source_tag}.jsonl'
            src_rows = read_jsonl(src_path)
            # 仅保留与本 tag 输入视角匹配的源记录，避免 modern 与 classical 重复
            picked = [r for r in src_rows if r.get('view') == pairing['input_view']]
            derived = [derive_record(r, pairing) for r in picked]
            out_path = out_dir / f'{split}_s800_{tag}.jsonl'
            write_jsonl(out_path, derived)
            per_tag_counts[split] = len(derived)

            # 顺手统计 family 分桶，便于 sanity check
            family_dist = defaultdict(int)
            for row in derived:
                family_dist[row['family']] += 1
            per_tag_counts[f'{split}_by_family'] = dict(sorted(family_dist.items()))

        summary['pairings'][f's800_{tag}'] = {
            'input_view': pairing['input_view'],
            'cot_style': pairing['cot_style'],
            'cot_field': pairing['cot_field'],
            'counts': per_tag_counts,
        }

    summary_path = out_dir / 'summary.json'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
