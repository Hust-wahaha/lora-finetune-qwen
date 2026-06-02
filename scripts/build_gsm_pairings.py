"""
从 data/final/gsm-1k/{train,test}/modern.jsonl 派生 6 种 (输入题面 × CoT 风格) 组合，
并从 train 中切 10% 作为验证集。

6 个 tag = 输入(2) × CoT(3)：

| tag | user 输入字段       | assistant CoT 字段 |
| --- | ------------------ | ------------------ |
| m2m | modern_question    | modern_think       |
| m2c | modern_question    | classical_think    |
| m2s | modern_question    | structured_think   |
| c2m | classical_question | modern_think       |
| c2c | classical_question | classical_think    |
| c2s | classical_question | structured_think   |

输出目录：data/final/gsm-1k-pairings/{train,val,test}_gsm1k_{tag}.jsonl + summary.json
规模（train 991 条 → 切 10% = 99 条 val，892 条 train；test 固定 248 条）：
  train: ~892 × 6 = 5352 条，val: ~99 × 6 = 594 条，test: 248 × 6 = 1488 条

约定（与 src/common/schema.py 对齐）：
- 保留源文件的所有原始字段，保证 validate_dataset.py 的 CORE + TRAINING 字段齐全。
- system 提示按 cot_style 设置（白话文 / 文言文 / 结构化方式），与 gsm-1k 原始文件一致。
- think_style 字段被覆盖为 modern / classical / structured。
- split 字段更新为 train / val / test。
- id 格式：{base_id}-{tag}，例如 gsm8k-0-m2m。
- view 字段取 pairing['input_view']（modern 或 classical），代表输入题面语言。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.common.paths import DATA_DIR  # noqa: E402
from src.common.schema import THINK_ANSWER_BOUNDARY, THINK_TAG_PREFIX  # noqa: E402

SOURCE_DIR = DATA_DIR / 'final' / 'gsm-1k'
OUTPUT_DIR_DEFAULT = DATA_DIR / 'final' / 'gsm-1k-pairings'
VAL_RATIO = 0.1
RANDOM_SEED = 42

# system 提示按 cot_style 区分，与 gsm-1k 原始文件风格一致
SYSTEM_BY_COT_STYLE = {
    'modern':     "你是一个擅长中文数学应用题的助手，要求输出简洁、准确、可验证。请用白话文进行思考，并将最终答案输出在字符'答案：'后面",
    'classical':  "你是一个擅长中文数学应用题的助手，要求输出简洁、准确、可验证。请用文言文进行思考，并将最终答案输出在字符'答案：'后面",
    'structured': "你是一个擅长中文数学应用题的助手，要求输出简洁、准确、可验证。请用结构化方式进行思考，并将最终答案输出在字符'答案：'后面",
}

PAIRINGS = [
    {'tag': 'm2m', 'input_view': 'modern',    'cot_style': 'modern',     'cot_field': 'modern_think'},
    {'tag': 'm2c', 'input_view': 'modern',    'cot_style': 'classical',  'cot_field': 'classical_think'},
    {'tag': 'm2s', 'input_view': 'modern',    'cot_style': 'structured', 'cot_field': 'structured_think'},
    {'tag': 'c2m', 'input_view': 'classical', 'cot_style': 'modern',     'cot_field': 'modern_think'},
    {'tag': 'c2c', 'input_view': 'classical', 'cot_style': 'classical',  'cot_field': 'classical_think'},
    {'tag': 'c2s', 'input_view': 'classical', 'cot_style': 'structured', 'cot_field': 'structured_think'},
]

REQUIRED_SOURCE_FIELDS = (
    'base_id', 'source', 'family', 'answer',
    'modern_question', 'classical_question',
    'modern_think', 'classical_think', 'structured_think',
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', type=Path, default=SOURCE_DIR,
                        help='gsm-1k 源目录（含 train/ 和 test/ 子目录）')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR_DEFAULT,
                        help='派生数据输出目录（默认 data/final/gsm-1k-pairings/）')
    parser.add_argument('--val-ratio', type=float, default=VAL_RATIO,
                        help='从 train 切出用作验证集的比例（默认 0.1）')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                        help='切 val 时的随机种子（默认 42）')
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open('r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def validate_source_rows(rows: list[dict], path: Path) -> None:
    if not rows:
        raise ValueError(f'Empty source file: {path}')
    for idx, row in enumerate(rows, start=1):
        missing = [f for f in REQUIRED_SOURCE_FIELDS if f not in row]
        if missing:
            raise ValueError(
                f'Source schema invalid at {path}:{idx} '
                f'(base_id={row.get("base_id", "<missing>")!r}); missing fields: {missing}'
            )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def assistant_with_think(cot_text: str, answer: str) -> str:
    return f'{THINK_TAG_PREFIX}{cot_text}{THINK_ANSWER_BOUNDARY}{answer}。'


def derive_record(src: dict, pairing: dict, split: str) -> dict:
    cot_style = pairing['cot_style']
    cot_text = src[pairing['cot_field']]
    user_text = src['modern_question'] if pairing['input_view'] == 'modern' else src['classical_question']
    system_text = SYSTEM_BY_COT_STYLE[cot_style]
    new_record = dict(src)
    new_record['id'] = f"{src['base_id']}-{pairing['tag']}"
    new_record['view'] = pairing['input_view']
    new_record['input_view'] = pairing['input_view']
    new_record['cot_style'] = cot_style
    new_record['think_style'] = cot_style
    new_record['split'] = split
    new_record['messages'] = [
        {'role': 'system',    'content': system_text},
        {'role': 'user',      'content': user_text},
        {'role': 'assistant', 'content': assistant_with_think(cot_text, src['answer'])},
    ]
    return new_record


def split_train_val(rows: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_ratio))
    return shuffled[n_val:], shuffled[:n_val]


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir.resolve()
    source_dir = args.source_dir.resolve()

    # 只需读 modern.jsonl 作为去重的权威来源（其他文件 base_id 完全相同，原始字段相同）
    train_src_path = source_dir / 'train' / 'modern.jsonl'
    test_src_path = source_dir / 'test' / 'modern.jsonl'

    all_train = read_jsonl(train_src_path)
    all_test = read_jsonl(test_src_path)
    validate_source_rows(all_train, train_src_path)
    validate_source_rows(all_test, test_src_path)

    train_rows, val_rows = split_train_val(all_train, args.val_ratio, args.seed)
    print(f'train={len(train_rows)}, val={len(val_rows)}, test={len(all_test)} (source records)')

    split_data = {
        'train': train_rows,
        'val':   val_rows,
        'test':  all_test,
    }

    summary: dict = {
        'source_dir': str(source_dir),
        'output_dir': str(out_dir),
        'val_ratio': args.val_ratio,
        'seed': args.seed,
        'source_counts': {s: len(r) for s, r in split_data.items()},
        'pairings': {},
    }

    per_tag_counts: dict[str, dict] = {p['tag']: {} for p in PAIRINGS}

    for pairing in PAIRINGS:
        tag = pairing['tag']
        for split, rows in split_data.items():
            derived = [derive_record(r, pairing, split) for r in rows]
            out_path = out_dir / f'{split}_gsm1k_{tag}.jsonl'
            write_jsonl(out_path, derived)
            per_tag_counts[tag][split] = len(derived)

    for pairing in PAIRINGS:
        tag = pairing['tag']
        summary['pairings'][f'gsm1k_{tag}'] = {
            'input_view': pairing['input_view'],
            'cot_style':  pairing['cot_style'],
            'cot_field':  pairing['cot_field'],
            'counts':     per_tag_counts[tag],
        }

    summary_path = out_dir / 'summary.json'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
