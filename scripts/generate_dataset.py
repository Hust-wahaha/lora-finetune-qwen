from __future__ import annotations

import argparse
import json
import random
import sys
from statistics import mean
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.common.naming import (
    dataset_file,
    dataset_summary_file,
    default_dataset_tag,
    interim_aligned_file,
)
from src.common.schema import THINK_STYLES

ROOT = ROOT_DIR

SYSTEM = '你是一个擅长中文数学应用题的助手，要求输出简洁、准确、可验证。'
rng = random.Random(42)

FAMILY_COUNTS = {
    'give_away': 70,
    'box_total': 70,
    'equal_division': 70,
    'unit_price': 70,
    'bundle_count': 60,
    'weight_remaining': 60,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-tag', type=str, default=None)
    parser.add_argument('--variant', choices=['visible', 'think'], default='visible')
    parser.add_argument('--think-style', choices=THINK_STYLES, default='by_view')
    return parser.parse_args()


def resolve_dataset_tag(args: argparse.Namespace) -> str:
    if args.dataset_tag:
        return args.dataset_tag
    return default_dataset_tag(args.variant)


def split_name(i: int) -> str:
    mod = i % 10
    if mod < 8:
        return 'train'
    if mod == 8:
        return 'val'
    return 'test'


def build_record(
    base_id: str,
    family: str,
    split: str,
    modern: str,
    classical: str,
    structured_think: str,
    modern_think: str,
    classical_think: str,
    answer: str,
) -> dict:
    return {
        'id': base_id,
        'source': 'synthetic-template-s800',
        'family': family,
        'split': split,
        'modern_question': modern,
        'classical_question': classical,
        'structured_think': structured_think,
        # Historical compatibility field. New code should use `structured_think`.
        'structured_cot': structured_think,
        'modern_think': modern_think,
        'classical_think': classical_think,
        'answer': answer,
    }


def render_assistant_content(
    *,
    variant: str,
    think_style: str,
    view: str,
    structured_think: str,
    modern_think: str,
    classical_think: str,
    answer: str,
) -> str:
    if variant == 'visible':
        return f'{structured_think} 答案：{answer}。'

    if think_style == 'structured':
        think_text = structured_think
    elif view == 'modern':
        think_text = modern_think
    else:
        think_text = classical_think
    return f'<think>\n{think_text}\n</think>\n\n答案：{answer}。'


def render_messages(user_text: str, assistant_text: str) -> list[dict]:
    return [
        {'role': 'system', 'content': SYSTEM},
        {'role': 'user', 'content': user_text},
        {'role': 'assistant', 'content': assistant_text},
    ]


def expand_training_records(item: dict, variant: str, think_style: str) -> list[dict]:
    common = {
        'base_id': item['id'],
        'source': item['source'],
        'family': item['family'],
        'split': item['split'],
        'modern_question': item['modern_question'],
        'classical_question': item['classical_question'],
        'structured_think': item['structured_think'],
        'modern_think': item['modern_think'],
        'classical_think': item['classical_think'],
        'answer': item['answer'],
        'dataset_variant': variant,
        'think_style': think_style,
    }
    modern_assistant = render_assistant_content(
        variant=variant,
        think_style=think_style,
        view='modern',
        structured_think=item['structured_think'],
        modern_think=item['modern_think'],
        classical_think=item['classical_think'],
        answer=item['answer'],
    )
    classical_assistant = render_assistant_content(
        variant=variant,
        think_style=think_style,
        view='classical',
        structured_think=item['structured_think'],
        modern_think=item['modern_think'],
        classical_think=item['classical_think'],
        answer=item['answer'],
    )
    return [
        {
            **common,
            'id': f"{item['id']}-modern",
            'view': 'modern',
            'messages': render_messages(item['modern_question'], modern_assistant),
        },
        {
            **common,
            'id': f"{item['id']}-classical",
            'view': 'classical',
            'messages': render_messages(item['classical_question'], classical_assistant),
        },
    ]


def build_aligned_records() -> list[dict]:
    aligned = []

    for i in range(FAMILY_COUNTS['give_away']):
        a = rng.randint(12, 60)
        b = rng.randint(3, max(3, a // 2))
        remain = a - b
        aligned.append(build_record(
            f'give-{i:03d}',
            'give_away',
            split_name(i),
            f'小明有{a}颗糖，送给小红{b}颗，还剩几颗？',
            f'明有糖{a}颗，遗红{b}颗，尚余几何？',
            f'初有{a}颗，送出{b}颗，故{a}-{b}={remain}。',
            f'原来有{a}颗糖，送给小红{b}颗，所以用减法：{a}-{b}={remain}。',
            f'初有{a}颗，遗{b}颗，故{a}-{b}={remain}。',
            str(remain),
        ))

    for i in range(FAMILY_COUNTS['box_total']):
        boxes = rng.randint(2, 12)
        each = rng.randint(4, 15)
        total = boxes * each
        aligned.append(build_record(
            f'box-{i:03d}',
            'box_total',
            split_name(i),
            f'一盒铅笔有{each}支，{boxes}盒一共有多少支？',
            f'一匣铅笔{each}支，{boxes}匣共几支？',
            f'每匣{each}支，{boxes}匣则{each}×{boxes}={total}。',
            f'每盒有{each}支铅笔，一共{boxes}盒，所以用乘法：{each}×{boxes}={total}。',
            f'每匣{each}支，凡{boxes}匣，故{each}乘{boxes}得{total}。',
            str(total),
        ))

    for i in range(FAMILY_COUNTS['equal_division']):
        people = rng.randint(2, 12)
        per = rng.randint(3, 15)
        total = people * per
        aligned.append(build_record(
            f'div-{i:03d}',
            'equal_division',
            split_name(i),
            f'把{total}个苹果平均分给{people}个小朋友，每人分几个？',
            f'有苹果{total}个，均分{people}人，人得几何？',
            f'总数{total}，分与{people}人，故{total}÷{people}={per}。',
            f'共有{total}个苹果，平均分给{people}个人，所以用除法：{total}÷{people}={per}。',
            f'共有{total}枚，均分{people}人，故以{total}除{people}得{per}。',
            str(per),
        ))

    for i in range(FAMILY_COUNTS['unit_price']):
        price = rng.randint(3, 20)
        qty = rng.randint(2, 10)
        total = price * qty
        aligned.append(build_record(
            f'price-{i:03d}',
            'unit_price',
            split_name(i),
            f'一支钢笔{price}元，买{qty}支需要多少钱？',
            f'一笔值{price}元，购{qty}支，共费几何？',
            f'每支{price}元，买{qty}支，故{price}×{qty}={total}。',
            f'每支钢笔{price}元，买{qty}支，所以总价是{price}×{qty}={total}元。',
            f'一笔值{price}元，购{qty}支，故总费为{price}乘{qty}得{total}。',
            str(total),
        ))

    for i in range(FAMILY_COUNTS['bundle_count']):
        bundle = rng.randint(3, 10)
        num = rng.randint(3, 10)
        total = bundle * num
        aligned.append(build_record(
            f'bundle-{i:03d}',
            'bundle_count',
            split_name(i),
            f'有{total}朵花，每{bundle}朵扎成一束，可以扎几束？',
            f'有花{total}朵，每{bundle}朵为一束，可成几束？',
            f'总花{total}朵，每束{bundle}朵，故{total}÷{bundle}={num}。',
            f'共有{total}朵花，每{bundle}朵扎一束，所以束数是{total}÷{bundle}={num}。',
            f'花共{total}朵，每束用{bundle}朵，故以{total}除{bundle}得{num}束。',
            str(num),
        ))

    for i in range(FAMILY_COUNTS['weight_remaining']):
        kg = rng.randint(1, 5)
        total_g = kg * 1000
        eaten = rng.choice([100, 200, 300, 400, 500, 600, 700, 800])
        if eaten >= total_g:
            eaten = total_g - 100
        remain = total_g - eaten
        aligned.append(build_record(
            f'gram-{i:03d}',
            'weight_remaining',
            split_name(i),
            f'妈妈买了{kg}千克苹果，吃掉了{eaten}克，还剩多少克？',
            f'购苹果{kg}千克，食去{eaten}克，尚余几克？',
            f'{kg}千克即{total_g}克，食去{eaten}克，故{total_g}-{eaten}={remain}。',
            f'{kg}千克先换成{total_g}克，再减去吃掉的{eaten}克，所以还剩{total_g}-{eaten}={remain}克。',
            f'{kg}千克化为{total_g}克，去其{eaten}克，故余{remain}克。',
            str(remain),
        ))

    return aligned


def write_jsonl(path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def main() -> None:
    args = parse_args()
    dataset_tag = resolve_dataset_tag(args)
    interim_out = interim_aligned_file(dataset_tag)
    train_out = dataset_file('train', dataset_tag)
    val_out = dataset_file('val', dataset_tag)
    test_out = dataset_file('test', dataset_tag)
    summary_out = dataset_summary_file(dataset_tag)

    aligned = build_aligned_records()
    all_records = []
    for item in aligned:
        all_records.extend(expand_training_records(item, args.variant, args.think_style))

    splits = {'train': [], 'val': [], 'test': []}
    for rec in all_records:
        splits[rec['split']].append(rec)

    write_jsonl(interim_out, aligned)
    write_jsonl(train_out, splits['train'])
    write_jsonl(val_out, splits['val'])
    write_jsonl(test_out, splits['test'])

    summary = {
        'dataset_tag': dataset_tag,
        'variant': args.variant,
        'think_style': args.think_style,
        'aligned_count': len(aligned),
        'train_records': len(splits['train']),
        'val_records': len(splits['val']),
        'test_records': len(splits['test']),
        'families': sorted({x['family'] for x in aligned}),
        'avg_modern_len': round(mean(len(x['modern_question']) for x in aligned), 2),
        'avg_classical_len': round(mean(len(x['classical_question']) for x in aligned), 2),
        'avg_structured_think_len': round(mean(len(x['structured_think']) for x in aligned), 2),
        'avg_modern_think_len': round(mean(len(x['modern_think']) for x in aligned), 2),
        'avg_classical_think_len': round(mean(len(x['classical_think']) for x in aligned), 2),
        'classical_over_modern_char_ratio': round(
            mean(len(x['classical_question']) / len(x['modern_question']) for x in aligned),
            4,
        ),
    }
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
