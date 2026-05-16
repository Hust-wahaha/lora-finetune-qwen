from __future__ import annotations

import json
import random
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
INTERIM_OUT = ROOT / 'data' / 'interim' / 'aligned_s800.jsonl'
TRAIN_OUT = ROOT / 'data' / 'final' / 'train_s800.jsonl'
VAL_OUT = ROOT / 'data' / 'final' / 'val_s800.jsonl'
TEST_OUT = ROOT / 'data' / 'final' / 'test_s800.jsonl'
SUMMARY_OUT = ROOT / 'data' / 'final' / 'dataset_s800_summary.json'
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


def split_name(i: int) -> str:
    mod = i % 10
    if mod < 8:
        return 'train'
    if mod == 8:
        return 'val'
    return 'test'


def build_record(base_id: str, family: str, split: str, modern: str, classical: str, cot: str, answer: str):
    return {
        'id': base_id,
        'source': 'synthetic-template-s800',
        'family': family,
        'split': split,
        'modern_question': modern,
        'classical_question': classical,
        'structured_cot': cot,
        'answer': answer,
    }


def render_messages(user_text: str, cot: str, answer: str):
    return [
        {'role': 'system', 'content': SYSTEM},
        {'role': 'user', 'content': user_text},
        {'role': 'assistant', 'content': f'{cot} 答案：{answer}。'},
    ]


def expand_training_records(item):
    common = {
        'base_id': item['id'],
        'source': item['source'],
        'family': item['family'],
        'split': item['split'],
        'modern_question': item['modern_question'],
        'classical_question': item['classical_question'],
        'structured_cot': item['structured_cot'],
        'answer': item['answer'],
    }
    return [
        {**common, 'id': f"{item['id']}-modern", 'view': 'modern', 'messages': render_messages(item['modern_question'], item['structured_cot'], item['answer'])},
        {**common, 'id': f"{item['id']}-classical", 'view': 'classical', 'messages': render_messages(item['classical_question'], item['structured_cot'], item['answer'])},
    ]


aligned = []

for i in range(FAMILY_COUNTS['give_away']):
    a = rng.randint(12, 60)
    b = rng.randint(3, max(3, a // 2))
    aligned.append(build_record(
        f'give-{i:03d}', 'give_away', split_name(i),
        f'小明有{a}颗糖，送给小红{b}颗，还剩几颗？',
        f'明有糖{a}颗，遗红{b}颗，尚余几何？',
        f'初有{a}颗，送出{b}颗，故{a}-{b}={a-b}。',
        str(a - b),
    ))

for i in range(FAMILY_COUNTS['box_total']):
    boxes = rng.randint(2, 12)
    each = rng.randint(4, 15)
    aligned.append(build_record(
        f'box-{i:03d}', 'box_total', split_name(i),
        f'一盒铅笔有{each}支，{boxes}盒一共有多少支？',
        f'一匣铅笔{each}支，{boxes}匣共几支？',
        f'每匣{each}支，{boxes}匣则{each}×{boxes}={each*boxes}。',
        str(each * boxes),
    ))

for i in range(FAMILY_COUNTS['equal_division']):
    people = rng.randint(2, 12)
    per = rng.randint(3, 15)
    total = people * per
    aligned.append(build_record(
        f'div-{i:03d}', 'equal_division', split_name(i),
        f'把{total}个苹果平均分给{people}个小朋友，每人分几个？',
        f'有苹果{total}个，均分{people}人，人得几何？',
        f'总数{total}，分与{people}人，故{total}÷{people}={per}。',
        str(per),
    ))

for i in range(FAMILY_COUNTS['unit_price']):
    price = rng.randint(3, 20)
    qty = rng.randint(2, 10)
    total = price * qty
    aligned.append(build_record(
        f'price-{i:03d}', 'unit_price', split_name(i),
        f'一支钢笔{price}元，买{qty}支需要多少钱？',
        f'一笔值{price}元，购{qty}支，共费几何？',
        f'每支{price}元，买{qty}支，故{price}×{qty}={total}。',
        str(total),
    ))

for i in range(FAMILY_COUNTS['bundle_count']):
    bundle = rng.randint(3, 10)
    num = rng.randint(3, 10)
    total = bundle * num
    aligned.append(build_record(
        f'bundle-{i:03d}', 'bundle_count', split_name(i),
        f'有{total}朵花，每{bundle}朵扎成一束，可以扎几束？',
        f'有花{total}朵，每{bundle}朵为一束，可成几束？',
        f'总花{total}朵，每束{bundle}朵，故{total}÷{bundle}={num}。',
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
        f'gram-{i:03d}', 'weight_remaining', split_name(i),
        f'妈妈买了{kg}千克苹果，吃掉了{eaten}克，还剩多少克？',
        f'购苹果{kg}千克，食去{eaten}克，尚余几克？',
        f'{kg}千克即{total_g}克，食去{eaten}克，故{total_g}-{eaten}={remain}。',
        str(remain),
    ))

all_records = []
for item in aligned:
    all_records.extend(expand_training_records(item))

splits = {'train': [], 'val': [], 'test': []}
for rec in all_records:
    splits[rec['split']].append(rec)

for path in [INTERIM_OUT, TRAIN_OUT, VAL_OUT, TEST_OUT]:
    path.parent.mkdir(parents=True, exist_ok=True)

with INTERIM_OUT.open('w', encoding='utf-8') as f:
    for item in aligned:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

for split, path in [('train', TRAIN_OUT), ('val', VAL_OUT), ('test', TEST_OUT)]:
    with path.open('w', encoding='utf-8') as f:
        for rec in splits[split]:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

summary = {
    'aligned_count': len(aligned),
    'train_records': len(splits['train']),
    'val_records': len(splits['val']),
    'test_records': len(splits['test']),
    'families': sorted({x['family'] for x in aligned}),
    'avg_modern_len': round(mean(len(x['modern_question']) for x in aligned), 2),
    'avg_classical_len': round(mean(len(x['classical_question']) for x in aligned), 2),
    'avg_structured_cot_len': round(mean(len(x['structured_cot']) for x in aligned), 2),
    'classical_over_modern_char_ratio': round(mean(len(x['classical_question']) / len(x['modern_question']) for x in aligned), 4),
}
SUMMARY_OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2))
