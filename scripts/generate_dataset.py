from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / 'data' / 'final' / 'pilot.jsonl'
SYSTEM = '你是一个擅长中文数学应用题的助手，要求输出简洁、准确、可验证。'

SAMPLES = [
    {
        'id': 'pilot-0001',
        'source': 'manual',
        'messages': [
            {'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': '小明有12颗糖，送给小红5颗，还剩几颗？'},
            {'role': 'assistant', 'content': '已知原有12颗，送出5颗，剩余12-5=7颗。答案：7。'},
        ],
        'modern_question': '小明有12颗糖，送给小红5颗，还剩几颗？',
        'classical_question': '小明有糖十二颗，遗红五颗，尚余几何？',
        'structured_cot': '已知初有12，送出5，故12-5=7。',
        'answer': '7',
    },
    {
        'id': 'pilot-0002',
        'source': 'manual',
        'messages': [
            {'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': '一盒铅笔有8支，3盒一共有多少支？'},
            {'role': 'assistant', 'content': '每盒8支，3盒就是8×3=24支。答案：24。'},
        ],
        'modern_question': '一盒铅笔有8支，3盒一共有多少支？',
        'classical_question': '一匣铅笔八支，三匣共几支？',
        'structured_cot': '每匣8支，3匣则8×3=24。',
        'answer': '24',
    },
]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('w', encoding='utf-8') as f:
        for obj in SAMPLES:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')
    print(OUT)


if __name__ == '__main__':
    main()
