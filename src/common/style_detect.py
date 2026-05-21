"""
think 风格识别与生成完整性判定的纯函数集合。

用于 `scripts/eval_five_metrics.py` 与后续可能的 inspect / 可视化脚本，集中维护
词表与正则；调用方不应在 `scripts/` 里自己再拼这些字符串。

词表初版基于 `data/final/test_s800_think.jsonl` 中 `modern_think` / `classical_think`
两个字段的高频差异词。改词表时记得跑一遍 `--self-check` 验证。
"""

from __future__ import annotations

import re

from .schema import THINK_TAG_PREFIX, THINK_ANSWER_BOUNDARY

THINK_OPEN = THINK_TAG_PREFIX.strip()
THINK_CLOSE = THINK_ANSWER_BOUNDARY.split('答案')[0].strip()
ANSWER_MARKER = '答案'

MODERN_KEYWORDS: tuple[str, ...] = (
    '原来', '所以', '用减法', '用加法', '用乘法', '用除法',
    '总共', '一共', '还剩', '每个', '平均分', '因为', '一支',
)

CLASSICAL_KEYWORDS: tuple[str, ...] = (
    '初', '故', '尚', '余', '几何', '遗', '凡', '焉', '矣', '共费',
)

_ENGLISH_RUN = re.compile(r'[A-Za-z]{4,}')
_THINK_BLOCK = re.compile(r'<think>\s*(.*?)\s*</think>', re.DOTALL)


def extract_think_block(text: str | None) -> str | None:
    if not text:
        return None
    match = _THINK_BLOCK.search(text)
    return match.group(1) if match else None


def is_cot_complete(text: str | None) -> bool:
    if not text:
        return False
    open_idx = text.find(THINK_OPEN)
    close_idx = text.find(THINK_CLOSE)
    return open_idx >= 0 and close_idx > open_idx


def is_generation_complete(text: str | None) -> bool:
    if not is_cot_complete(text):
        return False
    assert text is not None
    if ANSWER_MARKER not in text:
        return False
    return text.rstrip().endswith('。')


def _count_hits(text: str, keywords: tuple[str, ...]) -> int:
    return sum(1 for kw in keywords if kw in text)


def style_scores(cot: str | None) -> dict[str, int]:
    if not cot:
        return {'modern_hits': 0, 'classical_hits': 0}
    return {
        'modern_hits': _count_hits(cot, MODERN_KEYWORDS),
        'classical_hits': _count_hits(cot, CLASSICAL_KEYWORDS),
    }


def style_label(cot: str | None) -> str:
    scores = style_scores(cot)
    m, c = scores['modern_hits'], scores['classical_hits']
    if m == 0 and c == 0:
        return 'unknown'
    if m > 0 and c > 0:
        return 'mixed' if abs(m - c) <= 1 else ('modern' if m > c else 'classical')
    return 'modern' if m > 0 else 'classical'


def english_leak_count(cot: str | None) -> int:
    if not cot:
        return 0
    return len(_ENGLISH_RUN.findall(cot))


def has_english_leak(cot: str | None) -> bool:
    return english_leak_count(cot) > 0
