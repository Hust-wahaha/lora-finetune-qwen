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
# 答案抽取与「整体回答完整」判定共用同一组正则，确保 is_generation_complete 与
# extract_final_answer 永远同口径，杜绝以前那种「文档说严、实现说松」的协议-实现脱节。
#
# 三种合法收尾都接受（按优先级排序）：
#   (1) 训练目标格式 / 微调后正式格式 `…</think>\n\n答案：78。`
#   (2) baseline non-thinking 模板格式 `…</think>\n\n…答案：78` —— 句号常被省略，
#       但「答案：」这个强信号足以确认收尾，不必再要求句号
#   (3) 精简末尾格式 `…</think>\n\n78。`（无前缀，句号强制要求以与"截断在数字处"区分）
#
# 关键约束：
#   - 末端锚定 `\s*$`，所以只看「整段最后那个数字」，<think> 内部出现的 "78美元。"
#     / 中间式子 "= 78。" 都不会被误抽
#   - 数字与句号之间只允许空白，因此 "78美元。" 会被正确拒绝
#   - 句号兼容中文「。」与英文「.」，但保留小数点 ([0-9]+\.[0-9]+)
#   - 「答案：」前缀存在时（分支 1+2）句号可省，否则（分支 3）句号必须存在
_ANSWER_TAIL_PREFIXED = re.compile(
    r'答案[：:]\s*([0-9]+(?:\.[0-9]+)?)\s*[。.]?\s*$'
)
_ANSWER_TAIL_BARE = re.compile(
    r'([0-9]+(?:\.[0-9]+)?)\s*[。.]\s*$'
)


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


def extract_final_answer(text: str | None) -> str | None:
    """从模型整段输出末尾抽取数值答案。

    扫描范围：
      - 含 `<think>` 块时**只看 `</think>` 之后的尾部**——避免 think 体内的中间式子
        （例如 `…即5×0.6=3美元。`）或截断在 think 中的 `=15。` 被误抽成答案
      - 不含 `<think>` 块时（极少见，例如裸题面回答）扫描全文，保留兜底

    优先级：先试「答案：N（句号可省）」，再退回「裸 N。（句号必备）」。
    返回纯数字串（如 "78" / "3.5"），抽不到返回 None。
    """
    if not text:
        return None
    if THINK_OPEN in text:
        close_idx = text.find(THINK_CLOSE)
        if close_idx < 0:
            # think 已开但没闭合 = 被截断在推理中段，不算给出最终答案
            return None
        tail = text[close_idx + len(THINK_CLOSE):]
    else:
        tail = text
    match = _ANSWER_TAIL_PREFIXED.search(tail)
    if match:
        return match.group(1)
    match = _ANSWER_TAIL_BARE.search(tail)
    return match.group(1) if match else None


def is_generation_complete(text: str | None) -> bool:
    if not is_cot_complete(text):
        return False
    return extract_final_answer(text) is not None


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
