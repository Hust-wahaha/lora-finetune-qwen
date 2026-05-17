from __future__ import annotations

CORE_REQUIRED_FIELDS = {
    'id',
    'source',
    'family',
    'split',
    'modern_question',
    'classical_question',
    'answer',
}

TRAINING_REQUIRED_FIELDS = {
    'messages',
    'view',
    'modern_think',
    'classical_think',
}

STRUCTURED_THINK_FIELDS = ('structured_think', 'structured_cot')

THINK_TAG_PREFIX = '<think>\n'
THINK_ANSWER_BOUNDARY = '\n</think>\n\n答案：'

DATASET_VARIANTS = ('visible', 'think')
THINK_STYLES = ('by_view', 'structured')
SUPPORTED_VIEWS = ('modern', 'classical')


def has_structured_think_field(row: dict) -> bool:
    return any(field in row for field in STRUCTURED_THINK_FIELDS)
