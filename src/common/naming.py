from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from .paths import FINAL_DATA_DIR, INTERIM_DATA_DIR, RUNS_DIR, ensure_run_subdirs

DEFAULT_VISIBLE_DATASET_TAG = 's800'
DEFAULT_THINK_DATASET_TAG = 's800_think'
DEFAULT_MODEL_TAG = 'qwen3.5-0.8b'

LEGACY_DATASET_TAGS = {
    's800',
    's800_think',
}


def slugify_tag(text: str) -> str:
    cleaned = re.sub(r'[^a-zA-Z0-9._-]+', '-', text.strip())
    cleaned = re.sub(r'-{2,}', '-', cleaned).strip('-')
    return cleaned or 'default'


def default_dataset_tag(variant: str) -> str:
    return DEFAULT_THINK_DATASET_TAG if variant == 'think' else DEFAULT_VISIBLE_DATASET_TAG


def is_legacy_dataset_tag(dataset_tag: str) -> bool:
    return dataset_tag in LEGACY_DATASET_TAGS


def dataset_file(split: str, dataset_tag: str) -> Path:
    return FINAL_DATA_DIR / f'{split}_{dataset_tag}.jsonl'


def interim_aligned_file(dataset_tag: str) -> Path:
    return INTERIM_DATA_DIR / f'aligned_{dataset_tag}.jsonl'


def dataset_summary_file(dataset_tag: str) -> Path:
    return FINAL_DATA_DIR / f'dataset_{dataset_tag}_summary.json'


def timestamp() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def make_run_name(stage: str, dataset_tag: str, model_tag: str, suffix: str) -> str:
    return '_'.join([
        timestamp(),
        slugify_tag(stage),
        slugify_tag(dataset_tag),
        slugify_tag(model_tag),
        slugify_tag(suffix),
    ])


def make_run_dir(stage: str, dataset_tag: str, model_tag: str, suffix: str, base_dir: Path | None = None) -> Path:
    run_dir = (base_dir or RUNS_DIR) / make_run_name(stage, dataset_tag, model_tag, suffix)
    return ensure_run_subdirs(run_dir)
