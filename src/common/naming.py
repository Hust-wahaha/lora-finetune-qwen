from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from .paths import FINAL_DATA_DIR, INTERIM_DATA_DIR, RUNS_DIR, ensure_run_subdirs

DEFAULT_VISIBLE_DATASET_TAG = 's800'
DEFAULT_THINK_DATASET_TAG = 's800_think'
DEFAULT_MODEL_TAG = 'qwen3.5-0.8b'

MODEL_TAG_ALIASES = {
    'Qwen/Qwen3.5-0.8B': 'qwen3.5-0.8b',
    'Qwen/Qwen3.5-1.7B': 'qwen3.5-1.7b',
    'Qwen/Qwen3.5-2B': 'qwen3.5-2b',
    'Qwen/Qwen3.5-4B': 'qwen3.5-4b',
    'Qwen/Qwen2.5-0.5B-Instruct': 'qwen2.5-0.5b-instruct',
    'Qwen/Qwen2.5-1.5B-Instruct': 'qwen2.5-1.5b-instruct',
    'Qwen/Qwen2.5-3B-Instruct': 'qwen2.5-3b-instruct',
}

LEGACY_DATASET_TAGS = {
    's800',
    's800_think',
}

LEGACY_RUN_DATASET_ALIASES = {
    's800': 'synth_structuredthink_800_v1',
    's800_think': 'synth_think_800_v1',
}


def slugify_tag(text: str) -> str:
    cleaned = re.sub(r'[^a-zA-Z0-9._-]+', '-', text.strip())
    cleaned = re.sub(r'-{2,}', '-', cleaned).strip('-')
    return cleaned or 'default'


def default_dataset_tag(variant: str) -> str:
    return DEFAULT_THINK_DATASET_TAG if variant == 'think' else DEFAULT_VISIBLE_DATASET_TAG


def is_legacy_dataset_tag(dataset_tag: str) -> bool:
    return dataset_tag in LEGACY_DATASET_TAGS


def dataset_tag_for_run(dataset_tag: str) -> str:
    # Keep on-disk data file names backward-compatible, but expose clearer dataset
    # semantics in run directory names so later experiments remain readable.
    return LEGACY_RUN_DATASET_ALIASES.get(dataset_tag, dataset_tag)


def normalize_model_tag(model_id: str, model_tag: str | None = None) -> str:
    if model_tag:
        return slugify_tag(model_tag)
    return MODEL_TAG_ALIASES.get(model_id, slugify_tag(model_id.split('/')[-1]))


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
        slugify_tag(dataset_tag_for_run(dataset_tag)),
        slugify_tag(model_tag),
        slugify_tag(suffix),
    ])


def make_run_dir(stage: str, dataset_tag: str, model_tag: str, suffix: str, base_dir: Path | None = None) -> Path:
    run_dir = (base_dir or RUNS_DIR) / make_run_name(stage, dataset_tag, model_tag, suffix)
    return ensure_run_subdirs(run_dir)
