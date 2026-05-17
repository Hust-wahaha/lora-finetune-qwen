from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.common.schema import (
    CORE_REQUIRED_FIELDS,
    STRUCTURED_THINK_FIELDS,
    THINK_ANSWER_BOUNDARY,
    THINK_TAG_PREFIX,
    TRAINING_REQUIRED_FIELDS,
    has_structured_think_field,
)

REQUIRED = CORE_REQUIRED_FIELDS | TRAINING_REQUIRED_FIELDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=Path)
    parser.add_argument('--expect-think-tags', action='store_true')
    return parser.parse_args()


def validate(path: Path, expect_think_tags: bool) -> int:
    bad = 0
    with path.open('r', encoding='utf-8') as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f'[{idx}] invalid json')
                bad += 1
                continue
            keys = set(obj.keys())
            miss = REQUIRED - keys
            if miss:
                print(f'[{idx}] missing {sorted(miss)}')
                bad += 1
            if not has_structured_think_field(obj):
                print(f'[{idx}] missing one of {list(STRUCTURED_THINK_FIELDS)}')
                bad += 1
            msgs = obj.get('messages', [])
            if not isinstance(msgs, list) or len(msgs) != 3:
                print(f'[{idx}] bad messages length')
                bad += 1
                continue
            roles = [m.get('role') for m in msgs]
            if roles != ['system', 'user', 'assistant']:
                print(f'[{idx}] bad roles {roles}')
                bad += 1
                continue
            assistant = msgs[-1].get('content', '')
            if expect_think_tags:
                if not assistant.startswith(THINK_TAG_PREFIX):
                    print(f'[{idx}] missing think prefix')
                    bad += 1
                if THINK_ANSWER_BOUNDARY not in assistant:
                    print(f'[{idx}] bad think-answer boundary')
                    bad += 1
            else:
                if assistant.startswith(THINK_TAG_PREFIX):
                    print(f'[{idx}] unexpected think prefix for visible dataset')
                    bad += 1
    print(f'bad={bad}')
    return bad


if __name__ == '__main__':
    args = parse_args()
    sys.exit(validate(args.path, expect_think_tags=args.expect_think_tags))
