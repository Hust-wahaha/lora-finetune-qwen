from __future__ import annotations

import json
import sys
from pathlib import Path

REQUIRED = {'id', 'source', 'messages', 'modern_question', 'classical_question', 'structured_cot', 'answer', 'view', 'family', 'split'}


def validate(path: Path) -> int:
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
            miss = REQUIRED - obj.keys()
            if miss:
                print(f'[{idx}] missing {sorted(miss)}')
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
    print(f'bad={bad}')
    return bad


if __name__ == '__main__':
    sys.exit(validate(Path(sys.argv[1])))
