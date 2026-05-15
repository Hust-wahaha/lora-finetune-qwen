from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'runs'


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = RUNS / tag
    for sub in ['logs', 'checkpoints', 'predictions', 'metrics', 'samples']:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    print(run_dir)


if __name__ == '__main__':
    main()
