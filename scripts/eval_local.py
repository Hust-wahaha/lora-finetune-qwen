from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'runs'
print(f'Evaluation outputs should be stored under {RUNS}')
