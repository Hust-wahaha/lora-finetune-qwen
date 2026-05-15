from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / 'data' / 'raw'
DATA_INTERIM = ROOT / 'data' / 'interim'
DATA_FINAL = ROOT / 'data' / 'final'
SCRIPTS = ROOT / 'scripts'
RUNS = ROOT / 'runs'
ARTIFACTS = ROOT / 'artifacts'
