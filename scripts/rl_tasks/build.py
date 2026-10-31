"""Build one external source's RL cases: python -I scripts/rl_tasks/build.py SOURCE OUT_DIR [key=value ...]

Writes OUT_DIR/SOURCE/raw-cases.jsonl (all splits; verification and custody run afterwards)."""
import importlib
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.rl_tasks.common import jsonl_write  # noqa: E402

REGISTRY = {}  # source name -> (module, function)
for line in (Path(__file__).parent / 'SOURCES').read_text().split('\n'):
    if line.strip() and not line.startswith('#'):
        name, target = line.split()
        REGISTRY[name] = target


def main():
    name, out = sys.argv[1], Path(sys.argv[2])
    kwargs = {}
    for item in sys.argv[3:]:
        key, value = item.split('=', 1)
        kwargs[key] = int(value) if value.isdigit() else value
    module, function = REGISTRY[name].rsplit(':', 1)
    cases = getattr(importlib.import_module(f'scripts.rl_tasks.sources.{module}'), function)(**kwargs)
    ids = [c['id'] for c in cases]
    assert len(ids) == len(set(ids)), 'duplicate case ids'
    (out / name).mkdir(parents=True, exist_ok=True)
    jsonl_write(out / name / 'raw-cases.jsonl', cases)
    print(json.dumps({'source': name, 'cases': len(cases), 'splits': Counter(c['split'] for c in cases),
                      'shapes': len({c['curriculum']['shape'] for c in cases})}))


if __name__ == '__main__':
    main()
