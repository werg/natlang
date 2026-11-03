"""Lexical flags and eval-span extraction of scripts/neuralese_data/audit_eval_import_targets.py."""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'training' / 'neuralese'))

from neuralese_data import audit_eval_import_targets as audit  # noqa: E402


def test_eval_spans_find_each_code_argument():
    header = list(audit.HEADER)
    ids = [1, 2, *header, 10, 11, audit.PARAM_CLOSE, 3, *header, 12, audit.PARAM_CLOSE, audit.FN_OPEN, 5, 6]
    assert audit.eval_spans(ids) == [(8, 10), (18, 19)]
    # another tool's call and an unterminated eval are not eval spans / run to the end
    assert audit.eval_spans([audit.FN_OPEN, 7, 703, 1]) == []
    assert audit.eval_spans([*header, 4, 5]) == [(6, 8)]


def test_classify_separates_foreign_code_from_valid_typescript():
    assert 'start-python-import' in audit.classify('import os, json\nx = 1')
    assert 'start-python-import' in audit.classify('from pathlib import Path\nprint(1)')
    assert audit.classify("import { a } from 'pkg';\na.length") == ['start-js-import']
    assert audit.classify('import type { T } from "./types";\nconst x = 1;') == ['start-js-import']
    assert 'start-prose' in audit.classify("Let's just call nl on every ticket here.\nconst x = 1;")
    assert 'py-def' in audit.classify('def f(x):\n    return x')
    assert 'py-block' in audit.classify('for item in items:\n    pass')
    assert audit.classify('const x = 1;\nconsole.log(x);\nx') == []
    assert audit.classify('return await nl`Is the ticket resolved?`;') == []


def test_audit_leaf_counts_supervised_and_context(tmp_path):
    import json

    class Tokens:
        """Each token id is one character code; ids below 32 are markers used by the test."""
        def __len__(self):
            return 10 ** 6

        def decode(self, ids):
            return ''.join(chr(i) for i in ids)

    code = 'import os\nprint(1)'
    ids = [*audit.HEADER, *map(ord, code), audit.PARAM_CLOSE, *audit.HEADER, *map(ord, 'const x = 1;'), audit.PARAM_CLOSE]
    first = len(audit.HEADER) + len(code)
    weights = [1.0] * (first + 1) + [0.0] * (len(ids) - first - 1)
    row = {'id': 'r1', 'nodes': [{'name': 'n', 'token_ids': ids, 'target_weights': weights}]}
    leaf = tmp_path / 'leaf.jsonl'
    leaf.write_text(json.dumps(row, separators=(',', ':')) + '\n' + json.dumps({'id': 'r2', 'nodes': []}) + '\n')

    class Compiler:
        def __call__(self, code):
            return {'ok': False, 'diagnostics': [{'code': 'typescript-syntax', 'message': 'x'}]}

    events = []
    totals = audit.audit_leaf(leaf, Tokens(), Compiler(), events.append)
    assert totals['rows'] == 2 and totals['rows_with_eval'] == 1
    assert totals['eval_supervised'] == 1 and totals['eval_context'] == 1
    assert [e['kind'] for e in events] == ['supervised'] and events[0]['syntax_fail'] is True
    assert 'start-python-import' in events[0]['flags']
    assert Counter(totals)['flagged_syntax_fail_supervised'] == 1
