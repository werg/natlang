import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_reports_both_dependency_directions(tmp_path):
    def row(name, reads=(), writes=()):
        return {'id': name, 'split': 'train', 'source_groups': ['one-world'],
                'messages': [{'type': 'read', 'name': value} for value in reads],
                'target': [{'$write': {'name': value, 'source': value}} for value in writes]}
    rows = [row('a', writes=['a']), row('b', writes=['b']),
            row('c', reads=['a', 'b']), row('d', reads=['a']), row('e', reads=['a'])]
    source, out = tmp_path/'rows.jsonl', tmp_path/'audit.json'
    source.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    subprocess.run([sys.executable, str(ROOT/'scripts/audit_neuralese_recurrence.py'),
                    str(source), '--out', str(out)], check=True, capture_output=True)
    audit = json.loads(out.read_text())
    assert audit['schema'] == 'natlang.recurrence-audit/2'
    assert audit['max_producers_per_consumer'] == 2
    assert audit['max_consumers_per_producer'] == 3
    assert audit['linked_edges'] == 4
    assert audit['structurally_closed']
    assert 'max_branching' not in audit
