"""The Python servers declare every score-item field natlang's client sends (docs/ENGINE_PARITY.md)."""
import re
from pathlib import Path

from natlang_neuralese.serve.native_learning import SCORE_ITEM_FIELDS

ROOT = Path(__file__).resolve().parents[2]


def test_every_score_item_field_the_client_sends_is_declared_by_the_python_servers():
    source = (ROOT / 'ts-host/src/model/decision-scoring.ts').read_text()
    body = source[source.index('export function batchScoreItem'):]
    body = body[:body.index('\n}\n')]
    sent = set(re.findall(r'\b([a-z_]+): item\.', body)) | set(re.findall(r'\{ ([a-z_]+): ', body))
    assert {'messages', 'continuations', 'cache_messages'} <= sent, sent
    missing = sent - SCORE_ITEM_FIELDS.keys()
    assert not missing, (f'the client sends {sorted(missing)} on score items: serve them in native_learning.py (native '
                         'engine and SGLang actor) or declare them in SCORE_ITEM_FIELDS with a row in docs/ENGINE_PARITY.md')
