import importlib.util
import json
from pathlib import Path


spec = importlib.util.spec_from_file_location('natlang_coord', Path(__file__).parents[1] / 'scripts/coord.py')
coord = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coord)


def test_legacy_utc_inbox_import_preserves_messages_and_original(tmp_path, monkeypatch):
    monkeypatch.setenv('COORD_MACHINE', 'pop')
    store = coord.Store(tmp_path)
    original = '## 2026-10-08T13:25:30Z — dgx-agent\n\nKeep the shared trainer.\n'
    (store.root / 'inbox.md').write_text(original)
    (store.root / 'seen.json').write_text(json.dumps({'offset': 0}))
    assert store.migrate_legacy() == 1
    messages = store.messages()
    assert len(messages) == 1
    assert messages[0]['sent_at'] == '2026-10-08T13:25:30.000000Z'
    assert messages[0]['body'] == 'Keep the shared trainer.'
    assert messages[0]['legacy'] is False
    assert next((store.root / 'archive').glob('legacy-inbox-*.md')).read_text() == original
    assert store.migrate_legacy() == 0
    assert len(store.messages()) == 1
