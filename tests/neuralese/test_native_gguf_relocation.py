from pathlib import Path

import pytest

from natlang_neuralese.export import native_gguf
from natlang_neuralese.model import native_model


def test_export_supplies_relocated_base_to_checkpoint_loader(tmp_path, monkeypatch):
    checkpoint = tmp_path / 'checkpoint.pt'
    checkpoint.write_bytes(b'fixture')
    base = tmp_path / 'base'
    base.mkdir()
    seen = []

    def load(path, **kwargs):
        seen.append((path, kwargs['base']))
        raise RuntimeError('loader reached')

    monkeypatch.setattr(native_model, 'load_native_checkpoint', load)
    with pytest.raises(RuntimeError, match='loader reached'):
        native_gguf.export(checkpoint, tmp_path / 'out', base=base)
    assert seen == [(str(checkpoint.resolve()), base)]
