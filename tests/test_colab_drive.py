import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_ship_file_falls_back_to_the_tunnel(monkeypatch):
    sys.path[:0] = [str(ROOT), str(ROOT / 'training' / 'neuralese'), str(ROOT / 'scripts' / 'neuralese_data')]
    import pop_rl_actor as p
    from scripts import drive_tier

    class Broken:
        def lookup(self, key):
            raise OSError('no network')
    monkeypatch.setattr(drive_tier, 'Drive', lambda *a, **k: Broken())
    sent = []
    monkeypatch.setattr(p, 'sync_to_remote', lambda local, remote, host: sent.append((local, remote, host)))
    assert p.ship_file('/x/chains.pkl', '/vm/chains.pkl', 'colab-g4', {'profile': 'colab'}, 'ab' * 32) == 'tunnel'
    assert sent == [('/x/chains.pkl', '/vm/chains.pkl', 'colab-g4')]
    monkeypatch.setenv('NATLANG_COLAB_DRIVE', '0')
    assert p.ship_file('/x/c', '/vm/c', 'colab-g4', {'profile': 'colab'}, 'ab' * 32) == 'tunnel'
