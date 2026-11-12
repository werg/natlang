"""Learned-operator cases render to native graphs plus text-cut siblings (data/operator_graph.py)."""
import glob
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT)]
LFM3 = glob.glob(str(Path.home() / '.cache/huggingface/hub/models--LiquidAI--LFM3-1.7B/snapshots/*'))
pytestmark = pytest.mark.skipif(not LFM3, reason='LFM3 tokenizer not cached')


def test_cut_text_matches_the_runtime_cut():
    from natlang_neuralese.data.operator_graph import cut_text
    text = '\n'.join(f'line {index:04d}' for index in range(1000))
    cut = cut_text(text)
    assert cut.startswith('line 0000') and cut.rstrip().endswith('line 0999')
    assert '<<cut off: ' in cut and f'of {len(text)} characters not shown>>' in cut
    assert cut_text('short') == 'short'
