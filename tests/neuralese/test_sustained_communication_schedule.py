from collections import Counter

from natlang_neuralese.train.native_diversity import sustained_communication_interleave


def test_sustains_communication_and_preserves_batches_on_resume():
    entries = ([{'id': f'c{i}', 'communication': True} for i in range(10)]
               + [{'id': f'u{i}', 'training_priority': 1} for i in range(80)]
               + [{'id': f'd{i}'} for i in range(20)])
    batches = [(v, [i], False) for v in range(7) for i in range(10)]
    batches += [(0, [i], False) for i in range(10, 110)]
    for remaining in (batches, batches[15:]):
        result = sustained_communication_interleave(entries, remaining, schedule_offset=52)
        assert Counter((v, tuple(i)) for v, i, _ in result) == Counter((v, tuple(i)) for v, i, _ in remaining)
        width = 20
        for window in (result[:width], result[len(result)//2-width//2:len(result)//2+width//2], result[-width:]):
            assert any(entries[i[0]].get('communication') for _, i, _ in window)
            assert any(not entries[i[0]].get('communication') for _, i, _ in window)
        for visit in {v for v, _, _ in result}:
            positions = [n for n, (v, _, _) in enumerate(result) if v == visit]
            assert [n for n in positions if result[n][2]] == [positions[-1]]
