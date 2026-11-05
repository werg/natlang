import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('build_core_focus', Path(__file__).parents[1] / 'scripts/neuralese_data/build_core_focus.py')
bcf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bcf)

SCAN = {k: 0 for k in bcf.RX} | {'decisions': 0, 'ext_states': 0}


def row(index, family='natlang', tokens=4000, template=None, scan=None, flags=None, communication=False, depth=1):
    return {'index': index, 'id': f'r{index}', 'family': family, 'tokens': tokens, 'communication': communication,
            'producer_consumer': communication, 'external': False, 'depth': depth, 'fan_in': 0,
            'flags': flags or {}, 'scan': ({**SCAN, **scan} if scan is not None else {}), 'template': template or f'{family}|t{index}'}


def test_classification_uses_own_output_evidence_and_never_selects_the_wrapper():
    assert bcf.classify(row(0, scan={'await_nl': 2, 'finish_true': 1})) == {'semantic', 'typed_finish'}
    assert 'blocked' in bcf.classify(row(1, scan={'status_stop': 1}))
    assert bcf.classify(row(2, family='port-record-wrapper')) == {'capped:wrapper'}
    assert 'neuralese_d2' in bcf.classify(row(3, family='native-communication-rest27', communication=True, depth=3))
    assert 'neuralese_d1' in bcf.classify(row(6, family='native-communication', communication=True, depth=2))
    assert bcf.classify(row(4, scan={})) == {'general'}
    assert bcf.classify(row(5, family='coding-swe')) == {'coding'}


def test_selection_takes_whole_batches_caps_templates_and_never_duplicates():
    rows = [row(i, scan={'iterateOn': 1}, template=f'T{i // 3}') for i in range(30)]          # three rows per template
    rows += [row(30 + i, scan={'iterateOn': 1}, tokens=40000) for i in range(4)]                # over 32k: never
    rows += [row(34 + i, scan={'await_nl': 1}, tokens=20000) for i in range(10)]                # 16k-32k allowance
    units = [tuple(range(i, i + 2)) for i in range(0, 30, 2)] + [(30, 31), (32, 33)] + [(34 + i,) for i in range(10)]
    chosen, chosen_rows, tags, quota, count, avail, excluded, templates = bcf.select(rows, units, 20, 1, 2, 10, 0.10)
    taken = [i for unit in chosen for i in unit]
    assert len(taken) == len(set(taken)) == len(chosen_rows)
    assert all(unit in units for unit in chosen)                                                # whole packed batches only
    assert not set(taken) & {30, 31, 32, 33}                                                    # nothing above 32k tokens
    assert max(templates.values()) <= 2
    assert sum(rows[i]['tokens'] > 16384 for i in taken) <= 2                                   # 10% of 20 rows


def test_scarce_stratum_shortfall_moves_to_its_group_without_duplication():
    rows = [row(i, scan={'iterateOn': 1}) for i in range(40)] + [row(40, flags={'reducer': True})]
    units = [(i,) for i in range(41)]
    chosen, chosen_rows, tags, quota, count, *_ = bcf.select(rows, units, 40, 3, 2, 10, 0.10)
    assert count['reducer'] == 1 and quota['reducer'] == 1                                        # all that exists, frozen
    assert quota['iterate'] > round(bcf.QUOTAS['iterate'] * 40)                                   # same group takes the shortfall
    assert len({i for unit in chosen for i in unit}) == sum(len(unit) for unit in chosen)


def test_mixed_order_interleaves_strata():
    rows = [row(i, scan={'iterateOn': 1}) for i in range(4)] + [row(4 + i, scan={'await_nl': 1}) for i in range(4)]
    tags = {r['index']: bcf.classify(r) for r in rows}
    units = [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,)]
    order = bcf.mixed_order(units, rows, tags)
    assert {next(iter(tags[order[0][0]])), next(iter(tags[order[1][0]]))} == {'iterate', 'semantic'}
