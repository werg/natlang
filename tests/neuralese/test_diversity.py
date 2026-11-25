"""The shared diversity mechanism (natlang_neuralese.common.diversity) and the stages that use it: shape merge, target
near duplicates, shape caps of a selection and of an aux part, the mix report and its regression flags, the fill
order, the round health's mix flags (DECISIONS 2026-11-25 "diversity end to end")."""
import collections
import json
import random
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'training/neuralese'), str(REPO / 'scripts/neuralese_data'), str(REPO / 'scripts/rl_tasks')]

from natlang_neuralese.common import diversity as dv  # noqa: E402


def row(task_id, body, root='solve.nl', extra=None):
    files = {root: f'---\nargs: "{{ q: string }}"\nreturns: "string"\n---\n{body}\n', **(extra or {})}
    return {'id': task_id, 'semantics': {'root': root, 'files': files}, 'curriculum': {'mode': 'program'}}


TEMPLATE = ('Find every function or method whose own body calls something named {w} and return them as strings with '
            'the path relative to the codebase root, reading each candidate file once and listing the calls you saw.')


def test_task_shapes_merge_a_template_split_by_a_lowercase_item_word_and_keep_distinct_templates():
    rows = [row(f'a{i}', TEMPLATE.format(w=w)) for i, w in enumerate(['estimate', 'best', 'compile', 'status'])]
    other = row('b', 'Summarize the attached report in three sentences and name the single most important risk it '
                     'describes, quoting the sentence that states it and saying who owns the follow-up.')
    base = dv.task_shapes([*rows, other], merge=None)
    assert len({base[r['id']] for r in rows}) == 4                  # the base key splits the template per item word
    merged = dv.task_shapes([*rows, other])
    assert len({merged[r['id']] for r in rows}) == 1                # one template
    assert merged['b'] != merged['a0']
    # another file skeleton is another program, however alike its prose
    helper = row('c', TEMPLATE.format(w='estimate'), extra={'solve/check.nl': 'Check one file.'})
    assert dv.task_shapes([*rows, helper])['c'] != merged['a0']
    assert dv.shape_lookup(merged)('missing') == 'id:missing'


def test_token_signature_finds_near_duplicate_targets_and_short_targets_have_none():
    rng = np.random.RandomState(0)
    a = rng.randint(0, 60000, size=400)
    b = a.copy()
    b[200] = 7                                                     # one token changed
    c = rng.randint(0, 60000, size=400)
    index = dv.NearDuplicates(0.8)
    assert not index.seen(dv.token_signature(a))
    assert index.seen(dv.token_signature(b))
    assert not index.seen(dv.token_signature(c))
    chain = {'segs': [(0, 3, {'sample': np.array([1, 2, 3])})]}
    assert dv.unit_signature([chain]) is None                       # below NEAR_DUPLICATE_MIN_TOKENS


def chain(part, task, round_, ids):
    return {'part': part, 'task': task, 'round': round_, 'ids': list(ids),
            'segs': [(0, len(ids), {'sample': np.asarray(ids)})]}


def test_drop_near_duplicate_units_keeps_the_better_sources_copy():
    rng = np.random.RandomState(1)
    shared = rng.randint(0, 60000, size=300)
    chains = [chain('operators', 'o1', 'r', shared), chain('rl-own', 't1', 'k1', shared),
              chain('teacher', 't2', 'k2', rng.randint(0, 60000, size=300))]
    priority = {'rl-own': 0, 'teacher': 1}
    kept, report = dv.drop_near_duplicate_units(chains, 0.8, lambda p: priority.get(p, 9))
    assert [c['part'] for c in kept] == ['rl-own', 'teacher']
    assert report['dropped_units'] == 1 and report['parts'] == {'operators': 1}
    assert dv.drop_near_duplicate_units(chains, 0)[0] == chains      # opt-out


def test_shape_quota_and_shape_capped():
    q = dv.ShapeQuota(1000, 0.01, floor=0)
    assert q.allow('s', 6) and q.allow('s', 4) and not q.allow('s', 1) and q.allow('t', 50)   # first item always fits
    items = [('x', i) for i in range(30)] + [('y', i) for i in range(5)]
    kept = dv.shape_capped(items, lambda item: item[0], share=0.1, floor=4)
    assert collections.Counter(k for k, _ in kept) == {'x': 4, 'y': 4}
    assert dv.shape_capped(items, lambda item: item[0], share=0) == items


def test_diversity_report_and_regression_flags():
    rng = np.random.RandomState(2)
    dup = dv.token_signature(rng.randint(0, 60000, size=200))
    units = [{'part': 'teacher', 'shape': f's{i}', 'mass': 10., 'signature': dv.token_signature(rng.randint(0, 60000, 200))}
             for i in range(20)]
    units += [{'part': 'operators', 'shape': 'one', 'mass': 10., 'signature': dup} for _ in range(4)]
    report = dv.diversity_report(units)
    teacher, ops = report['parts']['teacher'], report['parts']['operators']
    assert teacher['shapes'] == 20 and abs(teacher['effective_shapes'] - 20) < 0.1 and teacher['near_duplicate_share'] == 0
    assert ops['top_shape_share'] == 1.0 and ops['near_duplicate_share'] == 0.75
    flags = dv.regression_flags(report, {'parts': {'teacher': {'effective_shapes': 40., 'top_shape_share': 0.05}}})
    kinds = {(f['flag'], f['part']) for f in flags}
    assert ('mix_effective_shapes_fell', 'teacher') in kinds and ('mix_near_duplicates', 'operators') in kinds
    assert not [f for f in dv.regression_flags(report, report) if f['part'] == 'teacher']


def test_capped_holds_one_shape_spanning_many_families():
    import sft_sources as s
    candidates = [{'key': f'k{i}', 'task_id': f't{i}', 'family': f'f{i % 50}', 'source': 'teacher', 'teacher': 'm'}
                  for i in range(200)]
    candidates += [{'key': f'u{i}', 'task_id': f'u{i}', 'family': f'g{i}', 'source': 'teacher', 'teacher': 'm'}
                   for i in range(100)]
    shape_of = lambda task_id: 'wrapper' if task_id.startswith('t') else task_id   # noqa: E731
    funnel = collections.defaultdict(collections.Counter)
    out = s.capped(candidates, random.Random(0), funnel, lambda t: 4, 1.0, lambda c: 0, shape_of=shape_of,
                   shape_share=0.02)
    assert sum(c['task_id'].startswith('t') for c in out) == 6       # 2% of 300
    assert sum(c['task_id'].startswith('u') for c in out) == 100
    assert funnel['teacher']['over_shape_cap'] == 194
    same = s.capped(candidates, random.Random(0), collections.defaultdict(collections.Counter), lambda t: 4, 1.0,
                    lambda c: 0)
    assert len(same) == 300                                        # no shape_of: the old selection


def test_fill_order_is_breadth_first_over_tasks_and_shape_capped():
    import filled_reasoning as fr
    episodes = [{'key': f'teacher:{t}:{j}', 'work': 'w', 'task_id': f't{t}', 'shape': 'big' if t < 40 else f's{t}'}
                for t in range(60) for j in range(2)]
    order = fr.ordered(episodes, [], None, seed=3, shape_cap=0.05)
    head = collections.Counter(e['shape'] for e in order[:40])
    assert head['big'] <= max(4, 0.05 * 40)
    assert len(order) == len(episodes)
    # opt-out: the shuffled order alone (one task's two episodes may follow each other)
    plain = fr.ordered(episodes, [], None, seed=3, shape_cap=0)
    assert sorted(e['key'] for e in plain) == sorted(e['key'] for e in episodes) and plain != order
    tasks_seen, positions = set(), []
    for index, e in enumerate(fr.ordered([e for e in episodes if e['shape'] != 'big'], [], None, 3, 0.5)):
        if e['task_id'] in tasks_seen:
            positions.append(index)
        tasks_seen.add(e['task_id'])
    assert min(positions) >= 20                                    # every task once before any twice


def test_health_adds_the_mix_diversity_flags(tmp_path):
    import collection_health
    build = tmp_path / 'build.json'
    build.write_text(json.dumps({'mix': {'diversity': {'all': {'units': 3, 'effective_shapes': 2.0},
                                                        'parts': {'teacher': {'units': 3, 'effective_shapes': 2.0}}},
                                         'diversity_flags': [{'flag': 'mix_near_duplicates', 'part': 'teacher',
                                                              'share': 0.2}]}}))
    health = collection_health.add_mix_diversity({'flags': [{'flag': 'no_episodes'}]}, build)
    assert [f['flag'] for f in health['flags']] == ['no_episodes', 'mix_near_duplicates']
    assert health['mix_diversity']['parts']['teacher']['effective_shapes'] == 2.0
    assert collection_health.add_mix_diversity({'flags': []}, tmp_path / 'missing.json') == {'flags': []}


def test_a_neuralese_variant_shares_its_originals_per_task_cap():
    import sft_sources as s
    candidates = [{'key': f'o{i}', 'task_id': 't', 'item': 't', 'family': 'f', 'source': 'rl-own', 'teacher': ''}
                  for i in range(3)]
    candidates += [{'key': f'v{i}', 'task_id': 't:nv', 'item': 't', 'family': 'f', 'source': 'rl-own', 'teacher': ''}
                   for i in range(3)]
    out = s.capped(candidates, random.Random(0), collections.defaultdict(collections.Counter), lambda t: 4, 1.0,
                   lambda c: 0)
    assert len(out) == 4                                           # one item, one cap (was 4 + 4 under two ids)


def test_the_round_build_takes_the_drivers_build_args():
    import expert_iteration as ei
    command = ei.mix_build_command('/r', work='/w', out='/o', frontier='/f', aux=[], prefer_unseen=[], teacher_budget=1,
                                   own_share=0.35, extra='--max-shape-share 0 --mix-near-duplicates keep'.split())
    assert command[-4:] == ['--max-shape-share', '0', '--mix-near-duplicates', 'keep']
    assert '--max-shape-share' not in ei.mix_build_command('/r', work='/w', out='/o', frontier='/f', aux=[],
                                                           prefer_unseen=[], teacher_budget=1, own_share=0.35)


def test_an_aux_share_is_a_share_of_the_mix_as_admitted():
    """aei1 r0009: operators at share 0.05 carried 23% of the loss, its budget computed from every loaded filled unit
    (15,480) though 470 trained. ``trim_shares`` holds each part to its share of the mix as it is."""
    import sft_sources as s

    def unit(part, i, mass):
        return {'part': part, 'task': f'{part}{i}', 'round': 'r', 'mass': mass}
    chains = [unit('rl-own', i, 10.) for i in range(10)] + [unit('operators', i, 10.) for i in range(10)]
    kept, report = s.trim_shares(chains, {'operators': 0.05}, mass_of=lambda c: c['mass'])
    ops = [c for c in kept if c['part'] == 'operators']
    assert sum(c['mass'] for c in ops) <= 0.05 * sum(c['mass'] for c in kept) + 1e-6
    assert [c['task'] for c in ops] == [] or ops[0]['task'] == 'operators0'      # the first-admitted units stay
    assert report['dropped_units']['operators'] == 10 - len(ops)
    assert len([c for c in kept if c['part'] == 'rl-own']) == 10
    big = [unit('rl-own', i, 10.) for i in range(100)] + [unit('operators', i, 10.) for i in range(10)]
    kept, _ = s.trim_shares(big, {'operators': 0.05}, mass_of=lambda c: c['mass'])
    assert len([c for c in kept if c['part'] == 'operators']) == 5               # 50 / 1050 <= 5%
    assert s.trim_shares(big, {}, mass_of=lambda c: c['mass'])[0] == big
