#!/usr/bin/env python3
"""Collection-only RL pilot: select, roll out and score with the real controller code, never update.

  rl_collection_pilot.py --config CONTROL.json --out DIR [--rounds N] [--no-teacher]

CONTROL is a prepared interleaved RL control (prepare_interleaved_rl.py). Each pilot round uses the controller's own
selection (task_curriculum, carried selector history plus the pilot's own earlier rounds), its exact
prepare_round_collection/collect_with_actor path and the bound actor, writing everything below DIR. No learner,
no optimizer update, no publication, no write to the control's own ``out``. Later rounds select from the earlier
rounds' scored outcomes (same policy), which shows how the selector adapts difficulty; they are not policy learning.
Reports per round: groups with outcome spread, per-family pass rates and the selected levels against each family's
practice level.
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import sys
from pathlib import Path


def family_report(selected, groups):
    from natlang_neuralese.rl.curriculum import metadata
    by_id = {row['id']: row for row in selected}
    families = collections.defaultdict(lambda: {'tasks': 0, 'scored': 0, 'successes': 0., 'spread': 0, 'levels': []})
    for group in groups:
        row = by_id.get(group['task_id'])
        if row is None:
            continue
        meta = metadata(row)
        entry = families[meta['family']]
        entry['tasks'] += 1
        entry['levels'].append(meta['level'])
        if group.get('scored'):
            entry['scored'] += group['scored']
            entry['successes'] += group['mean'] * group['scored']
            entry['spread'] += bool(group.get('spread'))
    return {family: {**value, 'pass_rate': round(value['successes'] / value['scored'], 3) if value['scored'] else None}
            for family, value in sorted(families.items())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--no-teacher', action='store_true', help='skip JOLT+ privileged teacher attempts')
    args = parser.parse_args(argv)
    from natlang_neuralese.common.jsonio import read_jsonl, write_json_durable
    from natlang_neuralese.rl import production, interleave
    from natlang_neuralese.rl.curriculum import difficulty_shift, policy_options, spread_yield
    from natlang_neuralese.train.recipe import resolve_recipe_data
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = resolve_recipe_data(Path(args.config))
    config = {**config, 'out': str(out / 'controller')}  # never the control's own controller directory
    Path(config['out']).mkdir(parents=True, exist_ok=True)
    state_path = out / 'pilot-state.json'
    state = (json.loads(state_path.read_text()) if state_path.exists()
             else production.initial_controller_state(config))
    config, declared = interleave.bind_stage(config, state, state_path)
    config = production.adopt_policy_loss_defaults(config, state, state_path)
    config = production.adopt_task_curriculum_defaults(config, state, state_path)
    config = production.adopt_actor_engine_defaults(config, state, state_path)
    if args.no_teacher:
        config['jolt_teacher_attempts'] = 0
    cases = list(read_jsonl(config['cases']))
    reports = []
    for _ in range(args.rounds):
        number = state['round']
        round_dir = out / f'round-{number:06d}'
        stats = [*state.get('task_curriculum_history', {}).values()]
        budget = {'rows': config['rows_per_round'], 'group': config['group'], 'offset': state['offset']}
        selection = production.select_tasks(cases, stats, budget, curriculum=config['task_curriculum'])
        selected = [cases[item['row']] for item in selection]
        round_dir.mkdir(parents=True, exist_ok=True)
        selected_path = round_dir / 'selected-cases.jsonl'
        selected_path.write_text(''.join(json.dumps(row) + '\n' for row in selected))
        shift = difficulty_shift(stats, policy_options(config['task_curriculum']))
        config_round, values, policy, policy_loss, expected_native, kwargs = production.prepare_round_collection(
            config, state, round_dir, selected_path, selection)
        report = production.collect_with_actor(config_round, values, **kwargs)
        groups = report.get('groups', [])
        history = state.setdefault('task_curriculum_history', {})
        for group in groups:
            if group.get('scored', 0) > 0:
                history[group['task_id']] = {**group, 'observed_policy_id': policy, 'observed_round': number,
                                             'collection_attempt': 0}
        summary = {'round': number, 'policy_sha256': policy, 'budget': budget,
                   'difficulty_shift_at_selection': shift, 'spread': spread_yield(groups),
                   'scored_episodes': report.get('scored'), 'successes': report.get('successes'),
                   'families': family_report(selected, groups), 'optimizer_updates': 0}
        reports.append(summary)
        write_json_durable(round_dir / 'pilot-round.json', summary)
        state.update(round=number + 1, offset=(selection[-1]['row'] + 1) % len(cases))
        write_json_durable(state_path, state)
        print(json.dumps(summary), flush=True)
    write_json_durable(out / 'pilot-report.json', {'schema': 'natlang.rl-collection-pilot/1', 'config': args.config,
        'cases': config['cases'], 'rounds': reports, 'optimizer_updates': 0,
        'scope': 'collection and scoring only; selection adaptation across rounds of one fixed policy'})
    return 0


if __name__ == '__main__':
    sys.exit(main())
