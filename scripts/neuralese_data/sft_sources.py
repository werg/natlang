#!/usr/bin/env python3
"""SFT data set from verified trajectories: portable GRPO/teacher successes replayed through the runtime and oracle,
and observed-live executor calls with scoped effect verification, capped, deduplicated, held-excluded and rendered with the LFM3 chat template
(plans/neuralese/SFT_SOURCES_20261117.md).

Stages (each reads the previous stage's output under WORK; read-only on every source):

  extract  every `task_success` episode of the GRPO rollout archives (`--rl RUN_DIR:ARCHIVE_DIR`, the run's live
           group directories included) and of teacher-generation pools (`--teacher POOL_DIR`) -> WORK/episodes.jsonl.zst
           (`--out-name`: a growing teacher pool is re-extracted alone as WORK/episodes-teacher.jsonl.zst before a build;
           the build refuses manifest successes that no extraction covers):
           the task row (natlang.program/2, from the group's cases.jsonl), the semantic turns ({messages, tools, target})
           and, for student episodes, the exact actor prompt/sample token ids of each call (renderer check only).
           A JOLT+ teacher attempt's success enters as `rl-own` (`role: teacher`) in student form
           (`rl/jolt.py student_episode`: guidance turn and the guidance policy's calls removed).
           `--october-teacher GROUP.jsonl.zst` turns October per-call teacher rows (`--october-model` filters) into
           `teacher` episodes (task row from the rows' source turns, held closure out) for the replay to verify.
  replay   each episode's task runs again under this checkout's runtime (`task_rollouts.prepare`, `run_bonsai_queue.py`,
           `task_rollouts.score`, the teacher_gen path) against a replay server that answers every model request with
           the recorded reply of the same call (matched by its visible messages); WORK/replay/<key>/rollouts.jsonl is the
           current runtime's record of the same decisions and its oracle verdict. A call the recording has no reply for
           ends the episode (the actions no longer lead where they led).
  build    the replayed successes -> per-call rows: held-v3/v2 closure excluded, per-task cap, near-duplicate removal,
           tasks weighted toward low base pass rates, per-source shares, one epoch; rendered with the chat template
           (prompt zero-weight, the reply the target) into OUT/rows.jsonl.zst + OUT/chains.pkl (`exp0_sft.py train
           --chains`) + OUT/build.json.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from unicodedata import normalize

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / 'training/neuralese'), str(REPO), str(HERE), str(REPO / 'scripts/rl_tasks')]

from natlang_neuralese.common.hashing import sha256_file_hex as sha256_file  # noqa: E402
from natlang_neuralese.common import data_registry  # noqa: E402
from natlang_neuralese.common.paths import resolve_str  # noqa: E402
import recorded_turns  # noqa: E402

SUCCESS = 'task_success'
MEMETIC = 'memetic_scored'   # a memetic improvement episode's outcome (task_rollouts.score); success: memetic_success


def episode_success(row):
    """A scored natlang.rollout/1 row's verified success: a binary task success, or a memetic improvement episode
    whose committed change set beat the trivial improvements on the private query worlds (``memetic_success``,
    task_rollouts.memetic_success: the supervisor's own calls train as an own success; the guidance it wrote is its own
    sampled action, and the nested runs of the guided program are never extracted)."""
    if row.get('outcome') == MEMETIC:
        return bool(row.get('memetic_success'))
    return row.get('outcome') == SUCCESS and float((row.get('reward') or {}).get('quality') or 0) >= 1
OPS = ('data_nvme', 'lfm3-rl-dgx-20261029', 'ops')


def zstd_lines(path):
    """Lines of a .zst or plain JSONL file, streamed."""
    import io
    import zstandard
    if str(path).endswith('.zst'):
        with open(path, 'rb') as raw:
            with zstandard.ZstdDecompressor().stream_reader(raw) as reader:
                yield from io.TextIOWrapper(reader, encoding='utf-8')
    else:
        with open(path, encoding='utf-8') as handle:
            yield from handle


class ZstdWriter:
    """JSONL writer to path.zst through a temp file renamed on close."""

    def __init__(self, path, level=6):
        import zstandard
        self.path = Path(path)
        self.partial = self.path.with_suffix(self.path.suffix + '.partial')
        self.raw = self.partial.open('wb')
        self.writer = zstandard.ZstdCompressor(level=level).stream_writer(self.raw)
        self.rows = 0

    def write(self, row):
        self.writer.write((json.dumps(row, ensure_ascii=False) + '\n').encode())
        self.rows += 1

    def close(self, *, publish=True):
        self.writer.close()   # ends the frame and closes the file
        if publish:
            os.replace(self.partial, self.path)
        else:
            self.partial.unlink()


# ---- extract ---------------------------------------------------------------------------------------------------------
def exact_tokens(turn):
    """(prompt ids, sampled ids) of a student call from its sampling capture, or None (teacher call, native imports)."""
    sampling = turn.get('sampling') or {}
    prompt = sampling.get('prompt') or []
    if not prompt or any(part.get('kind') != 'tokens' for part in prompt):
        return None
    ids = [i for part in prompt for i in part['token_ids']]
    return {'prompt_ids': ids, 'sample_ids': [s['token_id'] for s in sampling.get('samples') or []]}


def semantic_turn(turn, keep_exact):
    out = {k: turn.get(k) for k in ('messages', 'tools', 'target', 'invocation_id')}
    for key in ('function', 'definition_id', 'parent_invocation', 'invocation_role'):
        if key in turn:
            out[key] = turn[key]
    out['completion_tokens'] = (turn.get('usage') or {}).get('completion_tokens')
    if keep_exact:
        out['exact'] = exact_tokens(turn)
    return out


def group_cases(directory):
    rows = {}
    path = Path(directory) / 'cases.jsonl'
    if path.exists():
        for line in path.open():
            if line.strip():
                row = json.loads(line)
                rows[row['id']] = row
    return rows


def rl_rollout_files(run_dir, archive_dir):
    """(group name, rollouts file, group dir) of every group of one GRPO run: archived (zst) or still live."""
    seen = set()
    for path in sorted(Path(archive_dir).glob('g*.rollouts.jsonl.zst')):
        name = path.name.split('.')[0]
        seen.add(name)
        yield name, path, Path(run_dir) / 'groups' / name
    for directory in sorted(Path(run_dir, 'groups').glob('g*')):
        if directory.name not in seen and (directory / 'rollouts.jsonl').exists() and (directory / 'score.json').exists():
            yield directory.name, directory / 'rollouts.jsonl', directory


# Every output is recorded in the data registry when it is written (natlang_neuralese.common.data_registry): a WORK after
# each extract and replay (a refreshed WORK is a new snapshot), a built set with what it was built from.
DERIVED_LICENSE = 'derived: each input keeps its registered license (teacher providers, own rollouts, task sources)'
WORK_INCLUDE = ['*.jsonl.zst', '*.json', '*.jsonl', 'replay/*/rollouts.jsonl', 'replay/*/*.json']


def record_work(work, inputs):
    data_registry.record_dataset(data_registry.dataset_id(work, 'sft-work'), work, kind='trajectories', include=WORK_INCLUDE, inputs=inputs,
                                 format='sft_sources WORK: episodes*.jsonl.zst (extracted episodes), replay.jsonl '
                                        '(verdicts), replay/<episode>/rollouts.jsonl (verified replays)',
                                 license=DERIVED_LICENSE, closed=False)


def record_built(out, args, schema):
    inputs = [args.work, getattr(args, 'held_manifest', None), getattr(args, 'base_rates', None),
              getattr(args, 'october', None), getattr(args, 'live_holdout_registry', None),
              *(getattr(args, 'teacher_pool', None) or []), *(getattr(args, 'prefer_unseen', None) or []),
              *[a.split('=', 1)[1].split(',')[0] for a in getattr(args, 'aux', None) or []]]
    data_registry.record_dataset(data_registry.dataset_id(out, 'sft-rows'), out, kind='sft-rows', format=f'{schema}: chains.pkl (exp0_sft --chains), '
                                 'rows.jsonl.zst, cases.jsonl, build.json', inputs=[i for i in inputs if i],
                                 license=DERIVED_LICENSE)


def build_inputs(args):
    """A build's inputs, checked before it reads them: data no registry entry holds stops it here."""
    return [p for p in [args.work, getattr(args, 'held_manifest', None), getattr(args, 'base_rates', None),
                        getattr(args, 'october', None), *(getattr(args, 'teacher_pool', None) or []),
                        *(getattr(args, 'prefer_unseen', None) or []),
                        *[a.split('=', 1)[1].split(',')[0] for a in getattr(args, 'aux', None) or []]] if p]


def october_group(path):
    return Path(path).name.split('.')[0]


TRAJECTORY_ID = re.compile(r'"trajectory_id"\s*:\s*"([^"]+)"')


def october_task_rows(source_files, wanted):
    """{trajectory: task row} from the October sources the rows were rebuilt from: each source turn carries its task as
    `task.program_ir` (natlang.program/2, the cases.jsonl row form); one streamed pass per file, parsing only the turns
    of wanted trajectories."""
    rows, left = {}, set(wanted)
    for source in source_files:
        for line in zstd_lines(source):
            if not left:
                return rows
            match = TRAJECTORY_ID.search(line[:2000])
            if match is None or match.group(1) not in left:
                continue
            program = ((json.loads(line).get('task') or {}).get('program_ir'))
            if program:
                rows[match.group(1)] = program
                left.discard(match.group(1))
    return rows


def october_teacher_episodes(path, models, held_manifest, c):
    """Teacher episodes of one October group file (sft-sources-20261117/october/GROUP.jsonl.zst, per-call rows rebuilt
    for the current runtime; `models`: the provenance models kept, all when empty): rows grouped by trajectory in decision
    order, the task row read back from the turns' source files, held closure and unadmitted fixtures dropped. Each
    becomes a `teacher` episode the replay stage verifies like a teacher_gen success. Two passes (a group is GBs of
    JSON): the first counts each trajectory's rows, the second holds a trajectory's raw lines until it is complete."""
    from teacher_admission import fixture_task
    from teacher_provenance import SELF_HOSTED
    group = october_group(path)
    counts, sources = collections.Counter(), {}
    for line in zstd_lines(path):
        provenance = json.loads(line).get('provenance') or {}
        c['rows'] += 1
        if models and provenance.get('model') not in models:
            c['rows_other_model'] += 1
            continue
        counts[provenance['trajectory']] += 1
        sources.setdefault(provenance['trajectory'], provenance['source_file'])
    for source in sorted(set(sources.values())):   # a runs/ link or its target may be the registered path
        try:
            data_registry.require_registered([source])
        except data_registry.UnregisteredData:
            data_registry.require_registered([str(Path(source).resolve())])
    tasks = october_task_rows(sorted(set(sources.values())), counts)
    by_id = {}
    for task in tasks.values():
        by_id.setdefault(task['id'], task)
    held = held_closure(held_manifest)
    excluded = held_excluded_ids(held_manifest, by_id, held) if by_id else set()
    pending = collections.defaultdict(list)
    for line in zstd_lines(path):
        provenance = json.loads(line).get('provenance') or {}
        trajectory = provenance.get('trajectory')
        if trajectory not in counts:
            continue
        pending[trajectory].append(line)
        if len(pending[trajectory]) < counts[trajectory]:
            continue
        rows = [json.loads(raw) for raw in pending.pop(trajectory)]
        c['trajectories'] += 1
        rows.sort(key=lambda r: int(r['id'].rsplit(':', 1)[1]))
        if [int(r['id'].rsplit(':', 1)[1]) for r in rows] != list(range(len(rows))):
            c['decision_gaps'] += 1   # replayed anyway: a missing recorded call ends its replay
        task = tasks.get(trajectory)
        if task is None:
            c['no_task_row'] += 1
            continue
        if task['id'] != rows[0].get('task_id'):
            # fallow-v13r2 rows carry the source turn's stale `program_id`; the turn's own program_ir is the task its
            # messages show (and the replay verifies it). Both ids go through the held closure.
            c['task_id_from_program_ir'] += 1
        if task['id'] in excluded or held_hit(rows[0].get('task_id'), rows[0].get('source_ids') or (),
                                              rows[0].get('source_groups') or (), held):
            c['held_excluded'] += 1
            continue
        if fixture_task(task):
            c['fixture_not_admitted'] += 1
            continue
        c['episodes'] += 1
        model = rows[0]['provenance'].get('model')
        turns = [semantic_turn({'messages': r['messages'], 'tools': r['tools'], 'target': r['target'],
                                'invocation_id': r['provenance'].get('invocation'),
                                'usage': {'completion_tokens': r['provenance'].get('target_tokens')}}, False)
                 for r in rows]
        yield {'key': f'october-teacher:{group}:{trajectory}', 'source': 'teacher', 'run': 'october:' + group,
               'group': group, 'sample': 0, 'seed': None, 'task_id': task['id'],
               'family': rows[0].get('family') or task.get('family'),
               'teacher': {'model_id': model, 'provider': SELF_HOSTED.get(model, 'self-hosted-vllm'),
                           'origin': 'october:' + group, 'source_file': rows[0]['provenance']['source_file']},
               'base_rank': None, 'task': task, 'root_mode': 'student', 'turns': turns, 'execution_graph_events': 0}


def live_manifest_paths(manifests, capture_roots, *, allow_empty=False):
    paths = [Path(path).resolve() for path in manifests]
    for directory in capture_roots:
        root = Path(directory).resolve()
        candidates = [root / 'verified/manifest.json',
                      *sorted((root / 'cases').glob('*/verified/manifest.json'))]
        discovered = [path.resolve() for path in candidates if path.is_file()]
        if not discovered and (not allow_empty or not root.is_dir()):
            raise ValueError(f'No verified live teacher manifests under {root}')
        if any(not path.is_relative_to(root) for path in discovered):
            raise ValueError(f'Verified live teacher manifest escapes the capture directory {root}')
        paths.extend(discovered)
    return list(dict.fromkeys(map(str, paths)))


def call_policy(turn):
    """The policy that sampled one recorded call: the RL capture's ``sampling.policy`` (a rebind lands between calls,
    so an episode may span two adjacent policies), else a turn-level ``policy``."""
    for policy in ((turn.get('sampling') or {}).get('policy'), turn.get('policy')):
        if isinstance(policy, dict) and policy.get('checkpoint_sha256'):
            return policy['checkpoint_sha256']
    return None


def extract(args):
    live_runs = live_manifest_paths(getattr(args, 'live_run', None) or [],
                                    getattr(args, 'live_capture_root', None) or [])
    data_registry.require_registered([*(r.split(':', 1)[1] if ':' in r else r for r in args.rl), *args.teacher,
                                      *live_runs,
                                      *(getattr(args, 'october_teacher', None) or [])])
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    episodes_out = work / f'{args.out_name}.jsonl.zst'
    native_out = getattr(args, 'live_trajectory_out', None)
    if native_out:
        native_out = Path(native_out)
        if not native_out.name.endswith('.jsonl.zst') or native_out.resolve() == episodes_out.resolve():
            raise ValueError('live trajectory output needs a distinct .jsonl.zst path')
    out = ZstdWriter(episodes_out)
    native_lines = ZstdWriter(native_out) if native_out else None
    census = collections.OrderedDict()
    group_rates = collections.defaultdict(lambda: [0, 0])     # task -> [successes, scored] over all student groups
    for spec in args.rl:
        run_dir, archive_dir = spec.split(':')
        label = 'rl:' + '/'.join(Path(run_dir).parts[-2:])
        c = census.setdefault(label, collections.Counter())
        for name, path, group_dir in rl_rollout_files(run_dir, archive_dir):
            cases = group_cases(group_dir)
            c['groups'] += 1
            if not path.exists():   # a live group the run archived since the listing
                path = Path(archive_dir) / f'{name}.rollouts.jsonl.zst'
            for line in zstd_lines(path):
                if not line.strip():
                    continue
                row = json.loads(line)
                c['episodes'] += 1
                if row.get('unscored') or row.get('reward') is None:
                    c['unscored'] += 1
                    continue
                role = row.get('jolt_role', 'student')
                success = episode_success(row)
                if role != 'teacher':   # a guided attempt is no sample of the student's own pass rate
                    group_rates[row['task_id']][1] += 1
                    group_rates[row['task_id']][0] += success
                if not success:
                    continue
                task = cases.get(row['task_id'])
                if task is None:
                    c['success_without_task_row'] += 1
                    continue
                c['successes'] += 1
                if row.get('outcome') == MEMETIC:
                    c['memetic_successes'] += 1
                if role == 'teacher':
                    # a JOLT+ teacher success trains in student form: its consumer calls without the guidance turn,
                    # without the guidance policy's own calls (RESTORE_MEMETIC_JOLT_HINDSIGHT_20261117.md section 4);
                    # one whose replies quote or name the guidance, or whose guidance carried the answer, is left out
                    from natlang_neuralese.rl.jolt import guidance_leak, student_episode
                    from natlang_neuralese.rl.privileged_context import answer_leaves
                    if guidance_leak(row, answers=answer_leaves(task)):
                        c['teacher_attempt_guidance_leak'] += 1
                        continue
                    row = student_episode(row, c)
                    c['teacher_attempt_successes'] += 1
                ep_policy = (row.get('policy') or {}).get('checkpoint_sha256')
                call_policies = sorted({p for t in row.get('turns') or () for p in (call_policy(t),) if p})
                if ep_policy and ep_policy not in call_policies:
                    call_policies = sorted({*call_policies, ep_policy})
                out.write({'key': f"{label}:{name}:{row['sample']}",
                           'source': 'jolt-guided' if role == 'teacher' else 'rl-own',
                           'run': label, 'group': name,
                           **({'role': 'teacher'} if role == 'teacher' else {}),
                           'sample': row['sample'], 'seed': row.get('seed'), 'task_id': row['task_id'],
                           'family': row.get('family'), 'policy': ep_policy,
                           **({'policies': call_policies} if call_policies else {}),
                           'task': task, 'turns': [semantic_turn(t, True) for t in row['turns']],
                           'execution_graph_events': len(row.get('execution_graph') or [])})
    for pool in args.teacher:
        label = 'teacher:' + '/'.join(Path(pool).parts[-2:])
        c = census.setdefault(label, collections.Counter())
        for line in Path(pool, 'manifest.jsonl').open():
            if not line.strip():
                continue
            record = json.loads(line)
            c['tasks'] += 1
            directory = Path(record['rollouts']).parent
            cases = group_cases(directory)
            from teacher_admission import check_record, fixture_task
            reviewed = check_record(record, pool) if record.get('admission') else {}
            for rline in Path(record['rollouts']).open():
                if not rline.strip():
                    continue
                row = json.loads(rline)
                c['episodes'] += 1
                if row.get('outcome') != SUCCESS or float((row.get('reward') or {}).get('quality') or 0) < 1:
                    continue
                task = cases.get(row['task_id'])
                if task is None:
                    c['success_without_task_row'] += 1
                    continue
                admission = reviewed.get((row['task_id'], row['sample']))
                if fixture_task(task) and admission is None:
                    c['fixture_not_admitted'] += 1
                    continue
                c['successes'] += 1
                turns = []
                for turn in row['turns']:
                    metadata = (admission or {}).get('invocations', {}).get(turn.get('invocation_id'), {})
                    turns.append(semantic_turn({**turn, **metadata}, False))
                out.write({'key': f"{label}:{directory.name}:{row['sample']}", 'source': 'teacher', 'run': label,
                           'group': directory.name, 'sample': row['sample'], 'seed': row.get('seed'),
                           'task_id': row['task_id'], 'family': row.get('family'), 'teacher': record.get('teacher'),
                           'base_rank': record.get('base_rank'), 'task': task,
                           'root_mode': record.get('root_mode', 'student'),
                           'turns': turns,
                           **({'admission': {'path': admission['path'], 'sha256': admission['sha256'],
                                            'manifest_record': record, 'pool': str(Path(pool).resolve())}}
                              if admission else {}),
                           'execution_graph_events': len(row.get('execution_graph') or [])})
    for path in getattr(args, 'october_teacher', None) or []:
        for episode in october_teacher_episodes(path, getattr(args, 'october_model', None) or (), args.held_manifest,
                                                census.setdefault('october-teacher:' + october_group(path),
                                                                  collections.Counter())):
            out.write(episode)
    for path in live_runs:
        from live_teacher_sources import load_episode
        episode = load_episode(path)
        if episode.get('trajectory_ir'):
            if native_lines:
                native_lines.write(episode['trajectory_ir'])
            del episode['turns']  # the learner reads the decision targets from the native trajectory
        out.write(episode)
        census.setdefault('observed-live', collections.Counter())['admitted_tasks'] += 1
    out.close()
    if native_lines:
        native_lines.close()
    if args.rl:
        rates = {k: {'successes': s, 'scored': n} for k, (s, n) in sorted(group_rates.items())}
        (work / 'rl-group-rates.json').write_text(json.dumps(rates, indent=0) + '\n')
    summary = {k: dict(v) for k, v in census.items()}
    (work / f'{args.out_name}-extract.json').write_text(json.dumps({'sources': summary, 'episodes': out.rows,
                                                    'rl': args.rl, 'teacher': args.teacher, 'live_runs': live_runs,
                                                    'october_teacher': getattr(args, 'october_teacher', None) or []}, indent=1) + '\n')
    record_work(work, [*(r.split(':', 1)[1] if ':' in r else r for r in args.rl), *args.teacher,
                       *live_runs, *(getattr(args, 'october_teacher', None) or [])])
    print(json.dumps(summary, indent=1))


def episode_files(work):
    """Every extraction of WORK: `episodes.jsonl.zst` and `episodes-*.jsonl.zst` (e.g. a teacher pool re-extracted at
    build time as `--out-name episodes-teacher`)."""
    return sorted(Path(work).glob('episodes*.jsonl.zst'))


def all_episodes(work, workers=1, light=False, keep=None):
    """The extracted episodes of WORK, each key once (a later extraction of the same episode is the same record).
    ``light``: an episode's ``turns`` are replaced by their ``facts`` (``episode_facts``: what the build's candidate
    selection reads of them; observed-live episodes keep their turns), so ``workers`` > 1 processes can parse the
    lines and send back only that (``bounded_map``); the episodes, their order and the facts are the serial ones.
    ``keep``: a picklable predicate on the full episode; the others are not yielded (nor sent back)."""
    seen = set()
    lines = (line for path in episode_files(work) for line in zstd_lines(path) if line.strip())
    if workers > 1:
        import concurrent.futures
        import multiprocessing
        with concurrent.futures.ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            for batch in bounded_map(pool, _parse_episodes, batched(lines, 8), 2 * workers, light, keep):
                for episode in batch:
                    if episode['key'] not in seen:
                        seen.add(episode['key'])
                        if isinstance(episode.get('error'), Exception):
                            raise episode['error']
                        if not episode.get('dropped'):
                            yield episode
        return
    for line in lines:
        episode = json.loads(line)
        if episode['key'] not in seen:
            seen.add(episode['key'])
            episode = _episode_view(episode, light, keep)
            if episode is not None:
                yield episode


def _episode_view(episode, light, keep=None):
    if episode.get('trajectory_ir'):
        from live_teacher_sources import observed_training_turns
        episode['turns'] = observed_training_turns(episode['trajectory_ir'])
    if keep is not None and not keep(episode):
        return None
    if light and episode.get('verification_mode') != 'observed-live':
        episode['facts'] = episode_facts(episode.pop('turns'))
    return episode


def _parse_episodes(lines, light, keep=None):
    out = []
    for line in lines:
        episode = json.loads(line)
        try:
            out.append(_episode_view(episode, light, keep) or {'key': episode['key'], 'dropped': True})
        except Exception as error:   # raised by the reader only if this record is the one it yields
            out.append({'key': episode['key'], 'error': error})
    return out


def episode_facts(turns):
    """What candidate selection reads of an episode's turns: the action MinHash (near-duplicates), whether it met the
    text backend's Neuralese refusal, and whether its sibling calls are ambiguous (``drifted_success``)."""
    return {'signature': minhash(shingles(action_text(turns))), 'backend_workaround': backend_workaround(turns),
            'sibling_ambiguous': sibling_ambiguous(turns)}


def batched(items, n):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def bounded_map(pool, fn, items, window, *args):
    """``fn(item, *args)`` over ``items`` in order by an executor, at most ``window`` items in flight (an executor's
    own ``map`` submits every item at once)."""
    pending = collections.deque()
    for item in items:
        pending.append(pool.submit(fn, item, *args))
        if len(pending) >= window:
            yield pending.popleft().result()
    while pending:
        yield pending.popleft().result()


def teacher_coverage(pools, extracted):
    """(successes in the pools' manifests now, those not extracted yet): the build refuses a stale teacher snapshot."""
    total, missing = 0, []
    fixtures = {}       # group directory -> its fixture task ids (records of one group share its cases.jsonl)
    for pool in pools:
        label = 'teacher:' + '/'.join(Path(pool).parts[-2:])
        for line in Path(pool, 'manifest.jsonl').open():
            if not line.strip():
                continue
            record = json.loads(line)
            source_directory = Path(record['rollouts']).parent
            directory = source_directory.name
            from teacher_admission import fixture_task, check_record
            if source_directory not in fixtures:
                fixtures[source_directory] = {task_id for task_id, task in group_cases(source_directory).items()
                                              if fixture_task(task)}
            fixture_ids = fixtures[source_directory]
            reviewed = check_record(record, pool) if fixture_ids and record.get('admission') else {}
            for episode in record.get('episodes') or ():
                if episode.get('outcome') == SUCCESS:
                    if record.get('task_id') in fixture_ids and (record['task_id'], episode['sample']) not in reviewed:
                        continue
                    total += 1
                    key = f"{label}:{directory}:{episode['sample']}"
                    if key not in extracted:
                        missing.append(key)
    return total, missing


# ---- rendering -------------------------------------------------------------------------------------------------------
class TemplateRenderer:
    """The serving engine's own request rendering (`serve.engine.Engine.render_request`, `serve.targets.target_items`)
    on a tokenizer alone: no model is loaded. `render(messages, tools, target)` -> (prompt ids, target ids), the target
    being the assistant reply after the generation prompt through its end token. `check` compares it with the actor's
    recorded prompt and sampled ids (the RL captures)."""

    def __init__(self, tokenizer_dir):
        from transformers import AutoTokenizer
        from natlang_neuralese.serve.chat import bind_history_reasoning, tool_call_format
        from natlang_neuralese.serve.engine import Engine
        # the checkpoint's config.json minus auto_map (no remote code: only the tokenizer is loaded)
        config = json.loads(Path(tokenizer_dir, 'model_type.json').read_text())
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), trust_remote_code=False)
        self.tokenizer = bind_history_reasoning(tokenizer, model_type=config['model_type'])
        self.tool_call_format = tool_call_format(config['model_type'])
        self.prompt_rendering = 'full'
        self.identity = {'tokenizer_sha256': sha256_file(Path(tokenizer_dir, 'tokenizer.json')),
                         'chat_template_sha256': sha256_file(Path(tokenizer_dir, 'chat_template.jinja')),
                         'model_type': config['model_type'],
                         'history_reasoning': getattr(self.tokenizer, 'natlang_history_reasoning', None)}
        self._engine = Engine

    # the engine's methods, bound to this tokenizer-only object
    def _template(self, messages, tools):
        return self._engine._template(self, messages, tools)

    def render_request(self, messages, tools=None, template=None, *, response_tools=None):
        return self._engine.render_request(self, messages, tools, template, response_tools=response_tools)

    def _tokens(self, text):
        return self._engine._tokens(self, text)

    def _template_tokens(self, text, escape_nonce=''):
        return self._engine._template_tokens(self, text, escape_nonce)

    @property
    def specials(self):
        return self._engine.specials.fget(self)

    def render(self, messages, tools, target):
        from natlang_neuralese.serve.targets import target_items
        target = {k: v for k, v in target.items() if v is not None}
        before, rest = target_items(self, list(messages), tools, target)
        if any(kind != 'tok' for kind, _ in before + rest):
            raise ValueError('native-state blocks in a text-only rendering')
        return [i for _, i in before], [i for _, i in rest]


OWN_SOURCES = ('rl-own', 'jolt-guided')
TEACHER_THINK = ('context', 'train')
TEACHER_LEAD_IN = ('context', 'train')


def think_context(renderer, source, target_message, target_ids, teacher_think='context', lead_in='context',
                  own_empty_think='context'):
    """Leading target tokens that are context, not loss: for a reply the student did not sample, its think block and
    (``lead_in`` 'context') the text it writes before its first tool call (``serve.targets.non_student_context``, the
    one rule the program-tune rows share). sft-v3-run1 trained teacher think blocks: held-v3 reasoning p50 fell
    1,747 -> 336 characters, Luna's "So far ... Next," style appeared and ran-out rose 16.6% -> 29.6% (DECISIONS
    2026-11-18). Lead-in text is narration: Claude 28% of its tool calls, OpenHands 54%, the student 1.7%; held after
    sft-v3 12% vs 2%. The student's own replies keep every token as sampled, except (``own_empty_think`` 'context') the
    template's empty think block of an own reply without reasoning: a decision readout the actor scored, not sampled
    (its option is the scored continuation after the prefix all options share, so ``</think>`` at the first target
    position is never the student's choice; the base gives it ~20 nats). sft-v4-run1 trained it in 18 own rows."""
    from natlang_neuralese.serve.targets import non_student_context
    if teacher_think not in TEACHER_THINK:
        raise ValueError(f'teacher_think {teacher_think!r} not in {TEACHER_THINK}')
    if lead_in not in TEACHER_LEAD_IN:
        raise ValueError(f'lead_in {lead_in!r} not in {TEACHER_LEAD_IN}')
    if own_empty_think not in TEACHER_THINK:
        raise ValueError(f'own_empty_think {own_empty_think!r} not in {TEACHER_THINK}')
    if source in OWN_SOURCES:
        if own_empty_think == 'train' or (target_message.get('reasoning_content') or '').strip():
            return 0
        return non_student_context(renderer.tokenizer, target_message, target_ids, think='context', lead_in='train')
    return non_student_context(renderer.tokenizer, target_message, target_ids, think=teacher_think, lead_in=lead_in)


# Teacher reasoning is chosen per teacher model, source or episode (owner 2026-11-20): a treatment table published through
# the registry role ``teacher-think-treatment`` (the rationalization strand's per-model audit) says, per episode, whether
# a non-student reply's think block trains (``train``: long enough, close to the student's style, actions likely after
# it), stays masked context (``context``) or gives way to student-filled reasoning (``filled``: the filled version is an
# ``--aux NAME=PATH,replaces=teacher`` part, filled_reasoning.py; an episode without one renders as ``context``).
THINK_TREATMENTS = ('train', 'context', 'filled')
THINK_TREATMENT_ROLE = 'teacher-think-treatment'


class ThinkTreatment:
    """``natlang.teacher-think-treatment/1``: {"default": T, "models": {teacher model id: T}, "sources": {run or pool
    label: T}, "episodes": {episode key: T}}; the most specific entry wins (episode, source, model, default)."""

    def __init__(self, spec=None, default='context', path=None):
        spec = spec or {}
        self.path, self.default = path, spec.get('default', default)
        self.models, self.sources, self.episodes = (dict(spec.get(k) or {}) for k in ('models', 'sources', 'episodes'))
        for value in [self.default, *self.models.values(), *self.sources.values(), *self.episodes.values()]:
            if value not in THINK_TREATMENTS:
                raise ValueError(f'think treatment {value!r} not in {THINK_TREATMENTS}')
        self.counts = collections.Counter()

    @classmethod
    def load(cls, path, default):
        """PATH, 'none' (every episode ``default``), or None: the registry role's file when one dataset holds it."""
        if path == 'none':
            return cls(default=default)
        if path is None:
            paths = data_registry.role_paths(THINK_TREATMENT_ROLE)
            if not paths:
                return cls(default=default)
            if len(paths) > 1:
                raise ValueError(f'registry role {THINK_TREATMENT_ROLE} names {len(paths)} files')
            path = paths[0]
        return cls(json.loads(Path(path).read_text()), default, str(path))

    def of(self, c):
        """The treatment of candidate ``c`` (own episodes: None, every sampled token is theirs)."""
        if c.get('source') in OWN_SOURCES:
            return None
        for table, key in ((self.episodes, c.get('key')), (self.sources, c.get('run')), (self.models, c.get('teacher'))):
            if key is not None and key in table:
                return table[key]
        return self.default

    def render(self, c):
        """The ``teacher_think`` a candidate renders with (``filled`` while it is not replaced: context)."""
        value = self.of(c)
        return 'context' if value in (None, 'filled') else value

    def report(self):
        return {'path': self.path, 'sha256': sha256_file(self.path) if self.path else None, 'default': self.default,
                'models': self.models, 'sources': self.sources, 'episodes': len(self.episodes),
                'admitted_episodes': dict(self.counts)}


# The frontier (expert_iteration.py frontier): the measured pass rate of the policy a build trains, per task, and its
# tier. With ``build --frontier`` each task keeps its cheapest verified source (``frontier_drop``).
FRONTIER_TIERS = ('solved', 'sometimes', 'never', 'unmeasured')


def load_frontier_full(path):
    """``natlang.ei-frontier/1`` -> (policy sha, previous_policy sha or None, {task id: {'successes', 'scored', 'tier'}})."""
    if not path:
        return None, None, {}
    data = json.loads(Path(path).read_text())
    if data.get('schema') != 'natlang.ei-frontier/1':
        raise ValueError(f'{path}: not a frontier file')
    return data.get('policy'), data.get('previous_policy'), data['tasks']


def load_frontier(path):
    """``natlang.ei-frontier/1`` -> (policy sha, {task id: {'successes', 'scored', 'tier'}})."""
    policy, _, tasks = load_frontier_full(path)
    return policy, tasks


def frontier_drop(task_id, tiers, own_tasks, guided_tasks=()):
    """Why a lower-tier trajectory of ``task_id`` leaves a frontier build, or None (async-train §8):
    tier hierarchy is own unguided success > leak-checked ``jolt-guided`` success > teacher / filled trajectory;
    a task solved reliably needs only its own anchor."""
    if tiers.get(task_id) == 'solved':
        return 'frontier_task_solved'
    if task_id in own_tasks:
        return 'frontier_own_success'
    if task_id in guided_tasks:
        return 'frontier_jolt_guided'
    return None


def own_policy_tier(candidate, frontier_policy, previous_policy=None):
    """0 for own successes from ``frontier_policy`` or ``previous_policy`` (including rebind-straddling episodes
    whose per-call policies all lie within that adjacent pair, async-train §3); 1 for older policies."""
    if not frontier_policy:
        return 0
    allowed = {p for p in (frontier_policy, previous_policy) if p}
    policies = [p for p in (candidate.get('policies') or ()) if p]
    if not policies:
        p = candidate.get('policy')
        policies = [p] if p else []
    if policies and all(p in allowed for p in policies):
        return 0
    return 1


def replay_context_tokens(source):
    """The replay runtime's context for an extracted episode: the actor's own (131072) for the student's own
    distribution (``OWN_SOURCES``: rl-own and jolt-guided), 65536 for a teacher's."""
    return 131072 if source in OWN_SOURCES else 65536


def settle_deferred(deferred, admitted_tasks, task_of):
    """Teacher data (episodes, filled aux units) of guided-only tasks after the own share (``build`` step 5): (replaced:
    their task's guided successes entered the mix, trains: none did, so the teacher data trains)."""
    replaced = [item for item in deferred if task_of(item) in admitted_tasks]
    return replaced, [item for item in deferred if task_of(item) not in admitted_tasks]


def student_selection(unique, rng, funnel, cap_of, family_share, is_wrapper, wrapper_share, trained=frozenset(),
                      tiers=None):
    """The student's own distribution of a build: unguided own successes (``rl-own``) and leak-checked JOLT+ guided
    successes (``jolt-guided``) under one per-task cap, one family share and one wrapper share, own first under each
    (``capped`` ``rank``). With a frontier (``tiers``): a task's guided successes go where it has unguided ones
    (``frontier_drop``). Returns (pool, own tasks, guided-only tasks); the guided-only tasks' teacher data waits for
    the own share (``build`` step 5: it trains unless their guided successes enter the mix)."""
    own = [c for c in unique if c['source'] == 'rl-own']
    own_candidate_tasks = {c['task_id'] for c in own}
    candidates = list(own)
    for c in unique:
        if c['source'] != 'jolt-guided':
            continue
        why = frontier_drop(c['task_id'], tiers, own_candidate_tasks) if tiers is not None else None
        if why:
            funnel[c['source']][why] += 1
        else:
            candidates.append(c)

    def rank(c):
        return int(c['source'] == 'jolt-guided')
    pool = wrapper_capped(capped(candidates, rng, funnel, cap_of, family_share, lambda c: 0, trained, rank=rank), rng,
                          funnel, is_wrapper, wrapper_share, trained, rank=rank)
    if tiers is None:
        return pool, set(), set()
    own_tasks = {c['task_id'] for c in pool if c['source'] == 'rl-own'}
    return pool, own_tasks, {c['task_id'] for c in pool if c['source'] == 'jolt-guided'} - own_tasks


def check_render(args):
    """Render recorded student calls and compare with the actor's exact prompt and sampled ids."""
    renderer = TemplateRenderer(args.tokenizer)
    counts = collections.Counter()
    examples = []
    for episode in all_episodes(args.work):
        if episode['source'] not in OWN_SOURCES:
            continue
        for turn in episode['turns']:
            exact = turn.get('exact')
            if not exact:
                counts['no_exact'] += 1
                continue
            try:
                prompt, target = renderer.render(turn['messages'], turn['tools'], turn['target'])
            except Exception as error:
                counts['render_error'] += 1
                examples.append({'key': episode['key'], 'error': repr(error)[:300]})
                continue
            counts['turns'] += 1
            same_prompt = prompt == exact['prompt_ids']
            counts['prompt_equal'] += same_prompt
            # a later call's history holds the earlier replies as sampled, which may tokenize differently from the text
            counts['prompt_text_equal'] += (renderer.tokenizer.decode(prompt)
                                            == renderer.tokenizer.decode(exact['prompt_ids']))
            sample = exact['sample_ids']
            counts['target_equal'] += target == sample
            # a capture may end at a stop without the end token, or the reply may tokenize differently from the sample
            counts['target_text_equal'] += (renderer.tokenizer.decode(target).rstrip('<|im_end|>').rstrip()
                                            == renderer.tokenizer.decode(sample).rstrip('<|im_end|>').rstrip())
            if not same_prompt and len(examples) < 20:
                first = next((i for i, (a, b) in enumerate(zip(prompt, exact['prompt_ids'])) if a != b),
                             min(len(prompt), len(exact['prompt_ids'])))
                examples.append({'key': episode['key'], 'len': [len(prompt), len(exact['prompt_ids'])], 'first': first,
                                 'rendered': renderer.tokenizer.decode(prompt[max(0, first - 20):first + 40]),
                                 'recorded': renderer.tokenizer.decode(exact['prompt_ids'][max(0, first - 20):first + 40])})
            if counts['turns'] >= args.turns:
                break
        if counts['turns'] >= args.turns:
            break
    report = {'counts': dict(counts), 'identity': renderer.identity, 'examples': examples}
    Path(args.work, 'render-check.json').write_text(json.dumps(report, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps(report['counts']))


# ---- replay ----------------------------------------------------------------------------------------------------------
def arguments_key(arguments):
    """Tool-call arguments as compact sorted JSON: the runtime echoes a recorded call's arguments re-serialized
    (`{"code":"..."}` for a teacher's `{"code": "..."}`, 98 of 100 October Qwen replays missed on that alone), and
    the same arguments are the same action. Unparseable arguments stay as they are."""
    if not isinstance(arguments, str):
        return arguments
    try:
        return json.dumps(json.loads(arguments), ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    except ValueError:
        return arguments


def visible_key(messages):
    """What a call saw, minus the runtime's own wording: every non-system message's role, text, tool calls (name and
    arguments) and tool output. Tool-call ids and reasoning are left out (the runtime echoes them as it received them)."""
    out = []
    for m in messages:
        if m.get('role') == 'system':
            continue
        calls = [((c.get('function') or {}).get('name'), arguments_key((c.get('function') or {}).get('arguments')))
                 for c in m.get('tool_calls') or []]
        content = m.get('content')
        if isinstance(content, list):   # content parts: their text in order
            content = ''.join(part.get('text', '') if isinstance(part, dict) else str(part) for part in content)
        out.append((m.get('role'), content or '', calls))
    return json.dumps(out, ensure_ascii=False)


def reworded_key(messages):
    """``visible_key`` with the invocation's opening message cut to its first line (the call's signature): the runtime
    rewords the rest (call notes, protocol sentences; e.g. 1cb19e291 made return_result the only way to answer), and a
    recorded call whose every action, tool output and later message still matches is the same decision under today's
    wording."""
    out, opening = [], True
    for role, content, calls in json.loads(visible_key(messages)):
        if opening and role == 'user':
            content, opening = content.split('\n', 1)[0], False
        out.append((role, content, calls))
    return json.dumps(out, ensure_ascii=False)


def opening_key(messages):
    """The call's invocation: its first non-system message (the task or nl() request)."""
    for m in messages:
        if m.get('role') != 'system':
            return json.dumps([m.get('role'), m.get('content')], ensure_ascii=False)
    return ''


def signature_key(messages):
    """``opening_key`` cut to the opening message's first line (the call's signature), as ``reworded_key`` cuts it."""
    for m in messages:
        if m.get('role') != 'system':
            content = m.get('content')
            return json.dumps([m.get('role'), content.split('\n', 1)[0] if isinstance(content, str) else content],
                              ensure_ascii=False)
    return ''


class ReplayEpisode:
    """The recorded calls of one episode, served once each: an exact visible-history match first; otherwise the next
    unserved call of the same invocation with as many visible messages (a tool output that differs from the recording,
    counted as 'drifted'); otherwise no reply (the episode ends)."""

    def __init__(self, turns):
        import threading
        self.lock = threading.Lock()
        self.turns = turns
        self.keys = [visible_key(t['messages']) for t in turns]
        self.reworded = [reworded_key(t['messages']) for t in turns]
        self.openings = [opening_key(t['messages']) for t in turns]
        self.signatures = [signature_key(t['messages']) for t in turns]
        self.lengths = [sum(m.get('role') != 'system' for m in t['messages']) for t in turns]
        self.served = [False] * len(turns)
        self.log = []

    def closest(self, candidates, opening):
        """The candidate whose recorded opening is most like the request's: sibling child calls (one nl() per item)
        share a signature, so the first unserved one could be another item's call."""
        if len(candidates) < 2:
            return candidates[0] if candidates else None
        import difflib
        return max(candidates, key=lambda i: (difflib.SequenceMatcher(None, self.openings[i], opening).ratio(), -i))

    def field_reply_first(self, matches, messages):
        """Of the unserved recorded calls with the request's exact visible history, the one to serve: for a field-mode
        readout (``decision_graph.decision_forms``) whose steps were recorded as calls with the same messages (each its
        open step text), the call whose reply is the whole record (``field_path``), the step calls served with it: the
        runtime asks a declined readout once, by generation, for the whole value. Otherwise the first."""
        from natlang_neuralese.data.decision_graph import decision_forms, field_path
        forms = decision_forms(messages) if len(matches) > 1 else None
        if forms is None:
            return matches[0] if matches else None
        reply = lambda i: ((self.turns[i].get('target') or {}).get('content') or '').strip()
        whole = [i for i in matches if field_path(forms, reply(i)) is not None]
        if not whole:
            return matches[0]
        for i in matches:
            if i != whole[0] and field_path(forms, reply(i)) is None:
                self.served[i] = True
        return whole[0]

    def answer(self, body):
        if 'items' in body and 'messages' not in body:
            # a /v1/natlang/score request (ts-host/src/model/decision-scoring.ts): declined as below, item by item
            with self.lock:
                self.log.append({'kind': 'scoring-declined', 'turn': None, 'items': len(body.get('items') or [])})
            return {'results': [{'error': 'decision-unsupported: the replay answers readouts by generation'}
                                for _ in body.get('items') or []]}
        if 'prompt_logprobs' in body:
            # A decision readout's scoring request (ts-host/src/model/decision-scoring.ts). The recorded teacher could
            # not score replies, so its run declined the readout and ran the call's tool loop: answer as it did, with no
            # prompt_logprobs (the runtime reads that as decision-unsupported), and the tool-loop turns replay after it.
            with self.lock:
                self.log.append({'kind': 'scoring-declined', 'turn': None, 'messages': len(body.get('messages') or [])})
            return {'id': 'replay-unscored', 'object': 'chat.completion', 'model': body.get('model', 'replay'),
                    'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': ''}, 'finish_reason': 'length'}],
                    'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}}
        messages = body.get('messages') or []
        key, opening = visible_key(messages), opening_key(messages)
        length = sum(m.get('role') != 'system' for m in messages)
        with self.lock:
            match = self.field_reply_first([i for i, k in enumerate(self.keys) if not self.served[i] and k == key],
                                           messages)
            kind = 'exact' if match is not None else 'missing'
            if match is None:
                reworded = reworded_key(messages)
                match = self.closest([i for i, k in enumerate(self.reworded) if not self.served[i] and k == reworded],
                                     opening)
                kind = 'reworded' if match is not None else kind
            if match is None:
                # the same invocation (its opening, else its signature when the runtime reworded the opening) with
                # as many visible messages: a tool output that differs from the recording
                signature = signature_key(messages)
                for keys, want in ((self.openings, opening), (self.signatures, signature)):
                    match = self.closest([i for i in range(len(self.turns))
                                          if not self.served[i] and keys[i] == want and self.lengths[i] == length],
                                         opening)
                    if match is not None:
                        kind = 'drifted'
                        break
            entry = {'kind': kind, 'turn': match, 'messages': length}
            if match is None:   # what the runtime asked that no recorded call saw (its last two visible messages)
                entry['now'] = str(json.loads(key)[-2:])[:1200]
            if match is not None:
                self.served[match] = True
                recorded = self.turns[match]
                system = [m.get('content') for m in messages if m.get('role') == 'system']
                recorded_system = [m.get('content') for m in recorded['messages'] if m.get('role') == 'system']
                entry['system_same'] = system == recorded_system
                entry['tools_same'] = (json.dumps(body.get('tools') or [], sort_keys=True)
                                       == json.dumps(recorded.get('tools') or [], sort_keys=True))
                if kind == 'drifted':
                    recorded_visible = json.loads(self.keys[match])
                    now_visible = json.loads(key)
                    first = next((j for j, (a, b) in enumerate(zip(recorded_visible, now_visible)) if a != b), None)
                    entry['first_difference'] = first
                    if first is not None:
                        entry['recorded'] = str(recorded_visible[first])[:600]
                        entry['now'] = str(now_visible[first])[:600]
            self.log.append(entry)
        if match is None:
            return None
        target = {k: v for k, v in self.turns[match]['target'].items() if k != 'role' and v is not None}
        message = {'role': 'assistant', 'content': target.pop('content', '') or '', **target}
        tokens = self.turns[match].get('completion_tokens') or 0
        return {'id': f'replay-{match}', 'object': 'chat.completion', 'model': body.get('model', 'replay'),
                'choices': [{'index': 0, 'message': message,
                             'finish_reason': 'tool_calls' if message.get('tool_calls') else 'stop'}],
                'usage': {'prompt_tokens': 0, 'completion_tokens': tokens, 'total_tokens': tokens}}


class ReplayServer:
    """OpenAI-compatible chat endpoint on 127.0.0.1: POST /ep/<episode>/v1/chat/completions answers from that episode's
    recording (non-streaming JSON; the collector accepts it for a streamed request)."""

    def __init__(self):
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        self.episodes = {}
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self._send(200, {'object': 'list', 'data': [{'id': 'replay', 'object': 'model'}]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get('Content-Length') or 0)) or b'{}')
                parts = self.path.strip('/').split('/')
                episode = server.episodes.get(parts[1]) if len(parts) > 2 and parts[0] == 'ep' else None
                reply = episode.answer(body) if episode is not None else None
                if reply is None:
                    self._send(500, {'error': {'message': 'replay: the recording has no reply for this request',
                                               'type': 'replay_missing'}})
                else:
                    self._send(200, reply)

            def _send(self, status, payload):
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()


def safe_name(key):
    return hashlib.sha256(key.encode()).hexdigest()[:20]


def has_native_state(episode):
    return any(isinstance(m.get('content'), list) and any(isinstance(part, dict) and part.get('type') == 'native_state'
                                                           for part in m['content'])
               for turn in episode['turns'] for m in turn['messages'])


def replay_episode(args, server, episode, env):
    """Run one recorded episode's task under this checkout's runtime against its recording; returns its replay record."""
    import shutil
    import subprocess
    import time
    if episode.get('verification_mode') == 'observed-live':
        return {'key': episode['key'], 'outcome': 'observed_live_not_portable', 'exact': False,
                'detail': 'Live workspace/CDP effects require a test-owned live rerun; no portable replay claimed.'}
    from types import SimpleNamespace
    from natlang_neuralese.common.jsonio import read_jsonl
    from natlang_neuralese.rl import task_rollouts
    name = safe_name(episode['key'])
    directory = Path(args.work) / 'replay' / name
    done = directory / 'replay.json'
    if done.exists():
        return json.loads(done.read_text())
    if has_native_state(episode):   # its imported states came from the actor's store: no text endpoint can replay them
        return {'key': episode['key'], 'name': name, 'source': episode['source'], 'task_id': episode['task_id'],
                'outcome': 'skipped_native_state', 'exact': False, 'recorded_turns': len(episode['turns'])}
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    task, memetic = episode['task'], is_memetic(episode['task'])
    if memetic:
        task, refusal = memetic_replay_task(task, directory)
        if refusal:
            record = {'key': episode['key'], 'name': name, 'source': episode['source'], 'task_id': episode['task_id'],
                      'outcome': refusal, 'exact': False, 'recorded_turns': len(episode['turns'])}
            shutil.rmtree(directory)
            directory.mkdir(parents=True)
            done.write_text(json.dumps(record) + '\n')
            return record
    (directory / 'cases.jsonl').write_text(json.dumps(task) + '\n')
    task_rollouts.prepare(SimpleNamespace(
        cases=directory / 'cases.jsonl', out=directory, rows=1, offset=0, group=2, temperature=1.0,
        case_seconds=args.case_seconds,
        # a memetic supervisor episode ran under grpo's --memetic-model-requests: its recorded calls all replay
        max_model_requests=max(args.max_model_requests, len(episode['turns'])) if memetic else args.max_model_requests,
        max_turns=128,
        seed_base=0, root_mode=episode.get('root_mode', 'student'), match_seeds_from=None, policy_checkpoint_sha256=None))
    entry = list(read_jsonl(directory / 'queue.jsonl'))[0]
    entry.update(seed=episode.get('seed') or entry['seed'], prompt_rendering='full', request_retries=0,
                 context_tokens=replay_context_tokens(episode['source']))
    (directory / 'queue.jsonl').write_text(json.dumps(entry, sort_keys=True) + '\n')
    queue_file = directory / f"{entry['key']}.queue.jsonl"
    queue_file.write_text(json.dumps(entry, sort_keys=True) + '\n')
    (directory / 'chat-config.json').write_text('{}\n')
    replay = ReplayEpisode(episode['turns'])
    server.episodes[name] = replay
    url = f'http://127.0.0.1:{server.port}/ep/{name}/v1/chat/completions'
    command = [sys.executable, str(Path(args.runtime_repo) / 'scripts/run_bonsai_queue.py'), str(queue_file),
               str(directory / f"{entry['key']}.journal.jsonl"), '--runtime', str(Path(args.runtime_repo) / 'ts-host'),
               '--model-id', 'replay', '--model-concurrency', '1', '--server', f'http://127.0.0.1:{server.port}',
               '--chat-completions-url', url, '--chat-request-config', str(directory / 'chat-config.json')]
    started = time.time()
    with (directory / 'collector.log').open('w') as handle:
        result = subprocess.run(command, cwd=args.runtime_repo, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                timeout=args.case_seconds + 300)
    del server.episodes[name]
    with (directory / 'journal.jsonl').open('w') as merged:
        merged.write(Path(directory / f"{entry['key']}.journal.jsonl").read_text()
                     if (directory / f"{entry['key']}.journal.jsonl").exists() else '')
    outcome, turns, memetic_outcome = None, 0, None
    try:
        task_rollouts.score(directory, directory / 'rollouts.jsonl', neuralese_bonus=0., efficiency_weight=0.)
        rows = list(read_jsonl(directory / 'rollouts.jsonl'))
        if rows:
            outcome, turns = rows[0].get('outcome'), len(rows[0].get('turns') or [])
            if outcome == MEMETIC:   # verified again from the recorded evaluations: a success replays as one
                memetic_outcome = {'memetic_success': bool(rows[0].get('memetic_success')),
                                   'quality': (rows[0].get('reward') or {}).get('quality')}
                outcome = SUCCESS if episode_success(rows[0]) else 'memetic_not_improved'
    except Exception as error:   # a scoring failure is this episode's verdict, never the whole replay's
        outcome = 'score_error: ' + repr(error)[:300]
    kinds = collections.Counter(e['kind'] for e in replay.log)
    record = {'key': episode['key'], 'name': name, 'source': episode['source'], 'task_id': episode['task_id'],
              'outcome': outcome, 'returncode': result.returncode, 'seconds': round(time.time() - started, 1),
              'recorded_turns': len(episode['turns']), 'replayed_turns': turns,
              'served': dict(kinds), 'unserved': replay.served.count(False),
              'siblings_resolved': True,   # reworded/drifted matches chose the closest opening (ReplayEpisode.closest)
              'system_changed': sum(1 for e in replay.log if e.get('system_same') is False),
              'tools_changed': sum(1 for e in replay.log if e.get('tools_same') is False),
              'log': replay.log[:64],
              **({'memetic': memetic_outcome} if memetic else {})}
    # exact: every recorded call served to the same visible history, its opening message at most reworded by the runtime
    matched = {k: v for k, v in kinds.items() if k != 'scoring-declined'}
    record['exact'] = (outcome == SUCCESS and record['unserved'] == 0 and set(matched) <= {'exact', 'reworded'}
                       and sum(matched.values()) == len(episode['turns']) and turns == len(episode['turns']))
    for path in directory.iterdir():   # keep the verdict, the replayed record and the logs; drop the job scratch
        if path.is_dir():
            shutil.rmtree(path)
    if episode['source'] in recorded_turns.OWN_SOURCES:   # an own success renders as sampled (recorded_turns.py)
        recorded_turns.write(args.work, name, episode['turns'])
    done.write_text(json.dumps(record) + '\n')
    return record


def is_memetic(task):
    return ((task.get('semantics') or {}).get('world') or {}).get('kind') == 'memetic'


def memetic_replay_task(task, directory):
    """(task row, refusal) for replaying a memetic supervisor episode in ``directory``: the group's episode spec copied
    with ``recorded_only`` (every evaluation is read from the collection's recorded results, prepared evidence included,
    by the same names; one the recording lacks fails the replay instead of sampling a model) and its out under the
    replay. The collection's group directory is read only. Refused: a spec that is gone, or one that refines guidance
    states (a GPU fit the replay cannot repeat)."""
    import copy
    import hashlib
    import shutil
    world = task['semantics']['world']
    spec_path = Path(world['task'])
    if not spec_path.exists():
        return task, 'memetic_spec_missing'
    if hashlib.sha256(spec_path.read_bytes()).hexdigest() != world.get('spec_sha256', ''):
        return task, 'memetic_spec_changed'
    spec = json.loads(spec_path.read_text())
    if spec.get('refine'):
        return task, 'memetic_refined_not_replayable'
    recorded = Path(spec['out'])
    if not (recorded / 'prepared.json').exists():
        return task, 'memetic_not_prepared'
    out = directory / 'memetic'
    out.mkdir(parents=True)
    shutil.copyfile(recorded / 'prepared.json', out / 'prepared.json')
    for summary in sorted((recorded / 'evals').glob('*/summary.json')):
        (out / 'evals' / summary.parent.name).mkdir(parents=True)
        shutil.copyfile(summary, out / 'evals' / summary.parent.name / 'summary.json')
    spec.update(out=str(out), recorded_only=True)
    path = directory / 'memetic-spec.json'
    path.write_text(json.dumps(spec, ensure_ascii=False, sort_keys=True) + '\n')
    task = copy.deepcopy(task)
    task['semantics']['world'] = {**world, 'task': str(path), 'spec_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    return task, None


def replay(args):
    """Replay the extracted episodes that have no verdict in WORK/replay.jsonl yet: a periodic re-extraction of a growing
    teacher pool replays only its new episodes and appends each verdict once."""
    path = Path(args.work) / 'replay.jsonl'
    done = {json.loads(line).get('key') for line in path.open() if line.strip()} if path.exists() else set()
    return replay_episodes(args, (e for e in all_episodes(Path(args.work)) if e['key'] not in done))


def replay_episodes(args, episodes):
    """Replay canonical episodes, including staged sources awaiting corpus admission."""
    import concurrent.futures
    import threading
    work = Path(args.work)
    args.runtime_repo = str(Path(args.runtime_repo or REPO).resolve())
    from natlang_neuralese.common.code_provenance import code_provenance
    from natlang_neuralese.rl.held_eval import EPISODE_MODEL_CALLS, series
    args.max_model_requests = args.max_model_requests or EPISODE_MODEL_CALLS
    env = {**os.environ, 'NATLANG_REPO': args.runtime_repo,
           'PYTHONPATH': f'{args.runtime_repo}/training/neuralese:{args.runtime_repo}',
           'NATLANG_MODEL_FACING': series()['model_facing'], 'NATLANG_COLLECTOR_SHARED': '0',
           'NATLANG_MEMETIC_PYTHON': sys.executable}   # a memetic episode's workspace (world-bridge.ts kind memetic)
    server = ReplayServer()
    if args.limit:
        import itertools
        episodes = itertools.islice(episodes, args.limit)
    lock = threading.Lock()
    counts = collections.Counter()
    (work / 'replay').mkdir(exist_ok=True)
    with (work / 'replay-launches.jsonl').open('a') as handle:
        handle.write(json.dumps({'argv': sys.argv, 'runtime_repo': args.runtime_repo, 'code': code_provenance(),
                                 'model_facing': env['NATLANG_MODEL_FACING']}) + '\n')
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool, (work / 'replay.jsonl').open('a') as results:
        pending = set()

        def finished(future):
            try:
                record = future.result()
            except Exception as error:
                record = {'outcome': 'replay_error: ' + repr(error)[:300]}
            with lock:
                counts[str(record.get('outcome'))[:40]] += 1
                results.write(json.dumps({k: v for k, v in record.items() if k != 'log'}) + '\n')
                results.flush()
                if sum(counts.values()) % 50 == 0:
                    print(json.dumps(dict(counts)), flush=True)
        for episode in episodes:
            while len(pending) >= args.workers * 2:
                done, pending = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
            future = pool.submit(replay_episode, args, server, episode, env)
            future.add_done_callback(finished)
            pending.add(future)
        concurrent.futures.wait(pending)
    record_work(work, [])
    print(json.dumps(dict(counts)), flush=True)


# ---- build -----------------------------------------------------------------------------------------------------------
WORD = None


def shingles(text, n=5):
    import re
    global WORD
    WORD = WORD or re.compile(r'\w+')
    words = [w.lower() for w in WORD.findall(text)]
    words = ['0' if w.isdigit() else w for w in words]
    return {' '.join(words[i:i + n]) for i in range(max(1, len(words) - n + 1))} if words else set()


def minhash(items, perms=64):
    """64-permutation MinHash of a shingle set (the DATA_MIX_DIVERSITY.md method)."""
    import numpy as np
    if not items:
        return None
    hashes = np.array([int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), 'little') for s in items],
                      dtype=np.uint64)
    rng = np.random.RandomState(20261117)
    a = rng.randint(1, 2**31 - 1, size=perms).astype(np.uint64)
    b = rng.randint(0, 2**31 - 1, size=perms).astype(np.uint64)
    return ((hashes[None, :] * a[:, None] + b[:, None]) % np.uint64(2**61 - 1)).min(axis=1)


class NearDuplicates:
    """LSH (16 bands of 4) over MinHash signatures; `seen(sig)` is true when an admitted item has estimated Jaccard >=
    threshold, else the item is admitted."""

    def __init__(self, threshold=0.8, bands=16, rows=4):
        self.threshold, self.bands, self.rows = threshold, bands, rows
        self.buckets = collections.defaultdict(list)
        self.signatures = []

    def seen(self, signature):
        if signature is None:
            return False
        candidates = set()
        for band in range(self.bands):
            key = (band, signature[band * self.rows:(band + 1) * self.rows].tobytes())
            candidates.update(self.buckets.get(key, ()))
        for index in candidates:
            if (self.signatures[index] == signature).mean() >= self.threshold:
                return True
        index = len(self.signatures)
        self.signatures.append(signature)
        for band in range(self.bands):
            self.buckets[(band, signature[band * self.rows:(band + 1) * self.rows].tobytes())].append(index)
        return False


def action_text(turns):
    """An episode's own outputs as text: replies, tool calls (name and arguments) and reasoning."""
    parts = []
    for turn in turns:
        target = turn['target']
        parts.append(target.get('reasoning_content') or '')
        parts.append(target.get('content') or '')
        for call in target.get('tool_calls') or []:
            parts.append(f"{call['function']['name']} {call['function']['arguments']}")
    return '\n'.join(parts)


class HeldClosure(tuple):
    """(ids, groups, source ids) of the held sets, with ``reservations``: held sources reserved by rule
    (``held_eval.reserved``; held-v4's reserved briefs, apps and fixture-pack shares), which ``held_hit`` also applies."""
    reservations = ()


def _held_closure_one(manifest_path):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    from natlang_neuralese.rl.held_eval import RESERVATIONS_SCHEMA
    if manifest.get('schema') == RESERVATIONS_SCHEMA:
        return set(), set(), set(), [manifest]
    ids, groups, sources, reservations = {t['id'] for t in manifest['tasks']}, set(manifest['excluded_split_groups']), set(), []
    cases = manifest_path.parent / manifest['cases']['path']
    if sha256_file(cases) != manifest['cases']['sha256']:
        raise ValueError(f'held cases changed: {cases}')
    for line in cases.open():
        if line.strip():
            row = json.loads(line)
            ids.add(row['id'])
            groups.update(row.get('source_groups') or ())
            groups.add((row.get('curriculum') or {}).get('split_group') or row['id'])
            sources.update(row.get('source_ids') or ())
    for earlier in manifest.get('also_excludes') or ():
        if sha256_file(earlier['path']) != earlier['sha256']:
            raise ValueError('an excluded earlier held manifest changed: ' + earlier['path'])
        more = _held_closure_one(earlier['path'])
        ids |= more[0]
        groups |= more[1]
        sources |= more[2]
        reservations += more[3]
    return ids, groups, sources, reservations


def held_closure(manifest_path):
    """(ids, groups, source ids) of the held set, every earlier held set it excludes and every other ``held-eval*``
    registry role (held-v4 and its reservations: ``held_eval.held_manifests``): task ids, split groups, the held rows'
    source groups and source ids (the separation grpo.load_tasks applies, for rows outside the pool too)."""
    from natlang_neuralese.rl.held_eval import held_manifests
    ids, groups, sources, reservations, seen = set(), set(), set(), [], set()
    for path in [manifest_path, *[p for _, p in held_manifests()]]:
        if path is None or os.path.realpath(path) in seen:
            continue
        seen.add(os.path.realpath(path))
        more = _held_closure_one(path)
        ids |= more[0]
        groups |= more[1]
        sources |= more[2]
        reservations += more[3]
    groups.discard(None)
    closure = HeldClosure((ids, groups, sources))
    closure.reservations = tuple(reservations)
    return closure


def live_holdout_registry(path, held):
    """Add the predeclared Vibestudio holdouts to the learner's source-group closure."""
    group_splits, goal_splits, training = {}, {}, {}
    task_ids = set()
    test_cases = 0
    ids, groups, sources = held
    for line in zstd_lines(path):
        row = json.loads(line)
        if row.get('interaction') != 'open-user-goal':
            continue
        task_id, split, relatives = row.get('taskId'), row.get('split'), row.get('sourceGroups')
        digest = row.get('goalSha256')
        if (not isinstance(task_id, str) or not task_id.startswith('vibestudio:') or task_id in task_ids or
                split not in ('train', 'validation', 'test') or not isinstance(relatives, list) or not relatives or
                not all(isinstance(group, str) and group.strip() for group in relatives) or
                not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest)):
            raise ValueError('invalid or duplicate open-goal holdout registry row')
        task_ids.add(task_id)
        if goal_splits.setdefault(digest, split) != split:
            raise ValueError('open-goal prompt crosses splits')
        for group in relatives:
            if group_splits.setdefault(group, split) != split:
                raise ValueError(f'open-goal source group crosses splits: {group}')
        if split == 'train':
            training[task_id] = (frozenset(relatives), digest)
        else:
            test_cases += 1
            ids.add(task_id)
            sources.add(task_id)
            groups.update(relatives)
    if not training or not test_cases:
        raise ValueError('open-goal registry needs both train and held cases')
    return training


def require_registered_open_goal(candidate, training):
    from live_teacher_sources import open_goal_fingerprint
    task_id = candidate['task_id']
    _, digest = open_goal_fingerprint(candidate['open_goal'])
    if training.get(task_id) != (frozenset(candidate['source_groups']), digest):
        raise ValueError(f'open goal {task_id} differs from the pinned training registry')


CHARS_PER_TOKEN = 3.3   # estimate for selecting October rows before rendering (calibrated in build.json)
DEFAULT_OCTOBER_GROUPS = ('luna-curricula=3', 'qwen-s72=2', 'fallow-v13r2=2', 'qwen-campaign=1', 'luna-early-cohorts=1',
                          'luna-v19-decision-children=1', 'luna-clinc-children=1', 'luna-ep-campaigns=1',
                          'student-projection=1', 'acq-swe-rebench-openhands=2', 'acq-d32-openresearcher=2',
                          'acq-d13-cooperbench=1')


def held_hit(task_id, source_ids, groups, held):
    held_ids, held_groups, held_sources = held
    if task_id in held_ids or set(groups) & held_groups or set(source_ids) & held_sources:
        return True
    reservations = getattr(held, 'reservations', ())
    if reservations:
        from natlang_neuralese.rl.held_eval import reserved
        row = {'id': task_id, 'source_groups': list(groups), 'source_ids': list(source_ids)}
        return any(reserved(r, row) for r in reservations)
    return False


def water_fill(sizes, weights, budget):
    """Per-group budgets: shares of `budget` by weight; a group with less than its share gives the rest to the others."""
    budgets, open_groups, left = {}, {g for g in sizes if weights.get(g, 0) > 0}, budget
    while open_groups:
        total = sum(weights[g] for g in open_groups)
        small = {g for g in open_groups if sizes[g] <= left * weights[g] / total}
        if not small:
            for g in open_groups:
                budgets[g] = left * weights[g] / total
            break
        for g in small:
            budgets[g] = sizes[g]
            left -= sizes[g]
        open_groups -= small
    return budgets


DRIFTED_DEFAULT = True   # teacher replays whose display drifted are admitted by default (consumers' --admit-drifted off)


def sibling_ambiguous(turns):
    """Whether two recorded calls share a signature and a visible length but not their opening (sibling child calls,
    e.g. one nl() per item): a replay matched without ``siblings_resolved`` may have served one sibling's recorded reply
    to another's request, so a drifted success of such an episode is not admitted."""
    seen = collections.defaultdict(set)
    for t in turns:
        seen[(signature_key(t['messages']), sum(m.get('role') != 'system' for m in t['messages']))].add(
            opening_key(t['messages']))
    return any(len(v) > 1 for v in seen.values())


def drifted_success(record, turns=None, ambiguous=None):
    """A replay that succeeded with every recorded call served and nothing else asked, but where some call saw a
    display that differs from the recording (``drifted``: a tool output or host preamble rendered differently today).
    The actions are the teacher's and the oracle accepted them under today's runtime; the training rows render the
    replayed (today's) prompts (``replayed_turns``). With the episode's recorded ``turns``, an episode whose sibling calls
    a replay without ``siblings_resolved`` (the matcher before its best-opening choice) could have swapped is refused."""
    served = {k: v for k, v in (record.get('served') or {}).items() if k != 'scoring-declined'}
    if ambiguous is None:     # ``ambiguous``: sibling_ambiguous(turns) computed already (``episode_facts``)
        ambiguous = turns is not None and sibling_ambiguous(turns)
    if ambiguous and not record.get('siblings_resolved'):
        return False
    return (record.get('outcome') == SUCCESS and not record.get('exact') and record.get('unserved') == 0
            and 'missing' not in served and set(served) <= {'exact', 'reworded', 'drifted'}
            and record.get('replayed_turns') == record.get('recorded_turns'))


def verified_episodes(work, funnel, extracted=None, sources=None, drifted=DRIFTED_DEFAULT, workers=1, light=False):
    """(episode, replay record) of every extracted success of WORK (of ``sources`` only, when given) whose replay under
    the replaying runtime was exact and succeeded (fixture tasks still admitted); counted per source in ``funnel``.
    ``extracted`` (a set) collects every extracted key. ``drifted`` (default on): also teacher successes whose display drifted
    (``drifted_success``; counted as ``replayed_drifted``; ``drifted_sibling_ambiguous`` the ones refused for a possible
    sibling swap)."""
    work = Path(work)
    replays = {}
    replay_path = work / 'replay.jsonl'
    for line in replay_path.open() if replay_path.exists() else []:
        if line.strip():
            record = json.loads(line)
            if 'key' in record:
                replays[record['key']] = record
    from natlang_neuralese.common.hashing import sha256_file_hex_cached
    admission_cache, admission_digests = {}, {}
    for episode in all_episodes(work, workers, light):
        if extracted is not None:
            extracted.add(episode['key'])
        source = episode['source']
        if sources is not None and source not in sources:
            continue
        f = funnel[source]
        f['successes'] += 1
        if episode.get('verification_mode') == 'observed-live':
            from live_teacher_sources import load_episode
            binding = episode.get('live_source') or {}
            if not binding.get('path') or sha256_file(binding['path']) != binding.get('sha256'):
                raise ValueError('observed live source changed after extraction')
            current = load_episode(binding['path'])
            if current != episode:
                raise ValueError('extracted live episode differs from its original captured records')
            f['observed_live_verified'] += 1
            yield episode, {'name': None, 'verification_mode': 'observed-live', 'live_source': binding}
            continue
        from teacher_admission import check_record, fixture_task
        if fixture_task(episode['task']):
            admission = episode.get('admission') or {}
            path = admission.get('path')
            if path and path not in admission_digests:   # once per file (hashed per episode it read the HDD for minutes)
                admission_digests[path] = sha256_file_hex_cached(path, trust_archive_receipt=False)
            if not path or admission_digests[path] != admission.get('sha256'):
                raise ValueError('fixture source admission missing or changed after extraction')
            cache_key = path, admission['sha256']
            if cache_key not in admission_cache:
                admission_cache[cache_key] = check_record(admission['manifest_record'], admission['pool'])
            if (episode['task_id'], episode['sample']) not in admission_cache[cache_key]:
                raise ValueError('fixture source is no longer admitted')
        record = replays.get(episode['key'])
        if record is None:
            f['not_replayed'] += 1
            continue
        if record['outcome'] == 'skipped_native_state':
            f['native_state'] += 1
            continue
        if record['outcome'] != SUCCESS:
            f['replay_failed'] += 1
            continue
        if not record.get('exact'):
            drifted_here = drifted and source == 'teacher'   # own episodes keep exact replay (their sampled ids)
            if drifted_here and drifted_success(record, episode.get('turns'),
                                                (episode.get('facts') or {}).get('sibling_ambiguous')):
                f['replayed_drifted'] += 1
                yield episode, record
                continue
            if drifted_here and drifted_success(record):
                f['drifted_sibling_ambiguous'] += 1
                continue
            f['replay_inexact'] += 1
            continue
        f['replayed_exact'] += 1
        yield episode, record


BACKEND_WORKAROUND = 'neuralese-unsupported-backend'


def backend_workaround(turns):
    """Whether recorded turns ran into the text-only backend's Neuralese refusal (teacher_gen serves teachers through
    text endpoints): the actions after it work around an error the student's actor (which serves Neuralese) never
    raises, a detour off the student's surface (8% of v4's teacher episodes; 92% of composed-Neuralese and 70% of bgkit
    teacher successes of pool-v1 + pool-v2-rich). The action that met it (e.g. a producer's return_result) is the
    student's own surface. The refusal counts anywhere in a tool message: a child's refusal reaches its parent wrapped
    in the error text (\"... failed with \"neuralese-unsupported-backend: ...\"\"); matching only a message that starts
    with it left 1,070 parent detour calls (3.2% of sft-warmup-20261118-v1's loss) trained."""
    for turn in turns:
        for message in turn.get('messages') or ():
            content = message.get('content')
            if message.get('role') == 'tool' and isinstance(content, str) and BACKEND_WORKAROUND in content:
                return True
    return False


def mark_detour_spawned(turns):
    """Mark (``turn['detour_spawned']``) every call of an invocation that first appears after the episode's first
    Neuralese refusal (``backend_workaround``): the children the detour started (ad-hoc ``nl`` calls, a function
    re-invoked to work around the refusal). Their own prompts hold no refusal, yet they exist only because of the
    detour, which the student's actor never takes: 3,238 such teacher calls trained in sft-mix-20261120-v1 (DECISIONS
    2026-11-21 ran-out). Invocations already running before the refusal keep their calls. Returns ``turns``."""
    before, refused = set(), False
    for turn in turns:
        invocation = turn.get('invocation_id')
        if not refused and backend_workaround([turn]):
            refused = True
        if not refused:
            before.add(invocation)
        elif invocation not in before:
            turn['detour_spawned'] = True
    return turns


def detour_call(turn, backend_workarounds):
    """Whether ``call_row`` weights a non-student call as context under ``backend_workarounds``: its prompt holds the
    refusal (``context`` and ``context-refused-only``) or the detour spawned it (``context`` only)."""
    if backend_workarounds not in ('context', 'context-refused-only'):
        return False
    return backend_workaround([turn]) or (backend_workarounds == 'context' and bool(turn.get('detour_spawned')))


def held_excluded_ids(held_manifest, tasks, held):
    """Task ids among ``tasks`` ({id: task row}) in the held closure: ``held_eval.exclusion_ids`` (held tasks and their
    split groups) plus every row sharing a source group or source id with a held row (``held``: ``held_closure``)."""
    from natlang_neuralese.rl.held_eval import training_exclusion_ids
    excluded = set(training_exclusion_ids(list(tasks.values()), held_manifest))
    for task_id, task in tasks.items():
        groups = list(task.get('source_groups') or ()) + [(task.get('curriculum') or {}).get('split_group') or task['id']]
        if held_hit(task_id, task.get('source_ids') or (), groups, held):
            excluded.add(task_id)
        # a memetic row's support and query worlds (memetic_tasks.episode_case): none may be a held task
        if set((task.get('curriculum') or {}).get('world_ids') or ()) & held[0]:
            excluded.add(task_id)
    return excluded


def core_candidates(args, work, funnel, held):
    """Replay-verified successes (exact replay, current oracle success), held closure removed: metadata and signatures
    only (the turns are read again when rendering)."""
    candidates, tasks = [], {}
    extracted = set()
    for episode, record in verified_episodes(work, funnel, extracted, drifted=getattr(args, 'admit_drifted', 'on') == 'on',
                                             workers=getattr(args, 'workers', None) or 1, light=True):
        source = episode['source']
        if episode.get('verification_mode') == 'observed-live' and not episode['turns']:
            funnel[source]['observed_evidence_without_supervised_targets'] += 1
            continue  # Keep the native trajectory in the corpus, without reserving a learner task slot.
        if source == 'rl-own' and episode.get('role') == 'teacher':
            source = 'jolt-guided'
        facts = episode.get('facts') or episode_facts(episode['turns'])
        task = episode['task']
        tasks[episode['task_id']] = task
        ep_policies = list(episode.get('policies') or ([episode['policy']] if episode.get('policy') else []))
        candidates.append({'key': episode['key'], 'source': source, 'run': episode.get('run'),
                           'task_id': episode['task_id'], 'family': episode.get('family') or task.get('family'),
                           'domain': (task.get('curriculum') or {}).get('domain'),
                           'teacher': (episode.get('teacher') or {}).get('model_id'), 'policy': episode.get('policy'),
                           'policies': ep_policies,
                           'name': record['name'],
                           **({'live_source': record['live_source'], 'world': episode['world'], 'verification_mode': 'observed-live',
                               'task_completed': (episode.get('trajectory_ir') or {}).get('outcome', {}).get('task_completed', True)}
                              if record.get('verification_mode') == 'observed-live' else {}),
                           'open_goal': task['interaction']['goal']
                           if record.get('verification_mode') == 'observed-live' and task.get('interaction') else None,
                           'source_groups': task.get('source_groups', ()),
                           'guided': episode.get('role') == 'teacher' or source == 'jolt-guided',
                           'backend_workaround': facts['backend_workaround'],
                           'signature': facts['signature']})
    total, missing = teacher_coverage(args.teacher_pool, extracted)
    funnel['teacher']['manifest_successes_now'] = total
    funnel['teacher']['manifest_successes_not_extracted'] = len(missing)
    if missing and not args.allow_stale_teacher:
        raise SystemExit(f'{len(missing)} of {total} teacher successes in the pool manifests are not extracted yet '
                         f'(e.g. {missing[0]}): run extract --teacher POOL --out-name episodes-teacher and replay first')
    excluded = held_excluded_ids(args.held_manifest, tasks, held)
    kept = []
    for c in candidates:
        if c['task_id'] in excluded:
            funnel[c['source']]['held_excluded'] += 1
            continue
        kept.append(c)
    return kept, tasks


def october_candidates(args, funnel, held, weights):
    """One pass over the October candidate rows (sft-sources-20261117/october, per-call rows rebuilt for the current
    runtime): rows whose eval actions do not compile (`ts-host/scripts/check-eval-actions.mjs`), non-teacher rows and
    held rows are dropped; the rest are grouped into trajectories with a size estimate and an action signature (the
    MinHash of a union is the element-wise minimum of its parts' MinHashes)."""
    import numpy as np
    root = Path(args.october)
    trajectories = {}
    for group in sorted(weights):
        f = funnel['october:' + group]
        failed = set()
        check = root / 'eval-check' / f'{group}.jsonl'
        if check.exists():
            for line in check.open():
                row = json.loads(line)
                if row['failed']:
                    failed.add(row['id'])
        elif not group.startswith('acq-'):
            raise ValueError(f'no eval check for natlang group {group}: run ts-host/scripts/check-eval-actions.mjs')
        for line in zstd_lines(root / f'{group}.jsonl.zst'):
            row = json.loads(line)
            f['rows'] += 1
            provenance = row.get('provenance') or {}
            if row['id'] in failed:
                f['eval_does_not_compile'] += 1
                continue
            if group == 'student-projection' and provenance.get('collection_role') != 'teacher':
                f['not_teacher'] += 1
                continue
            if held_hit(row.get('task_id'), row.get('source_ids') or (), row.get('source_groups') or (), held):
                f['held_excluded'] += 1
                continue
            key = f"october:{group}:{provenance.get('trajectory') or row.get('task_id')}"
            size = (len(json.dumps(row['messages'], ensure_ascii=False)) + len(json.dumps(row['tools'], ensure_ascii=False))
                    + len(json.dumps(row['target'], ensure_ascii=False))) / CHARS_PER_TOKEN
            signature = minhash(shingles(action_text([row])))
            t = trajectories.get(key)
            if t is None:
                t = trajectories[key] = {'key': key, 'source': 'october:' + group, 'group': group, 'run': group,
                                         'task_id': row.get('task_id'), 'family': row.get('family'),
                                         'domain': row.get('domain'), 'teacher': provenance.get('model') or provenance.get('teacher'),
                                         'policy': None, 'rows': 0, 'size': 0., 'signature': signature}
            elif signature is not None:
                t['signature'] = signature if t['signature'] is None else np.minimum(t['signature'], signature)
            t['rows'] += 1
            t['size'] += size
            f['rows_kept'] += 1
    return list(trajectories.values())


def trained_in(builds):
    """(episode keys, task ids) an earlier build trained: its OUT/rows.jsonl.zst rows (``--prefer-unseen``)."""
    keys, tasks = set(), set()
    for build in builds:
        for line in zstd_lines(Path(build) / 'rows.jsonl.zst'):
            if line.strip():
                row = json.loads(line)
                keys.add(row['episode'])
                tasks.add(row['task_id'])
    return frozenset(keys), frozenset(tasks)


PREFER_UNSEEN_SCOPES = ('family', 'global')


def breadth_first(pool, key, trained_keys=frozenset(), scope='family'):
    """A pool in selection order: every task's first episode before any task's second, and within a round the
    families taking turns (a budget that ends mid-round trims the large families, not the small ones); ``key`` orders
    within a family's turn: held constructs first, then low base pass rates.

    ``trained_keys`` (``--prefer-unseen``): within each task its untrained episodes come first. ``scope`` 'family'
    (default) keeps the families' turns: in each family's turn its untrained tasks come before the tasks an earlier
    build trained, so a family whose fresh supply is gone still takes its turns with episodes the start already learned
    (a second epoch) instead of handing its share to the families that still have fresh ones. 'global' (the sft-v7
    rule) puts every untrained episode before any trained one: sft-v7's teacher budget then went to the unseen tail of
    the pool, 20 generator families of ~289 tasks each (74% of its teacher episodes in 25 families, 0.2% auth tasks
    against sft-v6's 23%), while auth carried sft-v6's held gain (DECISIONS 2026-11-22 sft-v7 no gain)."""
    if scope not in PREFER_UNSEEN_SCOPES:
        raise ValueError(f'prefer-unseen scope {scope!r} not in {PREFER_UNSEEN_SCOPES}')
    rank, seen = {}, collections.Counter()
    if scope == 'global':
        for c in pool:
            old = c['key'] in trained_keys
            rank[c['key']] = (old, seen[(old, c['task_id'])])
            seen[(old, c['task_id'])] += 1
        ordered = sorted(pool, key=lambda c: (rank[c['key']], key(c)))
    else:
        by_task = collections.defaultdict(list)
        for c in pool:
            by_task[c['task_id']].append(c)
        old_task = {}
        for task_id, members in by_task.items():
            members.sort(key=lambda c: c['key'] in trained_keys)    # stable: untrained first within the task
            for index, c in enumerate(members):
                rank[c['key']] = (index,)
            old_task[task_id] = members[0]['key'] in trained_keys   # the task offers nothing fresh
        ordered = sorted(pool, key=lambda c: (rank[c['key']], c['key'] in trained_keys and old_task[c['task_id']],
                                              key(c)))
    turn, taken = {}, collections.Counter()
    for c in ordered:
        turn[c['key']] = taken[(rank[c['key']], c['family'])]
        taken[(rank[c['key']], c['family'])] += 1
    return sorted(ordered, key=lambda c: (rank[c['key']], turn[c['key']]))


def round_robin(members, rng, order):
    """A task's episodes taking turns by producer (source and teacher model), the producers in `order`: a cap keeps
    different producers' solutions before a second one of any producer (no single teacher fills a task)."""
    producers = collections.defaultdict(list)
    for c in members:
        producers[(order(c), c['source'], c.get('teacher') or '')].append(c)
    queues = [producers[k] for k in sorted(producers, key=lambda k: (k[0], rng.random()))]
    for queue in queues:
        rng.shuffle(queue)
    out = []
    while any(queues):
        for queue in queues:
            if queue:
                out.append(queue.pop())
    return out


def capped(candidates, rng, funnel, cap_of, family_share, order, trained=frozenset(), rank=lambda c: 0):
    """Per-task caps (`cap_of(task_id)`, producers taking turns), then a per-family cap of `family_share` of the rest.
    Episodes whose key is in `trained` (an earlier build the start checkpoint learned, ``--prefer-unseen``) go after
    every untrained one under each cap; with none the selection is unchanged. ``rank`` orders tiers sharing one cap
    (own unguided successes 0 before JOLT+ guided 1) after that."""
    by_task = collections.defaultdict(list)
    for c in candidates:
        by_task[c['task_id']].append(c)
    selected = []
    for task_id, members in sorted(by_task.items(), key=lambda item: str(item[0])):
        members = round_robin(members, rng, order)
        members.sort(key=lambda c: (c.get('key') in trained, rank(c)))   # stable: producers keep their turns
        cap = cap_of(task_id)
        selected += members[:cap]
        for c in members[cap:]:
            funnel[c['source']]['over_task_cap'] += 1
    by_family = collections.defaultdict(list)
    for c in selected:
        by_family[c['family']].append(c)
    limit = max(1, int(family_share * len(selected)))
    final = []
    for family, members in sorted(by_family.items(), key=lambda item: str(item[0])):
        rng.shuffle(members)
        members.sort(key=lambda c: (c.get('key') in trained, rank(c)))
        final += members[:limit]
        for c in members[limit:]:
            funnel[c['source']]['over_family_cap'] += 1
    return final


# Program constructs the held set tests heavily (held-v3: 30% Neuralese results, 30% folder reducers, 18% iterateOn,
# 11% decide, 75% several functions) while dataset wrappers were 53% of v2's pool tasks and 9% of held-v3's.
HELD_CONSTRUCTS = ('neuralese_result', 'neuralese_argument', 'decide_call', 'folder_reducer', 'iterate',
                   'readout_decision', 'call_depth_ge_3', 'multi_function')


def task_constructs(task):
    """(held constructs the task's program declares, whether it is a one-function dataset wrapper)."""
    from measure_repetition import program_features
    try:
        features = program_features(task)
    except (KeyError, TypeError, AttributeError):
        return (), False
    return tuple(k for k in HELD_CONSTRUCTS if features.get(k)), bool(features.get('dataset_wrapper'))


def call_row(renderer, source, key, index, turn, teacher_think='context', lead_in='context',
             backend_workarounds='context', own_empty_think='context'):
    """One verified call as a learner row: {'id', 'invocation', 'index', 'prompt', 'sample'[, 'weight']} (int32 token ids
    from the chat template; a reply the student did not sample has its think block at weight 0, ``think_context``; with
    ``backend_workarounds`` 'context' a non-student call whose prompt holds the text backend's Neuralese refusal is all
    context, ``backend_workaround``). Training, validation and size measurement all render through here. Raises on a
    render error. Also returns (prompt length, target length, zero-weight lead)."""
    return rendered_row(render_call(renderer, source, turn, teacher_think, lead_in, own_empty_think), source, key, index,
                        turn, backend_workarounds)


def render_call(renderer, source, turn, teacher_think='context', lead_in='context', own_empty_think='context',
                decisions=False):
    """The expensive half of ``call_row`` (chat template, tokenizer, think context): (prompt ids, target ids as int32
    arrays, zero-weight lead), and with ``decisions`` a fourth item, the call's decision (``render_decision``). Raises
    on a render error."""
    import numpy as np
    prompt, target = renderer.render(turn['messages'], turn['tools'], turn['target'])
    context = think_context(renderer, source, turn['target'], target, teacher_think, lead_in, own_empty_think)
    rendered = np.asarray(prompt, np.int32), np.asarray(target, np.int32), context
    return (*rendered, render_decision(renderer, turn, rendered[0])) if decisions else rendered


def render_decision(renderer, turn, prompt):
    """The decision a call is, or None: a call on a decision readout's surface (no tools offered, the last message the
    readout's request, ``decision_graph.decision_replies``) whose reply is one of the offered values is that readout's
    choice, which the runtime scores rather than samples (``decide``). Rendered by the scorer's own rendering
    (``serve.native_learning.decision_items``, which the Python engine and the SGLang actor score with): {'options':
    per option its closed continuation ids (int32), 'label': the reply's option, 'values': the option texts,
    'label_source': 'verified-trajectory'}, the learner's option-normalized objective (``exp0_sft.decision_backward``).
    {'error': ...} when the scorer's prompt differs from the row's or the options do not render.

    A field-mode readout (the request offers record FORMs, ``decision_graph.decision_forms``) whose reply is a whole
    record is one decision per field step under teacher forcing of the reply's own choices (``field_path``): {'steps':
    [decision as above, its options the step's open continuations (``decision_items(..., open_reply=True)``) and
    'open_reply': True]}; steps with a single candidate carry no choice and are left out. A reply that is one step's
    open text (a recorded step, ``open_reply`` capture; the RL collector writes the step's chosen option as the reply) is
    {'field_step': True}: the turn of its last scored step trains it, and its plain row would teach ending the reply
    mid-record (``admit`` drops it). The last scored step's reply is the whole record, or an open text whose later
    steps have one candidate each (the runtime neither scores nor records those): its path completes it."""
    import numpy as np
    from natlang_neuralese.data.decision_graph import (decision_forms, decision_label, decision_replies,
                                                       field_candidates, field_path)
    target = turn.get('target') or {}
    messages = turn.get('messages') or []
    replies = decision_replies(messages)
    forms = decision_forms(messages) if replies is None else None
    if (replies is None and forms is None) or turn.get('tools') or target.get('tool_calls'):
        return None
    from natlang_neuralese.serve.native_learning import decision_items
    reply = (target.get('content') or '').strip()

    def rendered(options, label, open_reply=False):
        try:
            before, continuations = decision_items(renderer, list(turn['messages']), turn.get('tools'), options,
                                                   open_reply)
        except Exception as error:
            return {'error': repr(error)[:200]}
        if any(kind != 'tok' for kind, _ in before) or [int(i) for _, i in before] != [int(i) for i in prompt]:
            return {'error': 'prompt-differs-from-scorer'}
        return {'options': [np.asarray(ids, np.int32) for ids in continuations], 'label': label, 'values': options,
                'label_source': 'verified-trajectory', **({'open_reply': True} if open_reply else {})}
    if replies is not None:
        label = decision_label(replies, reply)
        return {'error': 'reply-not-an-option'} if label is None else rendered(replies, label)
    path = field_path(forms, reply)
    if path is None:
        chosen = []
        while True:   # a recorded step: the reply is the open text of one step on its own path
            step = [text for text, complete in field_candidates(forms, chosen) if not complete and reply.startswith(text)]
            if len(step) != 1:
                return {'error': 'reply-not-a-form'}
            chosen.append(step[0])
            if step[0] == reply:
                break
        # the last scored step: the steps after it have one candidate each (not scored, not recorded) up to the
        # whole record, which then trains every step; otherwise a later recorded step carries the record
        while True:
            following = field_candidates(forms, chosen)
            if len(following) != 1:
                return {'field_step': True}
            chosen.append(following[0][0])
            if following[0][1]:
                break
        path = field_path(forms, chosen[-1])
    steps = [rendered(candidates, label, True) for candidates, label in path if len(candidates) >= 2]
    failed = next((step for step in steps if 'error' in step), None)
    if failed is not None:
        return failed
    return {'steps': steps} if steps else None


def rendered_row(rendered, source, key, index, turn, backend_workarounds='context'):
    """The cheap half of ``call_row``: the learner row of one ``render_call`` result. Arrays that crossed a process
    boundary are re-viewed with numpy's canonical dtype objects: pickle memoizes a dtype by identity, so rows carrying
    unpickled dtype copies would pickle into a different (if equivalent) chains.pkl."""
    import numpy as np
    prompt, target, context, *decision = rendered
    row_id = turn.get('id') or hashlib.sha256(f"{key}#{index}".encode()).hexdigest()[:24]
    row = {'id': row_id, 'invocation': turn.get('invocation_id') or '', 'index': index,
           'prompt': prompt.view(np.int32), 'sample': target.view(np.int32)}
    if context:
        row['weight'] = (np.arange(len(target)) >= context).astype(np.float32)
    if source not in OWN_SOURCES and detour_call(turn, backend_workarounds):
        row['weight'] = np.zeros(len(target), np.float32)   # the detour after the refusal: context
    decision = decision[0] if decision else None
    view = lambda d: {**d, 'options': [ids.view(np.int32) for ids in d['options']]}
    if decision and 'options' in decision:
        # the plain row stays (``exp0_sft --no-decision-objective`` trains it as token CE, as before)
        row['decision'] = view(decision)
    elif decision and 'steps' in decision:
        row['decisions'] = [view(step) for step in decision['steps']]   # a field-mode readout: one per step
    elif decision and decision.get('field_step'):
        row['field_step'] = True   # trained by its record's turn (``render_decision``); ``admit`` drops the row
    elif decision:
        row['decision_error'] = decision['error']
    return row, context


def replayed_turns(work, name):
    """The semantic turns of a replayed episode (WORK/replay/NAME/rollouts.jsonl, the replaying runtime's record)."""
    with (Path(work) / 'replay' / name / 'rollouts.jsonl').open() as handle:
        replayed = [json.loads(line) for line in handle if line.strip()][0]
    return mark_detour_spawned([semantic_turn(t, False) for t in replayed['turns']])


def own_as_recorded(args, work, core, funnel):
    """Own candidates render their recorded turns (``--own-render recorded``, the default; recorded_turns.py): their
    files are backfilled from WORK's extractions; an own candidate whose recording is gone is left out
    (``recorded_turns_missing``), never rendered under a prompt it was not sampled with."""
    if getattr(args, 'own_render', 'recorded') != 'recorded':
        return core
    own = {c['key']: c['name'] for c in core if c['source'] in recorded_turns.OWN_SOURCES and c.get('name')}
    written, missing = recorded_turns.backfill(work, own, episode_files(work), zstd_lines)
    missing = set(missing)
    kept = []
    for c in core:
        if c['key'] in own:
            if c['key'] in missing:
                funnel[c['source']]['recorded_turns_missing'] += 1
                continue
            c['recorded_turns'] = True
            funnel[c['source']]['rendered_as_recorded'] += 1
        kept.append(c)
    funnel['own_render'].update(recorded_backfilled=written, recorded_missing=len(missing))
    return kept


def candidate_turns(work, c):
    """The turns ``build`` renders for candidate ``c``: an observed-live episode's captured turns (its source checked
    unchanged), else the replaying runtime's record."""
    if c.get('verification_mode') == 'observed-live':
        from live_teacher_sources import load_episode
        source = c['live_source']
        if sha256_file(source['path']) != source['sha256']:
            raise ValueError('observed live source changed before rendering')
        return mark_detour_spawned(load_episode(source['path'])['turns'])
    if c.get('recorded_turns'):   # an own success: the turns as sampled (recorded_turns.py)
        return mark_detour_spawned(recorded_turns.read(work, c['name']))
    return replayed_turns(work, c['name'])


def render_candidate(renderer, work, c, think, lead_in, own_empty_think):
    """(turns, calls) of candidate ``c``: its turns (``candidate_turns``) and per call its ``render_call`` result or
    the render error's repr (a str). ``turns`` is the exception itself when the turns could not be read. Pure in its
    inputs: the build renders candidates ahead in worker processes (``RenderAhead``)."""
    try:
        turns = candidate_turns(work, c)
    except Exception as error:
        return error, None
    calls = []
    for turn in turns:
        try:
            calls.append(render_call(renderer, c['source'], turn, think, lead_in, own_empty_think, decisions=True))
        except Exception as error:
            calls.append(repr(error))
    return turns, calls


class RenderError(Exception):
    """A call ``render_candidate`` could not render (its message: the original error's repr)."""


# Worker processes of ``RenderAhead``: one renderer each, set up once by the pool's initializer.
_WORKER = {}


def _render_worker_init(make_renderer, work, lead_in, own_empty_think):
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    _WORKER.update(renderer=make_renderer(), work=work, lead_in=lead_in, own_empty_think=own_empty_think)


def _render_worker(c, think):
    w = _WORKER
    return render_candidate(w['renderer'], w['work'], c, think, w['lead_in'], w['own_empty_think'])


RENDER_FIELDS = ('key', 'source', 'name', 'verification_mode', 'live_source')


def default_workers():
    return max(1, min(16, (os.cpu_count() or 2) - 4))


class RenderAhead:
    """``render_candidate`` of the candidates a build loop walks, in the loop's order, by ``workers`` processes up to
    ``window`` candidates ahead of the loop. The loop asks for a candidate's result with ``get`` (it may skip
    candidates; their renders are dropped); a candidate outside the order renders in this process. ``workers`` 1: no
    processes, every result renders on demand (the serial build). Results equal the serial ones by construction: the
    rendering is a pure function of the candidate, and the loop's selection stays sequential."""

    def __init__(self, renderer, work, think_of, lead_in, own_empty_think, workers=1, make_renderer=None, window=None):
        """``make_renderer``: a picklable callable building the workers' renderer (``functools.partial(TemplateRenderer,
        tokenizer_dir)``)."""
        self.renderer, self.work, self.think_of = renderer, work, think_of
        self.lead_in, self.own_empty_think = lead_in, own_empty_think
        self.workers, self.pool = workers, None
        self.window = window or 4 * workers
        self.order, self.position, self.cursor = [], {}, 0
        self.pending, self.last = collections.OrderedDict(), (None, None)
        if workers > 1:
            import concurrent.futures
            import multiprocessing
            self.pool = concurrent.futures.ProcessPoolExecutor(
                workers, mp_context=multiprocessing.get_context('spawn'), initializer=_render_worker_init,
                initargs=(make_renderer, str(work), lead_in, own_empty_think))

    def walk(self, candidates):
        """Start a loop's order (the previous one's renders still ahead are dropped)."""
        for future in self.pending.values():
            future.cancel()
        self.pending.clear()
        self.order = list(candidates)
        self.position = {c['key']: i for i, c in reversed(list(enumerate(self.order)))}
        self.cursor = 0                  # the next candidate of the order to submit
        self._fill()

    def _submit(self):
        c = self.order[self.cursor]
        self.cursor += 1
        if c['key'] not in self.pending:
            self.pending[c['key']] = self.pool.submit(_render_worker, {k: c[k] for k in RENDER_FIELDS if k in c},
                                                      self.think_of(c))

    def _fill(self):
        while self.pool is not None and len(self.pending) < self.window and self.cursor < len(self.order):
            self._submit()

    def get(self, c):
        key = c['key']
        if self.last[0] == key:          # size measurement and admission of one candidate share its render
            return self.last[1]
        if self.pool is not None and key not in self.pending and self.position.get(key, -1) >= self.cursor:
            self.cursor = self.position[key]              # skipped past the window: move it to this candidate
            self._submit()
        if key in self.pending:
            while next(iter(self.pending)) != key:        # candidates the loop skipped
                self.pending.popitem(last=False)[1].cancel()
            future = self.pending.pop(key)
            self._fill()
            result = future.result()
        else:
            result = render_candidate(self.renderer, self.work, c, self.think_of(c), self.lead_in, self.own_empty_think)
        self.last = (key, result)
        return result

    def close(self):
        if self.pool is not None:
            self.walk(())
            self.pool.shutdown(cancel_futures=True)
            self.pool = None


def wrapper_capped(selected, rng, funnel, is_wrapper, share, trained=frozenset(), rank=lambda c: 0):
    """Keep dataset-wrapper episodes to at most `share` of the selection (untrained ones first, then by ``rank``, as in
    `capped`)."""
    wrappers = [c for c in selected if is_wrapper(c)]
    rest = [c for c in selected if not is_wrapper(c)]
    limit = int(share / (1 - share) * len(rest)) if share < 1 else len(wrappers)
    rng.shuffle(wrappers)
    wrappers.sort(key=lambda c: (c.get('key') in trained, rank(c)))
    for c in wrappers[limit:]:
        funnel[c['source']]['over_wrapper_share'] += 1
    return rest + wrappers[:limit]


def target_mass_report(mass):
    """Actual rendered target mass, without introducing another weighting policy."""
    roles = {role: collections.Counter(calls=0, target_tokens=0, weighted_target_tokens=0)
             for role in ('root', 'child', 'unknown')}
    entries = []
    for (suite, function, role), counts in sorted(mass.items()):
        entries.append({'suite': suite, 'function': function, 'role': role, **dict(counts)})
        roles[role].update(counts)
    return {'by_suite_function': entries, 'by_role': {role: dict(counts) for role, counts in roles.items()}}


class EpisodeSizes:
    """(forward tokens, weighted target tokens) of a verified episode as `build` renders it, cached in WORK/sizes.jsonl
    per (episode, renderer identity, think and lead-in policy): budgets and loss shares choose on actual sizes."""

    def __init__(self, path, renderer, teacher_think, lead_in, backend_workarounds='context', own_empty_think='context'):
        # teacher_think: one value, or a function of the candidate (``ThinkTreatment.render``)
        self.path, self.renderer, self.lead_in = Path(path), renderer, lead_in
        self.think_of = teacher_think if callable(teacher_think) else (lambda c, value=teacher_think: value)
        self.backend_workarounds, self.own_empty_think = backend_workarounds, own_empty_think
        self.cache, self.new = {}, []
        if self.path.exists():
            for line in self.path.open():
                if line.strip():
                    record = json.loads(line)
                    self.cache[(record['key'], record['policy'])] = (record['forward'], record['weighted'])

    def policy(self, think):
        policy = [self.renderer.identity, think, self.lead_in, self.backend_workarounds, self.own_empty_think,
                  'wrapped-refusal', 'detour-spawned']
        return hashlib.sha256(json.dumps(policy, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def measure(self, c, rendered):
        """``rendered(c)``: the candidate's ``render_candidate`` result (``RenderAhead.get``), asked only on a cache
        miss."""
        from exp0_sft import chains_of
        think = self.think_of(c)
        policy = self.policy(think)
        hit = self.cache.get((c['key'], policy))
        if hit is not None:
            return hit
        turns, calls = rendered(c)
        if isinstance(turns, Exception) or any(isinstance(call, str) for call in calls):
            return None, None
        rows = [rendered_row(call, c['source'], c['key'], index, turn, self.backend_workarounds)[0]
                for index, (turn, call) in enumerate(zip(turns, calls))]
        forward = sum(len(chain['ids']) for chain in chains_of([{'task_id': c['task_id'], 'round': c['key'],
                                                                 'rows': rows}]))
        weighted = int(sum(float(r['weight'].sum()) if 'weight' in r else len(r['sample']) for r in rows))
        self.cache[(c['key'], policy)] = (forward, weighted)
        self.new.append({'key': c['key'], 'policy': policy, 'forward': forward, 'weighted': weighted})
        return forward, weighted

    def save(self):
        if self.new:
            with self.path.open('a') as handle:
                handle.write(''.join(json.dumps(record) + '\n' for record in self.new))
            self.new = []


def parse_aux(spec):
    """``NAME=PATH[,share=S|all][,task_tokens=N][,think=context|train][,replaces=SOURCE]``: a prebuilt chains file
    (operator rows, Neuralese text/graph rows, live-workflow rows, filled-reasoning rows) taking ``share`` of the build's
    weighted target tokens (``all``: every chain), each task's loss capped at ``task_tokens``; ``think`` context
    zero-weights a leading empty think block (``exp0_sft.mask_empty_think``: written or dataset targets, e.g. operator
    seeds and consumer answers). ``replaces`` SOURCE: a unit whose round is ``<prefix>:<episode key>`` stands in for
    that episode of SOURCE when the build admitted it, whose chains are then dropped (filled_reasoning.py rows: the same
    teacher actions after the student's filled reasoning, never trained twice)."""
    name, rest = spec.split('=', 1)
    path, *options = rest.split(',')
    part = {'name': name, 'path': path, 'share': 'all', 'task_tokens': 0, 'think': 'context', 'replaces': None}
    for option in options:
        key, value = option.split('=', 1)
        if key == 'share':
            part['share'] = value if value == 'all' else float(value)
        elif key == 'task_tokens':
            part['task_tokens'] = int(value)
        elif key == 'think':
            if value not in ('context', 'train'):
                raise ValueError(f'aux think {value!r}')
            part['think'] = value
        elif key == 'replaces':
            part['replaces'] = value
        else:
            raise ValueError(f'aux option {key!r} in {spec!r}')
    return part


def load_aux(part, tokenizer, held):
    """A prebuilt part's chains grouped into units (one episode: the chains sharing task and round), held rows dropped,
    empty think blocks masked (``part['think']`` context). Returns [{'task', 'chains', 'weighted'}]."""
    import pickle
    from exp0_sft import chain_weights, mask_after_end, mask_empty_think
    with open(part['path'], 'rb') as handle:
        chains = pickle.load(handle)['chains']
    if part['think'] == 'context':
        part['think_masked_rows'] = mask_empty_think(chains, tokenizer)
    part['after_end_masked_rows'] = mask_after_end(chains, tokenizer)   # the template's separator after <|im_end|>
    units = collections.defaultdict(list)
    for chain in chains:
        task = chain.get('task')
        if held_hit(task, (), (), held):
            part['held_dropped'] = part.get('held_dropped', 0) + 1
            continue
        units[(str(task), str(chain.get('round')))].append(chain)
    return [{'task': task, 'round': round_, 'chains': members,
             'weighted': float(sum(chain_weights(chain, 'cpu').sum().item() for chain in members))}
            for (task, round_), members in units.items()]


def replaced_round(part, unit):
    """The episode key (chain round) an aux unit stands in for (``parse_aux`` ``replaces``: a round
    ``<prefix>:<episode key>``), or None. The build drops the chains of that round whose part is the ``replaces``
    source, so any key form matches its own episode: ``teacher:``, ``october-teacher:`` and observed-live ``live:``
    teacher keys alike (the key's own prefix once had to equal the source, so October and live fills trained beside
    the teacher chains they replace)."""
    if not part.get('replaces') or ':' not in unit['round']:
        return None
    return unit['round'].split(':', 1)[1]


def open_goal_key(goal):
    return ' '.join(normalize('NFKC', goal).lower().split())


def mark_open_goal(chains, goal, turns):
    """Mark an open-goal episode's chains (``open_goal_fingerprint``, and ``open_goal_depth`` below the root invocation
    from the turns' live call ancestry) for the open-goal exposure check; no goal, no marks. The build's own chains and
    the filled-reasoning chains that replace them (filled_reasoning.py build) carry the same marks."""
    if not goal:
        return
    fingerprint = hashlib.sha256(open_goal_key(goal).encode()).hexdigest()
    depths = {}
    for turn in turns:
        ancestry = (turn.get('live_exchange') or {}).get('call_ancestry')
        if ancestry:
            depths[turn['invocation_id']] = len(ancestry) - 1
    for chain in chains:
        chain['open_goal_fingerprint'] = fingerprint
        invocation = chain['segs'][0][2].get('invocation')
        if invocation in depths:
            chain['open_goal_depth'] = depths[invocation]


def drop_unfilled_live(chains, mode, aux_parts):
    """(chains, episodes dropped): ``mode`` drop leaves out the observed-live teacher episodes (rounds ``live:...``,
    a live campaign's: no usable teacher reasoning) that no filled-reasoning unit replaced, so they train only with the
    student's filled reasoning, never after an empty or teacher think block; ``auto`` drops when a ``replaces=teacher``
    part is in the build (an expert-iteration round), else trains them (``train``) like other teacher episodes."""
    if mode == 'auto':
        mode = 'drop' if any(part.get('replaces') == 'teacher' for part in aux_parts) else 'train'
    if mode != 'drop':
        return chains, 0
    unfilled = {str(c.get('round')) for c in chains if c.get('part') == 'teacher' and str(c.get('round')).startswith('live:')}
    return [c for c in chains if not (c.get('part') == 'teacher' and str(c.get('round')) in unfilled)], len(unfilled)


def mix_report(chains, aux_report, open_goal_scope=None):
    """Per part (source or aux name): chains, forward and weighted target tokens and their shares; loss concentration
    over tasks (top 10 tasks, top 1% of tasks, the largest task); the open-goal exposure over ``open_goal_scope``'s
    chains (``exp0_sft.in_open_goal_scope``)."""
    from exp0_sft import chain_mass, open_goal_depth_report, open_goal_exposure, source_tokens
    parts = collections.defaultdict(collections.Counter)
    per_task = collections.Counter()
    for chain in chains:
        weighted = chain_mass(chain)
        p = parts[chain.get('part') or 'unknown']
        p['chains'] += 1
        p['forward_tokens'] += len(chain['ids']) + source_tokens(chain)
        p['weighted_target_tokens'] += weighted
        per_task[str(chain.get('task'))] += weighted
    total_w = sum(p['weighted_target_tokens'] for p in parts.values()) or 1
    total_f = sum(p['forward_tokens'] for p in parts.values()) or 1
    ranked = sorted(per_task.values(), reverse=True)
    top1 = max(1, len(ranked) // 100)
    return {'parts': {k: {'chains': v['chains'], 'forward_tokens': int(v['forward_tokens']),
                          'weighted_target_tokens': int(v['weighted_target_tokens']),
                          'loss_share': round(v['weighted_target_tokens'] / total_w, 4),
                          'forward_share': round(v['forward_tokens'] / total_f, 4)} for k, v in sorted(parts.items())},
            'weighted_target_tokens': int(total_w), 'forward_tokens': int(total_f), 'tasks': len(per_task),
            'loss_concentration': {'top10_tasks_share': round(sum(ranked[:10]) / total_w, 4),
                                   'top1pct_tasks_share': round(sum(ranked[:top1]) / total_w, 4),
                                   'max_task_tokens': int(ranked[0]) if ranked else 0},
            'open_user_goals': {**open_goal_exposure(chains, open_goal_scope), 'scope': open_goal_scope or 'all'},
            'open_user_goal_depths': open_goal_depth_report(chains),
            'decisions': decision_report(chains),
            'aux': aux_report}


def decision_report(chains):
    """The decision readout chains of a mix, per part: what ``exp0_sft`` trains with the option-normalized decision
    objective by default (``--no-decision-objective``: as the token CE of the chosen reply)."""
    parts = collections.defaultdict(collections.Counter)
    for chain in chains:
        if chain.get('decision') is not None:
            p = parts[chain.get('part') or 'unknown']
            p['chains'] += 1
            p['options'] += len(chain['decision']['options'])
            p['forward_tokens'] += sum(chain['segs'][0][0] + len(ids) for ids in chain['decision']['options'])
    return {'objective': 'option-normalized log score (exp0_sft.decision_backward; --no-decision-objective opts out)',
            'chains': sum(p['chains'] for p in parts.values()), 'parts': {k: dict(v) for k, v in sorted(parts.items())}}


def publish_sft_rows(rows, mix, minimum_open_goal_share):
    if minimum_open_goal_share is not None:
        from exp0_sft import check_open_goal_exposure
        try:
            check_open_goal_exposure(mix['open_user_goals'], minimum_open_goal_share)
        except SystemExit:
            rows.close(publish=False)
            raise
    rows.close()


def distinct_candidates(core, threshold, funnel):
    """Prefer whole successful demonstrations, while keeping distinct independently checked partial actions."""
    core = sorted(core, key=lambda c: (not bool(c.get('open_goal')), c.get('task_completed') is False, c['source'] != 'teacher'))
    dedup = NearDuplicates(threshold)
    unique = []
    seen_goals = set()
    for c in core:
        goal = open_goal_key(c['open_goal']) if c.get('open_goal') else None
        if c.get('task_completed') is not False and goal in seen_goals:
            funnel[c['source']]['repeated_open_goal'] += 1
        elif dedup.seen(c['signature']):
            funnel[c['source']]['near_duplicate'] += 1
        else:
            unique.append(c)
            if goal is not None and c.get('task_completed') is not False:
                seen_goals.add(goal)

    return unique


def build(args):
    data_registry.require_registered(build_inputs(args))
    import random
    import numpy as np
    import pickle
    from natlang_neuralese.common.code_provenance import code_provenance
    from exp0_sft import MAX_LEN, chain_mass, chains_of, source_tokens
    work, out = Path(args.work), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    funnel = collections.defaultdict(collections.Counter)
    held = held_closure(args.held_manifest)
    registry_path = getattr(args, 'live_holdout_registry', None)
    registry_digest = sha256_file(registry_path) if registry_path else None
    registered_training = live_holdout_registry(registry_path, held) if registry_path else {}
    if registry_path and sha256_file(registry_path) != registry_digest:
        raise ValueError('live holdout registry changed during read')
    # base pass rates: the exp1 measurement of the start policy, else the run groups' own (near-start) outcomes
    rate_data = json.loads(Path(args.base_rates).read_text()) if args.base_rates else {}
    rates = {k: (v['successes'], v['scored']) for k, v in rate_data.items()
             if v.get('scored')}
    group_rates = work / 'rl-group-rates.json'
    own_rates = json.loads(group_rates.read_text()) if group_rates.exists() else {}
    rates.update({k: (v['successes'], v['scored']) for k, v in own_rates.items()
                  if k not in rates and v['scored']})
    # the frontier: the trained policy's own measured rates replace every other rate of its tasks
    frontier_policy, frontier_prev_policy, frontier = load_frontier_full(getattr(args, 'frontier', None))
    previous_policy = getattr(args, 'previous_policy', None) or frontier_prev_policy
    rates.update({k: (v['successes'], v['scored']) for k, v in frontier.items() if v.get('scored')})
    tiers = {k: v.get('tier') for k, v in frontier.items()}
    treatment = ThinkTreatment.load(getattr(args, 'think_treatment', None), args.teacher_think)

    def base_rate(task_id):
        s, n = rates.get(task_id, (None, 0))
        return s / n if n else None

    # 1. verified successes of the GRPO pool (own and teacher): runs left out, guided attempts and backend detours by
    #    policy, near-duplicates out (teacher first), then each source capped on its own: task caps by base pass rate,
    #    family shares, the dataset-wrapper share
    core, tasks = core_candidates(args, work, funnel, held)
    core = own_as_recorded(args, work, core, funnel)
    for candidate in core:
        if candidate.get('open_goal'):
            require_registered_open_goal(candidate, registered_training)
    open_goals_only = getattr(args, 'open_goals_only', False)
    if open_goals_only and (args.october or args.aux):
        raise SystemExit('--open-goals-only cannot be combined with --october or --aux')
    excluded_runs = [re.compile(pattern) for pattern in args.exclude_run]
    kept = []
    for c in core:
        if open_goals_only and not c.get('open_goal'):
            funnel[c['source']]['excluded_non_open_goal'] += 1
        elif c['source'] in OWN_SOURCES and any(p.search(c.get('run') or '') for p in excluded_runs):
            funnel[c['source']]['excluded_run'] += 1
        elif c['source'] in OWN_SOURCES and c.get('guided') and args.guided == 'exclude':
            funnel[c['source']]['guided_excluded'] += 1
        elif c['source'] not in OWN_SOURCES and c.get('backend_workaround') and args.backend_workarounds == 'exclude':
            funnel[c['source']]['backend_workaround'] += 1
        else:
            kept.append(c)
    core = kept
    rng.shuffle(core)
    unique = distinct_candidates(core, args.dup_threshold, funnel)

    constructs = {task_id: task_constructs(task) for task_id, task in tasks.items()}

    def core_cap(task_id):
        if tiers.get(task_id) == 'solved':     # solved reliably: a light anchor
            return getattr(args, 'anchor_per_task', 1)
        top = args.rich_max_per_task if constructs.get(task_id, ((), False))[0] else args.max_per_task
        p = base_rate(task_id)
        return top if p is None else max(1, round(1 + (top - 1) * (1 - p)))

    def is_wrapper(c):
        return constructs.get(c['task_id'], ((), False))[1]
    trained_keys, trained_before = trained_in(getattr(args, 'prefer_unseen', None) or [])
    # the student's own distribution: unguided own successes and leak-checked JOLT+ guided successes share one per-task
    # cap, one family share and one wrapper share (own first under each); a frontier build drops a task's guided
    # successes where it has unguided ones (``frontier_drop``: own > jolt-guided > teacher, async-train §8)
    student_pool, own_tasks, guided_tasks = student_selection(
        unique, rng, funnel, core_cap, args.max_family_share, is_wrapper, args.wrapper_share, trained_keys,
        tiers if frontier else None)
    deferred = {'teacher': [], 'aux': []}
    teacher_candidates = []
    for c in unique:
        if c['source'] in OWN_SOURCES:
            continue
        why = frontier_drop(c['task_id'], tiers, own_tasks) if frontier else None
        if why:
            funnel[c['source']][why] += 1
        else:
            teacher_candidates.append(c)
    teacher_pool = wrapper_capped(capped(teacher_candidates, rng, funnel, core_cap, args.max_family_share, lambda c: 0,
                                         trained_keys), rng, funnel, is_wrapper, args.wrapper_share, trained_keys)

    renderer = TemplateRenderer(args.tokenizer)
    rows_out = ZstdWriter(out / 'rows.jsonl.zst')
    sft_episodes, all_chains = [], []
    stats = collections.defaultdict(collections.Counter)
    families, domains = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter)
    producers = collections.defaultdict(collections.Counter)
    target_mass = collections.defaultdict(collections.Counter)

    def admit(c, turns, calls=None):
        """Write one trajectory's rows and keep its token rows for the chains. ``calls``: its ``render_candidate``
        results (rendered here when not given)."""
        source, rows, written = c['source'], [], []
        if isinstance(turns, Exception):
            raise turns
        if calls is None:
            calls = []
            for turn in turns:
                try:
                    calls.append(render_call(renderer, source, turn, treatment.render(c), args.teacher_lead_in,
                                             args.own_empty_think, decisions=True))
                except Exception as error:
                    calls.append(repr(error))
        for index, (turn, call) in enumerate(zip(turns, calls)):
            try:
                if isinstance(call, str):
                    raise RenderError(call)
                row, context = rendered_row(call, source, c['key'], index, turn, args.backend_workarounds)
                if row.pop('field_step', False):
                    stats[source]['decision_field_step_rows_trained_by_record'] += 1
                    continue
            except Exception as error:
                stats[source]['render_error_calls'] += 1
                print(json.dumps({'event': 'render error', 'key': c['key'], 'call': index,
                                  'error': (str(error) if isinstance(error, RenderError) else repr(error))[:300]}),
                      flush=True)
                if c['source'] in ('rl-own', 'jolt-guided', 'teacher'):
                    return False   # a verified episode is admitted whole or not at all
                continue
            if source not in OWN_SOURCES and detour_call(turn, args.backend_workarounds):
                stats[source]['backend_detour_calls'] += 1   # call_row weighted the detour as context
            if source in OWN_SOURCES and context:
                stats[source]['own_unsampled_think_calls'] += 1
            prompt, target = row['prompt'], row['sample']
            if c.get('open_goal') and len(prompt) + len(target) > MAX_LEN:
                return False
            rows.append(row)
            written.append({'id': row['id'], 'episode': c['key'], 'source': source, 'run': c['run'],
                            'task_id': c['task_id'], 'family': c['family'], 'domain': c['domain'],
                            'teacher': c['teacher'], 'policy': c['policy'], 'call': index,
                            'invocation': turn.get('invocation_id'), 'messages': turn['messages'],
                            'function': turn.get('function') or 'unknown', 'definition_id': turn.get('definition_id'),
                            'parent_invocation': turn.get('parent_invocation'),
                            'invocation_role': turn.get('invocation_role') or 'unknown',
                            'tools': turn['tools'], 'target': turn['target'], 'prompt_tokens': len(prompt),
                            'target_tokens': len(target), 'loss_from': context,
                            'too_long': len(prompt) + len(target) > MAX_LEN,
                            **({'decision': {'label': row['decision']['label'], 'values': row['decision']['values'],
                                             'option_tokens': [len(o) for o in row['decision']['options']]}}
                               if 'decision' in row else {}),
                            **({'decision_steps': [{'label': d['label'], 'values': d['values'],
                                                    'option_tokens': [len(o) for o in d['options']]}
                                                   for d in row['decisions']]} if 'decisions' in row else {}),
                            **({'decision_error': row['decision_error']} if 'decision_error' in row else {}),
                            **({'verification_mode': c['verification_mode'], 'live_source': c['live_source'],
                                'world': c['world'], 'live_exchange': turn['live_exchange']}
                               if c.get('verification_mode') == 'observed-live' else {})})
        if not rows:
            return False
        for row in written:
            rows_out.write(row)
            target_mass[(c['family'], row['function'], row['invocation_role'])].update(
                calls=1, target_tokens=row['target_tokens'], weighted_target_tokens=row['target_tokens'] - row['loss_from'])
        funnel[source]['admitted'] += 1
        treatment.counts[treatment.of(c) or 'own'] += 1
        s = stats[source]
        s['calls'] += len(rows)
        s['target_tokens'] += sum(len(r['sample']) for r in rows)
        s['weighted_target_tokens'] += int(sum(float(r['weight'].sum()) if 'weight' in r else len(r['sample'])
                                               for r in rows))
        s['call_tokens'] += sum(len(r['prompt']) + len(r['sample']) for r in rows)
        s['too_long_calls'] += sum(len(r['prompt']) + len(r['sample']) > MAX_LEN for r in rows)
        # decision readouts carried with their option sets (exp0_sft trains them with the option-normalized objective)
        s['decision_rows'] += sum('decision' in r for r in rows)
        s['decision_options'] += sum(len(r['decision']['options']) for r in rows if 'decision' in r)
        s['decision_field_rows'] += sum('decisions' in r for r in rows)
        s['decision_field_steps'] += sum(len(r['decisions']) for r in rows if 'decisions' in r)
        s['decision_options'] += sum(len(d['options']) for r in rows for d in r.get('decisions') or ())
        for r in rows:
            if 'decision_error' in r:
                s[f"decision_surface_not_carried:{r.pop('decision_error')}"] += 1
        families[source][c['family']] += 1
        domains[source][c['domain']] += 1
        producer = f"{source}:{c.get('teacher') or c.get('run') or ''}"
        producers[producer]['episodes'] += 1
        producers[producer]['target_tokens'] += sum(len(r['sample']) for r in rows)
        producers[producer]['weighted_target_tokens'] += int(sum(float(r['weight'].sum()) if 'weight' in r
                                                                 else len(r['sample']) for r in rows))
        producers[producer]['think_context_calls'] += sum('weight' in r for r in rows)
        producers[producer]['reasoning_free_calls'] += sum(not (t['target'].get('reasoning_content') or '').strip()
                                                           for t in turns)
        episode = {'task_id': c['task_id'], 'round': c['key'], 'rows': rows, 'source': source,
                   'open_goal': c.get('open_goal')}
        episode_chains = chains_of([episode])
        for chain in episode_chains:
            chain['part'] = source
        mark_open_goal(episode_chains, episode['open_goal'], turns)
        for row in rows:          # a chain holds its prompts in 'ids'; the per-row copies only cost memory
            row.pop('prompt', None)
        all_chains.extend(episode_chains)
        episode['forward_tokens'] = sum(len(chain['ids']) for chain in episode_chains)
        episode['weighted_target_tokens'] = int(sum(float(r['weight'].sum()) if 'weight' in r else len(r['sample'])
                                                    for r in rows))
        s['forward_tokens'] += episode['forward_tokens']
        sft_episodes.append(episode)
        return True

    workers = getattr(args, 'workers', None) or 1
    import functools
    ahead = RenderAhead(renderer, work, treatment.render, args.teacher_lead_in, args.own_empty_think, workers=workers,
                        make_renderer=functools.partial(TemplateRenderer, args.tokenizer))

    def rendered(c):
        """(turns, calls) of a candidate (``render_candidate``), rendered ahead by ``--workers`` processes."""
        return ahead.get(c)

    episode_sizes = EpisodeSizes(work / 'sizes.jsonl', renderer, treatment.render, args.teacher_lead_in,
                                 args.backend_workarounds, args.own_empty_think)

    unseen_scope = getattr(args, 'prefer_unseen_scope', 'family')

    def ordered_pool(pool, key):
        """``breadth_first`` under this build's ``--prefer-unseen`` builds and ``--prefer-unseen-scope``."""
        return breadth_first(pool, key, trained_keys, unseen_scope)

    def priority(c):
        p = base_rate(c['task_id'])
        # a frontier build takes the trained (or adjacent previous) policy's own successes before older policies' (async-train §3)
        return (not bool(c.get('open_goal')), own_policy_tier(c, frontier_policy, previous_policy),
                not bool(constructs.get(c['task_id'], ((), False))[0]), 0.5 if p is None else p, rng.random())

    # teacher successes, breadth-first under the forward-token budget (0: all of the capped pool)
    task_tokens = collections.Counter()      # weighted target tokens admitted per task, all sources
    left_over = collections.defaultdict(list)    # candidates a budget or share left out: validation sets
    teacher_forward = 0
    ordered = ordered_pool(teacher_pool, priority)
    ahead.walk(ordered)

    def admit_teacher(c):
        nonlocal teacher_forward
        if args.teacher_forward_budget:
            forward, weighted = episode_sizes.measure(c, rendered)
            if forward is None:
                funnel[c['source']]['render_error'] += 1
                return
            if teacher_forward + forward > args.teacher_forward_budget:
                funnel[c['source']]['over_forward_budget'] += 1
                left_over['teacher'].append(c)
                return
        if not admit(c, *rendered(c)):
            funnel[c['source']]['render_error'] += 1
            return
        if c['key'] in trained_keys:
            funnel[c['source']]['admitted_trained_before'] += 1
        teacher_forward += sft_episodes[-1]['forward_tokens']
        task_tokens[c['task_id']] += sft_episodes[-1]['weighted_target_tokens']

    for c in ordered:
        if c['task_id'] in guided_tasks:
            deferred['teacher'].append(c)
            continue
        if args.teacher_forward_budget and teacher_forward >= 0.995 * args.teacher_forward_budget:
            funnel[c['source']]['over_forward_budget'] += 1     # full: the rest is left over unmeasured
            left_over['teacher'].append(c)
            continue
        admit_teacher(c)
    episode_sizes.save()

    # 2. October trajectories (no replayable task): compiled actions, held out, deduplicated against everything admitted,
    #    one trajectory per task by default, family caps, then a forward-token budget split by group weights
    october_budget, october_report = 0, None
    if args.october and args.october_share > 0:
        weights = dict((g, float(w)) for g, w in (spec.split('=') for spec in (args.october_group or DEFAULT_OCTOBER_GROUPS)))
        candidates = october_candidates(args, funnel, held, weights)
        rng.shuffle(candidates)
        unique = []
        for c in candidates:
            if dedup.seen(c['signature']):
                funnel[c['source']]['near_duplicate'] += 1
            else:
                unique.append(c)
        selected = capped(unique, rng, funnel, lambda task_id: args.october_max_per_task, args.october_family_share,
                          lambda c: 0)
        october_budget = teacher_forward * args.october_share / (1 - args.october_share)
        sizes = collections.Counter()
        for c in selected:
            sizes[c['group']] += c['size']
        rng.shuffle(selected)
        root = Path(args.october)
        scale = collections.defaultdict(lambda: 1.)        # actual forward tokens per estimated token, per group
        actual, estimated_done, admitted_keys = collections.Counter(), collections.Counter(), set()
        passes = []
        for attempt in range(args.october_passes):
            # budgets on the current size estimates (sizes still open, scaled), less what earlier passes used
            open_sizes = collections.Counter()
            for c in selected:
                if c['key'] not in admitted_keys:
                    open_sizes[c['group']] += c['size'] * scale[c['group']]
            budgets = water_fill({g: open_sizes[g] + actual[g] for g in weights if open_sizes[g] + actual[g] > 0},
                                 weights, october_budget)
            chosen, planned = {}, collections.Counter()
            for c in selected:
                g = c['group']
                if c['key'] in admitted_keys:
                    continue
                estimate = c['size'] * scale[g]
                if actual[g] + planned[g] + estimate <= budgets.get(g, 0):
                    planned[g] += estimate
                    chosen[c['key']] = c
            if not chosen:
                break
            # render the chosen trajectories, one group at a time (rows of a trajectory kept in file order)
            for group in sorted({c['group'] for c in chosen.values()}):
                pending = collections.defaultdict(list)
                for line in zstd_lines(root / f'{group}.jsonl.zst'):
                    row = json.loads(line)
                    key = f"october:{group}:{(row.get('provenance') or {}).get('trajectory') or row.get('task_id')}"
                    if key in chosen:
                        pending[key].append(row)
                for key, rows in pending.items():
                    c = chosen[key]
                    turns = [{'id': r['id'], 'messages': r['messages'], 'tools': r['tools'], 'target': r['target'],
                              'invocation_id': (r.get('provenance') or {}).get('invocation') or key} for r in rows]
                    c['run'] = group
                    before = stats[c['source']]['forward_tokens']
                    admitted_keys.add(key)
                    if not admit(c, turns):
                        funnel[c['source']]['render_error'] += 1
                        continue
                    actual[group] += stats[c['source']]['forward_tokens'] - before
                    estimated_done[group] += c['size']
                del pending
            for g in estimated_done:
                scale[g] = actual[g] / estimated_done[g] if estimated_done[g] else 1.
            passes.append({'chosen': len(chosen), 'actual_forward': dict(actual), 'budgets': {g: round(v) for g, v in budgets.items()}})
            if sum(actual.values()) >= 0.95 * october_budget:
                break
        for c in selected:
            if c['key'] not in admitted_keys:
                funnel[c['source']]['over_budget'] += 1
        october_report = {'weights': weights, 'budget_forward_tokens': round(october_budget),
                          'estimated_group_tokens': {g: round(v) for g, v in sizes.items()},
                          'actual_per_estimated_token': {g: round(v, 3) for g, v in scale.items()},
                          'passes': passes, 'chars_per_token_estimate': CHARS_PER_TOKEN}

    for e in sft_episodes:
        if e['source'].startswith('october:'):
            task_tokens[e['task_id']] += e['weighted_target_tokens']

    # 3. the loss shares: the teacher and October actions are the rest after the student's own share and each prebuilt
    #    part's share (``--aux``), all in weighted target tokens (the loss is token-normalized)
    actions_tokens = sum(e['weighted_target_tokens'] for e in sft_episodes)
    aux_parts = [parse_aux(spec) for spec in args.aux]
    fixed_share = args.own_loss_share + sum(part['share'] for part in aux_parts if part['share'] != 'all')
    if fixed_share >= 1:
        raise SystemExit(f'own and aux loss shares sum to {fixed_share} >= 1')
    loaded = [(part, load_aux(part, renderer.tokenizer, held)) for part in aux_parts]
    all_tokens = sum(sum(u['weighted'] for u in units) for part, units in loaded if part['share'] == 'all')
    total_tokens = (actions_tokens + all_tokens) / (1 - fixed_share)
    aux_report, replaced_rounds = {}, {}
    taken_rounds = set()    # an episode two parts both hold (a cycle's own Neuralese graph and an older graph build) trains once
    aux_state = {}

    def take_unit(part, unit):
        state = aux_state[part['name']]
        used, per_task = state['used'], state['per_task']
        if (unit['task'], unit['round']) in taken_rounds:
            used['duplicate_round'] += 1
            return
        if state['budget'] is not None and state['taken'] + unit['weighted'] > state['budget']:
            used['over_share'] += 1     # a unit that would overshoot: a smaller one may fit
            return
        if part['task_tokens'] and per_task[unit['task']] + unit['weighted'] > part['task_tokens']:
            used['over_task_tokens'] += 1
            return
        per_task[unit['task']] += unit['weighted']
        state['taken'] += unit['weighted']
        taken_rounds.add((unit['task'], unit['round']))
        replaced = replaced_round(part, unit)
        if replaced is not None:
            replaced_rounds[(part['replaces'], replaced)] = part['name']
        used['units'] += 1
        used['chains'] += len(unit['chains'])
        used['forward_tokens'] += sum(len(chain['ids']) + source_tokens(chain) for chain in unit['chains'])
        all_chains.extend(unit['chains'])
        for chain in unit['chains']:
            chain['part'] = part['name']

    for part, units in loaded:
        aux_state[part['name']] = {'budget': None if part['share'] == 'all' else part['share'] * total_tokens,
                                   'taken': 0, 'used': collections.Counter(), 'per_task': collections.Counter(),
                                   'units': len(units)}
        rng.shuffle(units)
        for unit in units:
            if frontier and part.get('replaces'):
                why = frontier_drop(unit['task'], tiers, own_tasks)
                if why:   # a filled teacher episode of a task the policy solves: its own successes train instead
                    aux_state[part['name']]['used'][why] += 1
                    continue
                if unit['task'] in guided_tasks:
                    deferred['aux'].append((part, unit))
                    continue
            take_unit(part, unit)

    # 4. the student's own verified successes (their reasoning included) up to their share, breadth-first over tasks,
    #    held constructs and low base pass rates first, each task's loss capped (``--max-task-target-tokens``);
    #    leak-checked JOLT+ guided successes follow unguided own successes within the student-distribution share
    own_budget = args.own_loss_share * total_tokens if args.own_loss_share else None
    own_tokens = 0
    ordered = ordered_pool(student_pool, priority)   # own before guided within each task, breadth-first over tasks
    ahead.walk(ordered)
    for c in ordered:
        if own_budget is not None and own_tokens >= 0.99 * own_budget:     # full: the rest is left over unmeasured
            funnel[c['source']]['over_loss_share'] += 1
            left_over['own'].append(c)
            continue
        forward, weighted = episode_sizes.measure(c, rendered)
        if forward is None:
            funnel[c['source']]['render_error'] += 1
            continue
        if own_budget is not None and own_tokens + weighted > own_budget:   # would overshoot: a smaller one may fit
            funnel[c['source']]['over_loss_share'] += 1
            left_over['own'].append(c)
            continue
        if args.max_task_target_tokens and task_tokens[c['task_id']] + weighted > args.max_task_target_tokens:
            funnel[c['source']]['over_task_target_tokens'] += 1
            continue
        if not admit(c, *rendered(c)):
            funnel[c['source']]['render_error'] += 1
            continue
        if c['key'] in trained_keys:
            funnel[c['source']]['admitted_trained_before'] += 1
        own_tokens += sft_episodes[-1]['weighted_target_tokens']
        task_tokens[c['task_id']] += sft_episodes[-1]['weighted_target_tokens']
    # 5. a task whose teacher data waited for its JOLT+ guided successes: those that entered replace it
    #    (``frontier_jolt_guided``); where none did (the own share or a cap left them out) the teacher data trains
    admitted_student = {e['task_id'] for e in sft_episodes if e['source'] in OWN_SOURCES}
    replaced, trains = settle_deferred(deferred['teacher'], admitted_student, lambda c: c['task_id'])
    for c in replaced:
        funnel[c['source']]['frontier_jolt_guided'] += 1
    for c in trains:
        if args.teacher_forward_budget and teacher_forward >= 0.995 * args.teacher_forward_budget:
            funnel[c['source']]['over_forward_budget'] += 1
            left_over['teacher'].append(c)
        else:
            funnel[c['source']]['frontier_jolt_guided_not_admitted'] += 1
            admit_teacher(c)
    replaced, trains = settle_deferred(deferred['aux'], admitted_student, lambda item: item[1]['task'])
    for part, unit in replaced:
        aux_state[part['name']]['used']['frontier_jolt_guided'] += 1
    for part, unit in trains:
        aux_state[part['name']]['used']['frontier_jolt_guided_not_admitted'] += 1
        take_unit(part, unit)
    episode_sizes.save()
    for part, units in loaded:
        state = aux_state[part['name']]
        aux_report[part['name']] = {**{k: v for k, v in part.items()}, 'sha256': sha256_file(part['path']),
                                    'available_units': state['units'], 'weighted_target_tokens': int(state['taken']),
                                    'tasks': len(state['per_task']),
                                    'budget_tokens': None if state['budget'] is None else int(state['budget']),
                                    'max_task_tokens': int(max(state['per_task'].values(), default=0)),
                                    **state['used']}
    if replaced_rounds:   # episodes an aux unit stands in for (``replaces``): their own chains go
        def replaced(chain):
            return (chain.get('part'), str(chain.get('round'))) in replaced_rounds
        dropped = [chain for chain in all_chains if replaced(chain)]
        kept = [chain for chain in all_chains if not replaced(chain)]
        for chain in dropped:
            r = aux_report[replaced_rounds[(chain.get('part'), str(chain.get('round')))]]
            r['replaced_chains'] = r.get('replaced_chains', 0) + 1
            r['replaced_weighted_target_tokens'] = r.get('replaced_weighted_target_tokens', 0) + int(chain_mass(chain))
            r.setdefault('replaced_episodes', set()).add(str(chain.get('round')))
        for r in aux_report.values():
            if 'replaced_episodes' in r:
                r['replaced_episodes'] = len(r['replaced_episodes'])
        all_chains[:] = kept
    all_chains[:], unfilled = drop_unfilled_live(all_chains, getattr(args, 'unfilled_live_teacher', 'auto'), aux_parts)
    if unfilled:
        funnel['teacher']['unfilled_live_dropped'] += unfilled
    chains = all_chains
    open_goal_scope = getattr(args, 'open_goal_scope', None) or 'all'
    mix = mix_report(chains, aux_report, open_goal_scope)
    minimum_open_goal_share = getattr(args, 'min_open_goal_share', None)
    if minimum_open_goal_share is None and (open_goals_only or any(
            task.get('source') == 'observed-live-workspace' and
            task.get('source_namespace', '').startswith('vibestudio:') for task in tasks.values()) or
            mix['open_user_goals']['distinct_goals']):
        minimum_open_goal_share = 0.8
    publish_sft_rows(rows_out, mix, minimum_open_goal_share)
    rng.shuffle(chains)
    with (out / 'chains.pkl.partial').open('wb') as handle:
        pickle.dump({'chains': chains, 'schema': 'natlang.sft-chains/1',
                     **({'minimum_open_goal_share': minimum_open_goal_share, 'open_goal_scope': open_goal_scope}
                        if minimum_open_goal_share is not None else {})}, handle, protocol=4)
    os.replace(out / 'chains.pkl.partial', out / 'chains.pkl')

    # validation sets: up to --val-episodes left-over successes per kind on tasks the build does not train (own:
    # the student's reasoning, whose NLL rising shows drift; teacher: actions on unseen tasks)
    trained_tasks = {e['task_id'] for e in sft_episodes} | trained_before   # the start checkpoint's tasks too
    validation = {}
    for kind, pool in sorted(left_over.items()):
        val_chains, val_tasks = [], set()
        ahead.walk([c for c in pool if c['task_id'] not in trained_tasks] if args.val_episodes else [])
        for c in pool:
            if len(val_tasks) >= args.val_episodes:
                break
            if c['task_id'] in trained_tasks or c['task_id'] in val_tasks:
                continue
            turns, calls = rendered(c)
            if isinstance(turns, Exception) or any(isinstance(call, str) for call in calls):
                continue
            try:
                rows = [rendered_row(call, c['source'], c['key'], index, turn, args.backend_workarounds)[0]
                        for index, (turn, call) in enumerate(zip(turns, calls))]
            except Exception:
                continue
            val_chains += chains_of([{'task_id': c['task_id'], 'round': c['key'], 'rows': rows}])
            val_tasks.add(c['task_id'])
        if val_chains:
            for chain in val_chains:
                for _, _, row in chain['segs']:
                    row.pop('prompt', None)
            with (out / f'val-{kind}.pkl').open('wb') as handle:
                pickle.dump({'chains': val_chains, 'schema': 'natlang.sft-chains/1'}, handle, protocol=4)
            validation[kind] = {'episodes': len(val_tasks), 'chains': len(val_chains),
                                'forward_tokens': int(sum(len(chain['ids']) for chain in val_chains))}
    ahead.close()
    rated = [base_rate(e['task_id']) for e in sft_episodes if e['round'].split(':')[0] in ('rl', 'teacher')]
    rated = [p for p in rated if p is not None]
    forward = int(sum(len(c['ids']) for c in chains))
    report = {
        'schema': 'natlang.sft-sources-build/1', 'code': code_provenance(), 'argv': sys.argv,
        'inputs': {'episodes': {str(p): sha256_file(p) for p in episode_files(work)}, 'teacher_pools': args.teacher_pool,
                   'replay': str(work / 'replay.jsonl') if (work / 'replay.jsonl').exists() else None,
                   'replay_sha256': sha256_file(work / 'replay.jsonl') if (work / 'replay.jsonl').exists() else None,
                   'observed_live': [c['live_source'] for c in core if c.get('verification_mode') == 'observed-live'],
                   'october': args.october, 'held_manifest': args.held_manifest,
                   'held_manifest_sha256': sha256_file(args.held_manifest), 'base_rates': args.base_rates,
                   'live_holdout_registry': {'path': registry_path, 'sha256': registry_digest} if registry_path else None,
                   'renderer': renderer.identity},
        'policy': {'max_per_task': args.max_per_task, 'rich_max_per_task': args.rich_max_per_task,
                   'held_constructs': HELD_CONSTRUCTS, 'wrapper_share': args.wrapper_share,
                   'cap_rule': 'round(1 + (cap - 1) * (1 - base pass rate)); cap = rich_max_per_task for a task '
                               'declaring a held construct, else max_per_task',
                   'max_family_share': args.max_family_share, 'dup_threshold': args.dup_threshold, 'seed': args.seed,
                   'admit_drifted': getattr(args, 'admit_drifted', 'on'), 'october_share': args.october_share, 'october_max_per_task': args.october_max_per_task,
                   'october_family_share': args.october_family_share,
                   'teacher_think': args.teacher_think, 'think_treatment': treatment.report(),
                   'frontier': {'path': getattr(args, 'frontier', None), 'policy': frontier_policy,
                                'previous_policy': previous_policy,
                                'sha256': sha256_file(args.frontier) if getattr(args, 'frontier', None) else None,
                                'tiers': dict(collections.Counter(tiers.values())),
                                'anchor_per_task': getattr(args, 'anchor_per_task', 1),
                                'rule': 'own successes where the task has any, else leak-checked jolt-guided, else '
                                        'teacher (filled first); a solved task keeps anchor_per_task own episodes'}
                   if frontier else None, 'teacher_lead_in': args.teacher_lead_in,
                   'exclude_run': args.exclude_run, 'guided': args.guided,
                   'prefer_unseen': {'builds': list(getattr(args, 'prefer_unseen', None) or []),
                                     'episodes': len(trained_keys), 'tasks': len(trained_before),
                                     'scope': unseen_scope},
                   'backend_workarounds': args.backend_workarounds, 'own_empty_think': args.own_empty_think,
                   'teacher_forward_budget': args.teacher_forward_budget, 'own_loss_share': args.own_loss_share,
                   'max_task_target_tokens': args.max_task_target_tokens, 'aux': args.aux,
                   'order': 'teacher: breadth-first over tasks (every task once before any twice), held constructs '
                            'then low base pass rate first, under teacher_forward_budget; October by forward share of '
                            'the teacher forward; aux parts and own successes by share of the weighted target tokens',
                   'loss': 'per-call rows: prompt (system, task, history, tool output) zero weight; the own reply '
                           '(reasoning, text, tool calls, end token) weight 1; a reply the student did not sample '
                           '(teacher, October, live): its think block through </think> zero weight (teacher_think '
                           'context; train: only an empty think block), the action after it weight 1 (row weight)'},
        'october': october_report,
        'mix': mix,
        'validation': validation,
        'funnel': {k: dict(v) for k, v in funnel.items()},
        'stats': {k: dict(v) for k, v in stats.items()},
        'episodes': len(sft_episodes), 'tasks': len({e['task_id'] for e in sft_episodes}),
        'admitted_episode_keys': sorted(e['round'] for e in sft_episodes),
        'chains': len(chains), 'forward_tokens': forward,
        'target_tokens': int(sum(len(r['sample']) for e in sft_episodes for r in e['rows'])),
        'target_mass': target_mass_report(target_mass),
        'epoch_hours_at_5500_tokens_per_second': round(forward / 5500 / 3600, 2),
        'producers': {k: {**dict(v), 'weighted_target_token_share': round(v['weighted_target_tokens'] / max(1, sum(
            p['weighted_target_tokens'] for p in producers.values())), 4)} for k, v in sorted(producers.items())},
        'families': {k: len(v) for k, v in families.items()},
        'family_counts': {k: dict(v.most_common()) for k, v in families.items()},
        'domain_counts': {k: dict(v.most_common()) for k, v in domains.items()},
        'core_base_rate': {'measured_episodes': len(rated),
                           'mean': round(float(np.mean(rated)), 3) if rated else None},
    }
    (out / 'build.json').write_text(json.dumps(report, indent=1) + '\n')
    admitted_tasks = {e['task_id'] for e in sft_episodes}
    (out / 'cases.jsonl').write_text(''.join(json.dumps(tasks[t]) + '\n' for t in sorted(tasks) if t in admitted_tasks))
    record_built(out, args, 'natlang.sft-sources-build/1')
    print(json.dumps({k: report[k] for k in ('funnel', 'stats', 'episodes', 'tasks', 'chains', 'forward_tokens',
                                             'target_tokens')}, indent=1))


def neuralese_parts(messages):
    """Number of native_state parts (imported Neuralese state) in a call's messages."""
    return sum(isinstance(m.get('content'), list) and sum(isinstance(part, dict) and part.get('type') == 'native_state'
                                                          for part in m['content']) for m in messages)


def current_surface_run(args, episode):
    """An own episode collected by a run on the current runtime (``--current-surface-run REGEX`` on its run label,
    e.g. the expert-iteration cycle's own collection on the cycle's pin): its surfaces are current by construction.
    The replay-derived set holds only the surfaces the WORK's exact replays rendered, and a WORK seeded with an earlier
    pin's verdicts renders that pin's prompts: sft-v7's 146 own Neuralese successes were all dropped as unseen."""
    return any(re.search(pattern, episode.get('run') or '') for pattern in getattr(args, 'current_surface_run', None) or ())


def surface_key(turn):
    """(system prompt, tool schemas) of a call: what the runtime surface renders around the task."""
    system = ''.join(m['content'] if isinstance(m['content'], str) else json.dumps(m['content'])
                     for m in turn['messages'] if m.get('role') == 'system')
    return (hashlib.sha256(system.encode()).hexdigest()[:16],
            hashlib.sha256(json.dumps(turn.get('tools'), sort_keys=True).encode()).hexdigest()[:16])


def _surfaces_of(names, work):
    from natlang_neuralese.common.jsonio import read_jsonl
    surfaces = set()
    for name in names:
        for row in read_jsonl(Path(work) / 'replay' / name / 'rollouts.jsonl'):
            surfaces.update(surface_key(t) for t in row.get('turns') or ())
    return surfaces


def replay_surfaces(work, workers=1):
    """The (system prompt, tool schemas) of every call of every exact replay of WORK: the surfaces the current runtime
    renders (``workers`` processes read the replays)."""
    names = []
    for line in (Path(work) / 'replay.jsonl').open():
        record = json.loads(line) if line.strip() else {}
        if record.get('exact'):
            names.append(record['name'])
    if workers <= 1:
        return _surfaces_of(names, work)
    import concurrent.futures
    import multiprocessing
    surfaces = set()
    with concurrent.futures.ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        for part in bounded_map(pool, _surfaces_of, batched(names, 64), 4 * workers, str(work)):
            surfaces |= part
    return surfaces


def neuralese_text(args):
    """Text rows of the verified successes the build skips for their Neuralese state (`has_native_state`): every call
    whose messages hold no imported native state, i.e. the Neuralese producer invocations (the prompted calls whose
    returned text becomes the state) and the consumer's calls before its first import. A call after an import needs the
    imported state as context (no text rendering is exact) and is counted, not written. No replay is possible for these
    episodes (their states lived only in the actor's store), so validity is: oracle success at collection, the held-v3/v2
    closure, and each call's system prompt and tool schemas equal to ones the current runtime rendered in an exact replay
    (calls with an unseen surface are dropped). Writes OUT/chains.pkl (`exp0_sft.py train --chains`, the build's format),
    OUT/rows.jsonl.zst and OUT/build.json."""
    data_registry.require_registered(build_inputs(args))
    import random
    import numpy as np
    import pickle
    from natlang_neuralese.common.code_provenance import code_provenance
    from natlang_neuralese.rl.held_eval import training_exclusion_ids
    from exp0_sft import MAX_LEN, chains_of
    work, out = Path(args.work), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    held = held_closure(args.held_manifest)
    funnel = collections.Counter()
    workers = getattr(args, 'workers', None) or 1
    surfaces = replay_surfaces(work, workers)    # the current runtime's surfaces: every call of every exact replay
    episodes = list(all_episodes(work, workers, keep=has_native_state))
    funnel['native_state_episodes'] = len(episodes)
    excluded = training_exclusion_ids([e['task'] for e in episodes], args.held_manifest)
    kept = []
    for e in episodes:
        task = e['task']
        groups = list(task.get('source_groups') or ()) + [(task.get('curriculum') or {}).get('split_group') or task['id']]
        if e['task_id'] in excluded or held_hit(e['task_id'], task.get('source_ids') or (), groups, held):
            funnel['held_excluded'] += 1
            continue
        kept.append(e)
    rng.shuffle(kept)
    per_task = collections.Counter()
    renderer = TemplateRenderer(args.tokenizer)
    rows_out = ZstdWriter(out / 'rows.jsonl.zst')
    sft_episodes, tasks = [], {}
    stats = collections.defaultdict(collections.Counter)
    for e in kept:
        if per_task[e['task_id']] >= args.max_per_task:
            funnel['over_task_cap'] += 1
            continue
        generator = 'bgkit' if e['task_id'].startswith('ext:bgkit:') else 'other'
        root = e['turns'][0].get('invocation_id')
        rows, written = [], []
        for index, turn in enumerate(e['turns']):
            role = 'root' if turn.get('invocation_id') == root else 'child'
            s = stats[f'{generator}:{role}']
            if neuralese_parts(turn['messages']):
                s['calls_after_import'] += 1
                s['target_tokens_after_import'] += turn.get('completion_tokens') or 0
                continue
            if surface_key(turn) not in surfaces and not current_surface_run(args, e):
                s['calls_unseen_surface'] += 1
                continue
            try:
                prompt, target = renderer.render(turn['messages'], turn['tools'], turn['target'])
            except Exception as error:
                s['render_error'] += 1
                print(json.dumps({'event': 'render error', 'key': e['key'], 'call': index, 'error': repr(error)[:300]}),
                      flush=True)
                continue
            row_id = hashlib.sha256(f"{e['key']}#{index}".encode()).hexdigest()[:24]
            rows.append({'id': row_id, 'invocation': turn.get('invocation_id') or '', 'index': index,
                         'prompt': np.asarray(prompt, np.int32), 'sample': np.asarray(target, np.int32)})
            written.append({'id': row_id, 'episode': e['key'], 'source': 'rl-own-neuralese', 'run': e.get('run'),
                            'task_id': e['task_id'], 'family': e.get('family') or e['task'].get('family'),
                            'generator': generator, 'role': role, 'policy': e.get('policy'), 'call': index,
                            'invocation': turn.get('invocation_id'), 'messages': turn['messages'],
                            'tools': turn['tools'], 'target': turn['target'], 'prompt_tokens': len(prompt),
                            'target_tokens': len(target), 'loss_from': 0,
                            'too_long': len(prompt) + len(target) > MAX_LEN})
            s['calls'] += 1
            s['target_tokens'] += len(target)
            s['call_tokens'] += len(prompt) + len(target)
            s['too_long_calls'] += len(prompt) + len(target) > MAX_LEN
        if not rows:
            funnel['no_text_calls'] += 1
            continue
        per_task[e['task_id']] += 1
        funnel['admitted'] += 1
        funnel[f'admitted_{generator}'] += 1
        tasks[e['task_id']] = e['task']
        for row in written:
            rows_out.write(row)
        episode = {'task_id': e['task_id'], 'round': f"rl-neuralese:{e['key']}", 'rows': rows}
        stats[generator]['forward_tokens'] += sum(len(c['ids']) for c in chains_of([episode]))
        stats[generator]['episodes'] += 1
        sft_episodes.append(episode)
    rows_out.close()
    chains = chains_of(sft_episodes)
    rng.shuffle(chains)
    with (out / 'chains.pkl.partial').open('wb') as handle:
        pickle.dump({'chains': chains, 'schema': 'natlang.sft-chains/1'}, handle, protocol=4)
    os.replace(out / 'chains.pkl.partial', out / 'chains.pkl')
    report = {
        'schema': 'natlang.sft-sources-neuralese-text/1', 'code': code_provenance(), 'argv': sys.argv,
        'inputs': {'episodes': {str(p): sha256_file(p) for p in episode_files(work)},
                   'replay_sha256': sha256_file(work / 'replay.jsonl'), 'held_manifest': args.held_manifest,
                   'held_manifest_sha256': sha256_file(args.held_manifest), 'renderer': renderer.identity},
        'policy': {'max_per_task': args.max_per_task, 'seed': args.seed, 'current_surfaces': len(surfaces),
                   'calls': 'no native_state part in the messages (producer invocations; consumer calls before the '
                            'first import); system prompt and tools equal to an exact replay\'s',
                   'loss': 'per-call rows: prompt zero weight; the own sampled reply weight 1'},
        'funnel': dict(funnel), 'stats': {k: dict(v) for k, v in sorted(stats.items())},
        'episodes': len(sft_episodes), 'tasks': len(tasks), 'chains': len(chains),
        'calls': int(sum(len(e['rows']) for e in sft_episodes)),
        'forward_tokens': int(sum(len(c['ids']) for c in chains)),
        'target_tokens': int(sum(len(r['sample']) for e in sft_episodes for r in e['rows'])),
    }
    (out / 'build.json').write_text(json.dumps(report, indent=1) + '\n')
    (out / 'cases.jsonl').write_text(''.join(json.dumps(tasks[t]) + '\n' for t in sorted(tasks)))
    record_built(out, args, 'natlang.sft-sources-neuralese-text/1')
    print(json.dumps({k: report[k] for k in ('funnel', 'stats', 'episodes', 'tasks', 'chains', 'calls',
                                             'forward_tokens', 'target_tokens')}, indent=1))


def extract_rl_specs(work):
    """The ``--rl RUN_DIR:ARCHIVE_DIR`` specs of every extraction of WORK, in order, each once: ``extract`` writes
    ``<out-name>-extract.json`` (``extract.json`` in WORKs older than ``--out-name``). Reading only ``extract.json``
    left `neuralese-graph` unable to run on any current WORK (sft-sources-20261120b, the expert-iteration WORK)."""
    specs = []
    for path in sorted(Path(work).glob('*extract.json')):
        for spec in json.loads(path.read_text()).get('rl') or ():
            if spec not in specs:
                specs.append(spec)
    return specs


def neuralese_graph(args):
    """Whole Neuralese episodes as whole-graph SFT rows: the verified successes `build` skips for their imported state
    (`has_native_state`), every call as the actor ran it (the exact prompt and sampled ids of its capture, read again
    from the run's rollout archive: `lower_rollout_turns.episode_turns`), consumer calls after an import included. An
    import is a pass-through: its producer document is the episode's recorded write (`write_sources`), which the learner
    runs in the consumer's forward (`exp0_sft.row_inputs`, plans/neuralese/NEURALESE_PASSTHROUGH_20261117.md), so no
    stored state is needed and the consumer's loss trains the producer's state. An episode with an import whose
    write was not recorded is dropped (counted). Validity as `neuralese-text`: oracle success at collection, the held-v3/v2
    closure, and each call's system prompt and tool schemas equal to ones the current runtime rendered in an exact
    replay (an episode with a call on an unseen surface is dropped whole: its graph would lose a dependency). Writes
    OUT/chains.pkl (`exp0_sft.py train --chains`), OUT/rows.jsonl.zst and OUT/build.json."""
    data_registry.require_registered(build_inputs(args))
    import random
    import pickle
    from natlang_neuralese.common.code_provenance import code_provenance
    from natlang_neuralese.rl.held_eval import training_exclusion_ids
    from exp0_sft import MAX_LEN, chains_of, source_tokens
    from lower_rollout_turns import episode_turns
    work, out = Path(args.work), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    held = held_closure(args.held_manifest)
    funnel = collections.Counter()
    workers = getattr(args, 'workers', None) or 1
    surfaces = replay_surfaces(work, workers)
    archives = {}
    for spec in extract_rl_specs(work):
        run_dir, archive_dir = spec.split(':')
        archives['rl:' + '/'.join(Path(run_dir).parts[-2:])] = (run_dir, archive_dir)
    excluded_runs = [re.compile(pattern) for pattern in getattr(args, 'exclude_run', None) or ()]
    episodes = [e for e in all_episodes(work, workers, keep=has_native_state) if e.get('source') in OWN_SOURCES]
    funnel['native_state_episodes'] = len(episodes)
    funnel['excluded_run'] = sum(1 for e in episodes if any(p.search(e.get('run') or '') for p in excluded_runs))
    episodes = [e for e in episodes if not any(p.search(e.get('run') or '') for p in excluded_runs)]
    excluded = training_exclusion_ids([e['task'] for e in episodes], args.held_manifest)
    kept = []
    for e in episodes:
        task = e['task']
        groups = list(task.get('source_groups') or ()) + [(task.get('curriculum') or {}).get('split_group') or task['id']]
        if e['task_id'] in excluded or held_hit(e['task_id'], task.get('source_ids') or (), groups, held):
            funnel['held_excluded'] += 1
            continue
        kept.append(e)
    rng.shuffle(kept)
    files = {}
    for label, (run_dir, archive_dir) in archives.items():
        for name, path, _ in rl_rollout_files(run_dir, archive_dir):
            files[(label, name)] = path
    wanted = collections.defaultdict(dict)
    for e in kept:
        wanted[(e['run'], e['group'])][e['sample']] = e
    lowered_of = {}         # key -> (every call on a current surface, lowering counters, calls): rows are not kept
    for (label, name), samples in sorted(wanted.items()):
        path = files.get((label, name))
        if path is None:
            funnel['rollouts_missing'] += len(samples)
            continue
        for line in zstd_lines(path):
            if line.strip():
                row = json.loads(line)
                if row.get('sample') in samples and row.get('task_id') == samples[row['sample']]['task_id']:
                    counters = collections.Counter()
                    calls = episode_turns(row, counters)
                    for call in calls:      # SFT: the own sampled tokens are the target (no importance weights)
                        for field in ('logp', 'probabilities'):
                            call.pop(field, None)
                    lowered_of[samples[row['sample']]['key']] = (
                        current_surface_run(args, samples[row['sample']]) or
                        all(surface_key(t) in surfaces for t in row['turns'] if t.get('messages')), counters, calls)
                del row
    per_task = collections.Counter()
    stats = collections.Counter()
    rows_out = ZstdWriter(out / 'rows.jsonl.zst')
    sft_episodes, tasks = [], {}
    for e in kept:
        if e['key'] not in lowered_of:
            funnel['rollout_row_missing'] += 1
            continue
        if per_task[e['task_id']] >= args.max_per_task:
            funnel['over_task_cap'] += 1
            continue
        current, lowered, calls = lowered_of[e['key']]
        if not current:
            funnel['unseen_surface'] += 1
            continue
        stats.update(lowered)
        if lowered['skip_native_state_prompt'] or lowered['skip_native_state_missing']:
            funnel['import_without_recorded_write'] += 1
            continue
        if not any(c['imports'] for c in calls):
            funnel['no_lowered_import'] += 1
            continue
        episode = {'task_id': e['task_id'], 'round': f"rl-neuralese-graph:{e['key']}", 'rows': calls}
        chains = chains_of([episode])
        per_task[e['task_id']] += 1
        funnel['admitted'] += 1
        tasks[e['task_id']] = e['task']
        generator = 'bgkit' if e['task_id'].startswith('ext:bgkit:') else 'other'
        stats[f'{generator}:episodes'] += 1
        stats[f'{generator}:calls'] += len(calls)
        stats[f'{generator}:consumer_calls'] += sum(1 for c in calls if c['imports'])
        stats[f'{generator}:target_tokens'] += sum(len(c['sample']) for c in calls)
        stats[f'{generator}:consumer_target_tokens'] += sum(len(c['sample']) for c in calls if c['imports'])
        stats[f'{generator}:forward_tokens'] += sum(len(c['ids']) + source_tokens(c) for c in chains)
        stats[f'{generator}:producer_document_tokens'] += sum(source_tokens(c) for c in chains)
        stats[f'{generator}:too_long_calls'] += sum(len(c['prompt']) + len(c['sample']) > MAX_LEN for c in calls)
        for call in calls:
            rows_out.write({'episode': e['key'], 'source': 'rl-own-neuralese-graph', 'run': e.get('run'),
                            'task_id': e['task_id'], 'family': e.get('family') or e['task'].get('family'),
                            'generator': generator, 'policy': e.get('policy'), 'call': call['index'],
                            'invocation': call['invocation'], 'imports': [list(i) for i in call['imports']],
                            'prompt_tokens': len(call['prompt']), 'target_tokens': len(call['sample'])})
        sft_episodes.append(episode)
    rows_out.close()
    chains = chains_of(sft_episodes)
    rng.shuffle(chains)
    with (out / 'chains.pkl.partial').open('wb') as handle:
        pickle.dump({'chains': chains, 'schema': 'natlang.sft-chains/1'}, handle, protocol=4)
    os.replace(out / 'chains.pkl.partial', out / 'chains.pkl')
    report = {
        'schema': 'natlang.sft-sources-neuralese-graph/1', 'code': code_provenance(), 'argv': sys.argv,
        'inputs': {'episodes': {str(p): sha256_file(p) for p in episode_files(work)},
                   'replay_sha256': sha256_file(work / 'replay.jsonl'), 'held_manifest': args.held_manifest,
                   'held_manifest_sha256': sha256_file(args.held_manifest), 'archives': archives},
        'policy': {'max_per_task': args.max_per_task, 'seed': args.seed, 'current_surfaces': len(surfaces),
                   'calls': 'every lowered call of the episode, exact actor ids; imports as pass-through producer '
                            'documents (the recorded write); episodes with an unrecorded write or an unseen surface dropped',
                   'loss': 'per-call rows: prompt and producer documents zero weight; the own sampled reply weight 1'},
        'funnel': dict(funnel), 'stats': dict(stats), 'episodes': len(sft_episodes), 'tasks': len(tasks),
        'chains': len(chains), 'importing_chains': sum(1 for c in chains if c['imports']),
        'calls': int(sum(len(e['rows']) for e in sft_episodes)),
        'forward_tokens': int(sum(len(c['ids']) + source_tokens(c) for c in chains)),
        'target_tokens': int(sum(len(r['sample']) for e in sft_episodes for r in e['rows'])),
    }
    (out / 'build.json').write_text(json.dumps(report, indent=1) + '\n')
    (out / 'cases.jsonl').write_text(''.join(json.dumps(tasks[t]) + '\n' for t in sorted(tasks)))
    record_built(out, args, 'natlang.sft-sources-neuralese-graph/1')
    print(json.dumps({k: report[k] for k in ('funnel', 'stats', 'episodes', 'tasks', 'chains', 'importing_chains',
                                             'calls', 'forward_tokens', 'target_tokens')}, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    e = sub.add_parser('extract')
    e.add_argument('--work', required=True)
    e.add_argument('--rl', action='append', default=[], metavar='RUN_DIR:ARCHIVE_DIR')
    e.add_argument('--teacher', action='append', default=[], metavar='POOL_DIR')
    e.add_argument('--live-run', action='append', default=[], metavar='MANIFEST',
                   help='verified observed workspace/CDP task; captured contexts only, no portable replay claim')
    e.add_argument('--live-capture-root', action='append', default=[], metavar='CAPTURE_DIR',
                   help='discover verified manifests in a single or multi-case live teacher capture directory')
    e.add_argument('--live-trajectory-out', metavar='OUTPUT.jsonl.zst',
                   help='also export verified open goals as standalone canonical trajectory IR')
    e.add_argument('--out-name', default='episodes', help='WORK/NAME.jsonl.zst (build reads every episodes*.jsonl.zst)')
    e.add_argument('--october-teacher', action='append', default=[], metavar='GROUP.jsonl.zst',
                   help='October per-call rows (sft-sources-20261117/october) as teacher episodes for replay')
    e.add_argument('--october-model', action='append', default=[], metavar='MODEL_ID',
                   help='keep only these provenance models of --october-teacher rows (default: all)')
    e.add_argument('--held-manifest', help='held closure for --october-teacher (default: the registry role held-eval)')
    r = sub.add_parser('replay')
    r.add_argument('--work', required=True)
    r.add_argument('--runtime-repo', help='checkout (a pin) whose runtime, collector and oracle replay (default: this one)')
    r.add_argument('--workers', type=int, default=6)
    r.add_argument('--limit', type=int, default=0)
    r.add_argument('--case-seconds', type=int, default=900)
    r.add_argument('--max-model-requests', type=int, default=None)
    c = sub.add_parser('check-render')
    c.add_argument('--work', required=True)
    c.add_argument('--tokenizer', required=True)
    c.add_argument('--turns', type=int, default=400)
    b = sub.add_parser('build')
    b.add_argument('--work', required=True)
    b.add_argument('--out', required=True)
    b.add_argument('--tokenizer', required=True)
    b.add_argument('--held-manifest', help='default: the registry role held-eval (held-v3)')
    b.add_argument('--live-holdout-registry', metavar='CATALOG.jsonl',
                   help='required for open-goal training; output of pnpm natlang:teacher --list')
    b.add_argument('--no-default-teacher-pool', action='store_true',
                   help='live-only build: do not require coverage of the historical default teacher pool')
    b.add_argument('--min-open-goal-share', type=float, choices=[0.8],
                   help='require at least 80%% distinct open user goals and weighted targets, even with no such captures')
    b.add_argument('--open-goal-scope', choices=('vibestudio', 'all'), default='vibestudio',
                   help='chains the open-goal minimum is measured over: vibestudio (default: the Vibestudio tasks\' '
                        'chains, so a mixed expert-iteration build holds its open goals to the contract against the '
                        'historical Vibestudio assignments, not against every other part) or all (every chain)')
    b.add_argument('--unfilled-live-teacher', choices=('auto', 'drop', 'train'), default='auto',
                   help='observed-live teacher episodes without a filled reasoning unit (--aux ...,replaces=teacher): '
                        'drop (they train only with the student\'s filled reasoning), train (their actions after the '
                        'teacher\'s think block as context, like other teacher episodes) or auto (default: drop when '
                        'a replaces=teacher part is given, else train)')
    b.add_argument('--open-goals-only', action='store_true',
                   help='assemble only independently verified open user goal trajectories')
    b.add_argument('--base-rates', default=resolve_str(*OPS, 'exp1-rft', 'base-pass-rates.json'))
    b.add_argument('--max-per-task', type=int, default=4,
                   help='episodes of a task the start policy never solves; a task it always solves keeps 1')
    b.add_argument('--rich-max-per-task', type=int, default=8,
                   help='the per-task cap for tasks whose program declares a held construct (HELD_CONSTRUCTS)')
    b.add_argument('--wrapper-share', type=float, default=0.15,
                   help='dataset-wrapper episodes kept to at most this share of the GRPO-pool selection (1 = no cap)')
    b.add_argument('--max-family-share', type=float, default=0.05)
    b.add_argument('--dup-threshold', type=float, default=0.8)
    b.add_argument('--seed', type=int, default=20261117)
    b.add_argument('--teacher-pool', action='append', default=None,
                   help='teacher pools whose current manifests the extraction must cover (default: teacher-gen pool-v1)')
    b.add_argument('--allow-stale-teacher', action='store_true')
    b.add_argument('--october', help='directory of October candidate rows (GROUP.jsonl.zst, eval-check/GROUP.jsonl)')
    b.add_argument('--october-group', action='append', metavar='GROUP=WEIGHT',
                   help='October groups taken and their budget weights (default: %s)' % ', '.join(DEFAULT_OCTOBER_GROUPS))
    b.add_argument('--october-share', type=float, default=0.4,
                   help='October share of the forward tokens (prompt + reply of every call)')
    b.add_argument('--october-max-per-task', type=int, default=1)
    b.add_argument('--october-passes', type=int, default=4,
                   help='selection passes: each renders, measures actual forward tokens and refills toward the budget')
    b.add_argument('--october-family-share', type=float, default=0.05)
    b.add_argument('--teacher-lead-in', choices=TEACHER_LEAD_IN, default='context',
                   help='text a non-student reply writes before its first tool call: context (default) or train')
    b.add_argument('--exclude-run', action='append', default=[], metavar='REGEX',
                   help="own successes of runs matching REGEX stay out (e.g. 'grpo-v2/'; run labels 'rl:grpo-v3/run1')")
    b.add_argument('--guided', choices=('train', 'exclude'), default='train',
                   help='JOLT+ leak-checked guided attempts (own, student form, jolt-guided tier; async-train §8): '
                        'train (default, between own successes and teacher trajectories) or exclude (opt-out)')
    b.add_argument('--backend-workarounds', choices=('context', 'context-refused-only', 'exclude', 'train'),
                   default='context',
                   help="teacher episodes that met the text backend's Neuralese refusal: context (default: every call "
                        "whose prompt holds the refusal, and every call of an invocation the detour started after it, "
                        "is zero weight; the calls before it, e.g. the producer's own return, train), "
                        "context-refused-only (only the calls whose prompt holds the refusal), exclude (the whole "
                        "episode) or train")
    b.add_argument('--own-empty-think', choices=('context', 'train'), default='context',
                   help="an own reply without reasoning (a decision readout the actor scored, not sampled): its "
                        "template empty think block is context (default) or trains")
    b.add_argument('--teacher-forward-budget', type=int, default=0,
                   help='forward tokens of teacher episodes, taken breadth-first over tasks (0: every capped episode)')
    b.add_argument('--own-loss-share', type=float, default=0.25,
                   help='share of the weighted target tokens taken by the student\'s own successes (0: every capped one)')
    b.add_argument('--max-task-target-tokens', type=int, default=0,
                   help='per task cap on weighted target tokens that own successes may fill up to (0: none)')
    b.add_argument('--prefer-unseen', action='append', default=[], metavar='BUILD_OUT',
                   help='an earlier build the start checkpoint already trained (its OUT dir): its episodes go after '
                        'every untrained one at each cap and budget, and its tasks stay out of validation (for SFT from '
                        'a checkpoint descended from that build; repeatable)')
    b.add_argument('--prefer-unseen-scope', choices=PREFER_UNSEEN_SCOPES, default='family',
                   help='family (default): each family takes its breadth-first turns, untrained tasks first within '
                        'its turn, then tasks an earlier build trained (a second epoch where a family has nothing '
                        'fresh); global: every untrained episode before any trained one (sft-v7: the budget went to '
                        'the unseen tail of 20 generator families; DECISIONS 2026-11-22)')
    b.add_argument('--val-episodes', type=int, default=40,
                   help='left-over own and teacher successes on untrained tasks written as OUT/val-{own,teacher}.pkl')
    b.add_argument('--aux', action='append', default=[],
                   metavar='NAME=PATH[,share=S|all][,task_tokens=N][,think=context|train]',
                   help='a prebuilt chains.pkl taking a share of the weighted target tokens (parse_aux)')
    b.add_argument('--teacher-think', choices=TEACHER_THINK, default='context',
                   help='think block of replies the student did not sample (teacher, October, live): context (default: '
                   'zero weight through </think>, the action is the target) or train (visible reasoning is target; '
                   'an empty think block stays context). think_context has the evidence. The default treatment: '
                   '--think-treatment chooses per teacher model, source or episode')
    b.add_argument('--think-treatment', default=None, metavar='JSON|none',
                   help='teacher think treatment per model/source/episode (ThinkTreatment; default: the registry role '
                        f'{THINK_TREATMENT_ROLE} when registered, else --teacher-think for every episode)')
    b.add_argument('--frontier', help='expert_iteration.py frontier JSON: per-task pass rates and tiers of the policy '
                   'this build trains; each task keeps its cheapest verified source (frontier_drop)')
    b.add_argument('--previous-policy', help='adjacent previous policy SHA-256 whose own successes share top priority '
                   'with --frontier policy (async-train §3; default: frontier.json previous_policy)')
    b.add_argument('--own-render', choices=('recorded', 'replayed'), default='recorded',
                   help="own successes (rl-own, jolt-guided) render the turns they were sampled with (recorded_turns.py); "
                        "'replayed': the replaying runtime's record of them, as before 2026-11-23")
    b.add_argument('--admit-drifted', choices=('on', 'off'), default='on',
                   help='teacher successes whose replay display drifted (sft_sources.drifted_success) join the exact ones')
    b.add_argument('--anchor-per-task', type=int, default=1,
                   help='own episodes of a task the frontier marks solved (a light anchor)')
    b.add_argument('--workers', type=int, default=default_workers(),
                   help='processes rendering candidates ahead of the sequential selection (RenderAhead; the output is '
                        'identical for every value); 1: the serial build')
    n = sub.add_parser('neuralese-text', help='text rows (producers, calls before an import) of Neuralese successes')
    n.add_argument('--work', required=True)
    n.add_argument('--workers', type=int, default=default_workers(), help='processes reading replays and episodes')
    n.add_argument('--out', required=True)
    n.add_argument('--tokenizer', required=True)
    n.add_argument('--held-manifest', help='default: the registry role held-eval (held-v3)')
    n.add_argument('--max-per-task', type=int, default=4)
    n.add_argument('--current-surface-run', action='append', default=[], metavar='REGEX',
                   help='own runs on the current runtime (episode run label): their calls count as on a current surface')
    n.add_argument('--seed', type=int, default=20261117)
    g = sub.add_parser('neuralese-graph', help='whole Neuralese episodes (consumers included) as pass-through graph rows')
    g.add_argument('--work', required=True)
    g.add_argument('--workers', type=int, default=default_workers(), help='processes reading replays and episodes')
    g.add_argument('--out', required=True)
    g.add_argument('--held-manifest', help='default: the registry role held-eval (held-v3)')
    g.add_argument('--max-per-task', type=int, default=2)
    g.add_argument('--current-surface-run', action='append', default=[], metavar='REGEX',
                   help='own runs on the current runtime (episode run label): their calls count as on a current surface')
    g.add_argument('--exclude-run', action='append', default=[], metavar='REGEX',
                   help='own runs left out (as build --exclude-run), matched against the episode run label')
    g.add_argument('--seed', type=int, default=20261117)
    args = p.parse_args()
    if hasattr(args, 'held_manifest') and not args.held_manifest:
        args.held_manifest = data_registry.role_path('held-eval')
    if args.command == 'build' and args.teacher_pool is None:
        args.teacher_pool = [] if args.no_default_teacher_pool else [resolve_str(*OPS, 'teacher-gen-20261115', 'pool-v1')]
    {'extract': extract, 'replay': replay, 'check-render': check_render, 'build': build,
     'neuralese-text': neuralese_text, 'neuralese-graph': neuralese_graph}[args.command](args)


if __name__ == '__main__':
    main()
