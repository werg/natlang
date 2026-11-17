#!/usr/bin/env python3
"""Build ordinary LFM3 SFT chains from replay-verified Vibestudio teacher episodes.

The collection manifest binds the episode file and the complete suite split inventory.
Only successful training roots enter this build. Rendering, reasoning-free think-prefix
masking and prefix-sharing chains use the current SFT builder's implementations.
No model is loaded and no source, template checkout or live run is modified.

  vibestudio_sft.py --episodes episodes.jsonl --manifest collection.json --out OUT --tokenizer TOK
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import pickle
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / 'training/neuralese'), str(REPO)]

from sft_sources import (TemplateRenderer, ZstdWriter, held_closure, held_hit,
                         reasoning_free_context, sha256_file, zstd_lines, visible_key, action_text)

EPISODE_SCHEMA = 'vibestudio.teacher-episode/1'
MANIFEST_SCHEMA = 'vibestudio.teacher-collection/1'
SHA256 = re.compile(r'^[0-9a-f]{64}$')
COMMIT = re.compile(r'^[0-9a-f]{40}$')


def digest(value):
    """The collector's canonical JSON digest (sorted object keys, UTF-8)."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def request_hash(turn):
    request = {'messages': turn['messages'], 'tools': turn['tools']}
    # Preserve the collector's exact JSON bytes: JS and Python spell exponents
    # differently (1e-7 vs 1e-07). The parsed content must still bind this turn.
    if 'request_canonical' in turn:
        canonical = turn['request_canonical']
        if not isinstance(canonical, str) or digest(json.loads(canonical)) != digest(request):
            raise ValueError('request_canonical does not describe this exact request')
        return hashlib.sha256(canonical.encode()).hexdigest()
    return digest(request)


def nonempty_strings(value, name):
    if (not isinstance(value, list) or not value or
            any(not isinstance(v, str) or not v.strip() for v in value) or len(set(value)) != len(value)):
        raise ValueError(f'{name} must be a nonempty list of unique strings')
    return value


def case_key(case):
    suite = case.get('suite') or {}
    if not isinstance(suite.get('id'), str) or not suite['id'] or not SHA256.fullmatch(suite.get('hash', '')):
        raise ValueError('case needs suite id and immutable suite hash')
    if not isinstance(case.get('id'), str) or not case['id']:
        raise ValueError('case needs an id')
    return suite['id'], suite['hash'], case['id']


def inventory(manifest):
    if manifest.get('schema') != MANIFEST_SCHEMA:
        raise ValueError(f'expected {MANIFEST_SCHEMA}')
    cases = manifest.get('cases')
    if not isinstance(cases, list) or not cases:
        raise ValueError('manifest needs the complete case split inventory')
    known, groups, inputs = {}, collections.defaultdict(set), collections.defaultdict(set)
    for case in cases:
        key = case_key(case)
        if key in known:
            raise ValueError(f'duplicate case inventory entry: {key}')
        if case.get('split') not in ('train', 'validation', 'test'):
            raise ValueError(f'{key}: explicit train/validation/test split required')
        if not isinstance(case.get('group'), str) or not case['group']:
            raise ValueError(f'{key}: source group required')
        if not SHA256.fullmatch(case.get('input_sha256', '')):
            raise ValueError(f'{key}: input_sha256 required in the complete split inventory')
        for field in ('source_ids', 'source_groups', 'source_revisions'):
            nonempty_strings(case.get(field), field)
        known[key] = case
        for group in case['source_groups']:
            groups[group].add(case['split'])
        inputs[(key[:2], case['input_sha256'])].add(case['split'])
    contaminated = {group for group, splits in groups.items() if len(splits) > 1}
    duplicate_inputs = {key for key, splits in inputs.items() if len(splits) > 1}
    return known, contaminated, duplicate_inputs


def episode_reason(episode, cases, contaminated, duplicate_inputs, held):
    """Return an admission rejection reason; malformed provenance is a hard error."""
    if episode.get('schema') != EPISODE_SCHEMA:
        raise ValueError(f'expected {EPISODE_SCHEMA}')
    if not isinstance(episode.get('id'), str) or not episode['id']:
        raise ValueError('episode needs an id')
    source = episode.get('source') or {}
    if source.get('dirty') is not False:
        raise ValueError('episode source must attest a clean pinned tree')
    for field in ('template_commit', 'natlang_commit', 'vibestudio_commit'):
        if not COMMIT.fullmatch(source.get(field, '')):
            raise ValueError(f'source.{field} needs an exact git commit')
    if not SHA256.fullmatch(source.get('program_sha256', '')) or not source.get('template'):
        raise ValueError('source needs template and immutable program hash')
    teacher = episode.get('teacher') or {}
    if not teacher.get('provider') or not teacher.get('model_id') or not teacher.get('identity'):
        raise ValueError('teacher provider, model_id and identity required')
    case = episode.get('case') or {}
    key = case_key({**case, 'suite': episode.get('suite')})
    if key not in cases:
        raise ValueError(f'episode case absent from complete inventory: {key}')
    expected = cases[key]
    for field in ('split', 'group'):
        if case.get(field) != expected.get(field):
            raise ValueError(f'episode {field} does not match inventory')
    if not SHA256.fullmatch(case.get('input_sha256', '')):
        raise ValueError('case.input_sha256 required')
    if expected.get('input_sha256') and case['input_sha256'] != expected['input_sha256']:
        raise ValueError('episode input digest does not match inventory')
    for field in ('source_ids', 'source_groups', 'source_revisions'):
        if nonempty_strings(episode.get(field), field) != expected[field]:
            raise ValueError(f'episode {field} does not match inventory')
    if source['template_commit'] not in episode['source_revisions']:
        raise ValueError('source revision does not bind template commit')
    if case['split'] != 'train':
        return 'protected_split'
    if set(episode['source_groups']) & contaminated:
        return 'source_group_crosses_splits'
    if (key[:2], case['input_sha256']) in duplicate_inputs:
        return 'input_crosses_splits'
    if any(held_hit(task_id, episode['source_ids'], episode['source_groups'], held)
           for task_id in [case['id'], *episode['source_ids']]):
        return 'held_excluded'
    verification = episode.get('verification') or {}
    if (episode.get('rejection') is not None or verification.get('verdict') != 'accepted' or
            not isinstance(verification.get('reason'), str) or not verification['reason'].strip() or
            verification.get('method') != 'independent-model-review' or
            not all((verification.get('teacher') or {}).get(field) for field in ('provider', 'model_id', 'identity'))):
        return 'semantic_verification_missing_or_rejected'
    outcome = episode.get('outcome') or {}
    gates = outcome.get('gates')
    if outcome.get('kind') != 'returned' or type(outcome.get('quality')) not in (float, int) or outcome['quality'] != 1:
        return 'not_full_success'
    if not isinstance(gates, dict) or not gates or any(value is not True for value in gates.values()):
        return 'gate_failed'
    if type(outcome.get('tool_error_count')) is not int or outcome['tool_error_count'] != 0:
        return 'tool_errors_or_missing_evidence'
    turns = episode.get('turns')
    if not isinstance(turns, list) or not turns:
        return 'no_model_calls'
    hashes = []
    for turn in turns:
        if not isinstance(turn.get('messages'), list) or not turn['messages'] or not isinstance(turn.get('tools'), list):
            raise ValueError('turn needs exact messages and tools')
        target = turn.get('target') or {}
        if target.get('role') != 'assistant' or not (target.get('content') or target.get('tool_calls')):
            return 'empty_or_nonassistant_target'
        if not isinstance(turn.get('invocation_id'), str) or not turn['invocation_id']:
            raise ValueError('turn needs invocation identity')
        hashes.append(request_hash(turn))
    replay = episode.get('replay') or {}
    if replay.get('verified') is not True or replay.get('request_hashes') != hashes:
        return 'replay_not_verified'
    return None


def admit(episodes, manifest, held, max_per_case=4):
    cases, contaminated, duplicate_inputs = inventory(manifest)
    accepted, census, ids, signatures, per_case = [], collections.Counter(), set(), set(), collections.Counter()
    for episode in episodes:
        census['seen'] += 1
        if episode.get('id') in ids:
            raise ValueError(f'duplicate episode id: {episode.get("id")}')
        ids.add(episode.get('id'))
        reason = episode_reason(episode, cases, contaminated, duplicate_inputs, held)
        if reason:
            census[reason] += 1
            continue
        key = case_key({**episode['case'], 'suite': episode['suite']})
        # Attempt-specific invocation/tool IDs are provenance, not new examples.
        signature = digest({'source_ids': episode['source_ids'], 'program': episode['source']['program_sha256'],
                            'requests': [(visible_key(t['messages']), t['tools']) for t in episode['turns']],
                            'actions': action_text(episode['turns'])})
        if signature in signatures:
            census['duplicate_trajectory'] += 1
            continue
        if per_case[key] >= max_per_case:
            census['case_cap'] += 1
            continue
        signatures.add(signature)
        per_case[key] += 1
        census['admitted'] += 1
        accepted.append(episode)
    return accepted, dict(census)


def load_collections(args):
    specs = [(Path(directory) / 'manifest.json', None) for directory in getattr(args, 'collection', [])]
    if getattr(args, 'manifest', None):
        specs.append((Path(args.manifest), Path(args.episodes) if getattr(args, 'episodes', None) else None))
    if not specs:
        raise ValueError('at least one --collection or --manifest is required')
    episodes, inventories, bindings, seen_paths = [], {}, [], set()
    for manifest_path, explicit_episodes in specs:
        if manifest_path.resolve() in seen_paths:
            raise ValueError(f'collection supplied twice: {manifest_path}')
        seen_paths.add(manifest_path.resolve())
        manifest = json.loads(manifest_path.read_text())
        # Validate each inventory before de-duplicating repeated suites across
        # independent collection attempts. A case's protected split never changes.
        known, _, _ = inventory(manifest)
        declared_suite = manifest.get('suite') or {}
        suite_identity = {field: declared_suite.get(field) for field in ('id', 'hash')}
        if (not suite_identity['id'] or not SHA256.fullmatch(suite_identity.get('hash') or '') or
                not isinstance(manifest.get('source'), dict) or not manifest['source'] or
                not isinstance(manifest.get('teacher'), dict) or not manifest['teacher']):
            raise ValueError('collection manifest needs suite, source and teacher identity bindings')
        for key, case in known.items():
            if case['suite'] != suite_identity:
                raise ValueError('case inventory suite does not match collection manifest')
            if key in inventories and digest(inventories[key]) != digest(case):
                raise ValueError(f'case inventory changed between collections: {key}')
            inventories[key] = case
        binding = manifest.get('episodes') or {}
        if not isinstance(binding.get('path'), str) or not binding['path']:
            raise ValueError('collection manifest needs an episode file binding')
        bound_path = Path(binding['path'])
        if not bound_path.is_absolute():
            bound_path = manifest_path.parent / bound_path
        episodes_path = explicit_episodes or bound_path
        if bound_path.resolve() != episodes_path.resolve():
            raise ValueError('collection manifest binds a different episode path')
        if binding.get('sha256') != sha256_file(episodes_path):
            raise ValueError('episode file does not match collection manifest')
        for line in zstd_lines(episodes_path):
            if not line.strip():
                continue
            episode = json.loads(line)
            # The clean commit and program/model identities are collection-wide
            # attestations, not merely syntactically valid labels on each row.
            # This catches a mixed/copied episode even if the new file SHA was
            # accidentally written into the manifest during a resume.
            for field, expected in (('suite', suite_identity), ('source', manifest['source']),
                                    ('teacher', manifest['teacher'])):
                if digest(episode.get(field)) != digest(expected):
                    raise ValueError(f'episode {field} identity does not match collection manifest')
            episodes.append(episode)
        bindings.append({'episodes': str(episodes_path.resolve()), 'episodes_sha256': sha256_file(episodes_path),
                         'manifest': str(manifest_path.resolve()), 'manifest_sha256': sha256_file(manifest_path)})
    return episodes, {'schema': MANIFEST_SCHEMA, 'cases': list(inventories.values())}, bindings


def build(args, renderer=None):
    import numpy as np
    from exp0_sft import MAX_LEN, chains_of
    from natlang_neuralese.common.code_provenance import code_provenance

    out = Path(args.out)
    episodes, manifest, bindings = load_collections(args)
    accepted, census = admit(episodes, manifest, held_closure(args.held_manifest), args.max_per_case)
    if not accepted:
        raise ValueError(f'no admissible training episodes: {census}')
    if out.exists():
        raise ValueError(f'output already exists; choose a new immutable build path: {out}')
    out.mkdir(parents=True)
    # Keep the exact closed collection inputs with the published data. No source
    # hardlinks: ctime changes would disturb readers, and sources may be mutable.
    for index, binding in enumerate(bindings):
        directory = out / 'collections' / f'{index:04d}'
        directory.mkdir(parents=True)
        for field, filename in (('manifest', 'manifest.json'), ('episodes', 'episodes.jsonl')):
            source_path, snapshot = Path(binding[field]), directory / filename
            shutil.copyfile(source_path, snapshot)
            expected = binding[f'{field}_sha256']
            if sha256_file(snapshot) != expected or sha256_file(source_path) != expected:
                raise ValueError(f'collection input changed while snapshotting: {source_path}')
            binding[f'{field}_snapshot'] = str(snapshot.relative_to(out))
    renderer = renderer or TemplateRenderer(args.tokenizer)
    writer, rows, sft_episodes = ZstdWriter(out / 'rows.jsonl.zst'), [], []
    try:
        for episode in accepted:
            token_rows, semantic_rows = [], []
            for index, turn in enumerate(episode['turns']):
                prompt, target = renderer.render(turn['messages'], turn['tools'], turn['target'])
                if not target or len(prompt) + len(target) > MAX_LEN:
                    raise ValueError(f'{episode["id"]}: a complete call cannot fit the learner; no silent truncation')
                context = reasoning_free_context(renderer, 'teacher', turn['target'], target)
                if context >= len(target):
                    raise ValueError(f'{episode["id"]}: call has no supervised tokens')
                row_id = hashlib.sha256(f'{episode["id"]}#{index}'.encode()).hexdigest()[:24]
                row = {'id': row_id, 'invocation': turn['invocation_id'], 'index': index,
                       'prompt': np.asarray(prompt, np.int32), 'sample': np.asarray(target, np.int32)}
                if context:
                    row['weight'] = (np.arange(len(target)) >= context).astype(np.float32)
                token_rows.append(row)
                semantic_rows.append({'id': row_id, 'episode': episode['id'], 'source': 'vibestudio-teacher',
                                      'task_id': episode['source_ids'][0], 'family': episode['suite']['id'],
                                      'teacher': episode['teacher'], 'call': index,
                                      'invocation': turn['invocation_id'], 'messages': turn['messages'],
                                      'tools': turn['tools'], 'target': turn['target'],
                                      'prompt_tokens': len(prompt), 'target_tokens': len(target), 'loss_from': context,
                                      'source_ids': episode['source_ids'], 'source_groups': episode['source_groups'],
                                      'source_revisions': episode['source_revisions'], 'provenance': episode['source']})
            for row in semantic_rows:
                writer.write(row)
            rows.extend(semantic_rows)
            sft_episodes.append({'task_id': episode['source_ids'][0], 'round': episode['id'], 'rows': token_rows})
        writer.close()
        chains = chains_of(sft_episodes)
        with (out / 'chains.pkl.partial').open('wb') as handle:
            pickle.dump({'chains': chains, 'schema': 'natlang.sft-chains/1'}, handle, protocol=4)
        os.replace(out / 'chains.pkl.partial', out / 'chains.pkl')
        report = {'schema': 'natlang.vibestudio-sft-build/1', 'code': code_provenance(),
                  'inputs': {'collections': bindings, **(bindings[0] if len(bindings) == 1 else {}),
                             'held_manifest': str(Path(args.held_manifest).resolve()),
                             'held_manifest_sha256': sha256_file(args.held_manifest), 'renderer': renderer.identity},
                  'policy': {'split': 'train', 'quality': 1, 'all_gates': True, 'exact_replay': True,
                             'semantic_verification': 'accepted', 'tool_error_count': 0,
                             'max_per_case': args.max_per_case,
                             'loss': 'own assistant output only; empty teacher think prefix zero weight'},
                  'funnel': census, 'episodes': len(accepted), 'calls': len(rows), 'chains': len(chains),
                  'tasks': len({e['source_ids'][0] for e in accepted}),
                  'forward_tokens': sum(len(c['ids']) for c in chains),
                  'target_tokens': sum(r['target_tokens'] for r in rows),
                  'weighted_target_tokens': sum(r['target_tokens'] - r['loss_from'] for r in rows),
                  'outputs': {name: sha256_file(out / name) for name in ('rows.jsonl.zst', 'chains.pkl')}}
        (out / 'build.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        return report
    except BaseException:
        # A failed directory is deliberately retained as evidence, without a build.json
        # completion receipt; subsequent builds require a fresh output path.
        if not writer.writer.closed:
            writer.writer.close()
        raise


def registration_entry(out, ident, owner):
    """Validate a closed build and describe it to the shared immutable publisher."""
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', ident):
        raise ValueError('register id must be one safe corpus path component')
    if owner not in ('dgx', 'pop'):
        raise ValueError('registration needs the actual data owner: dgx or pop')
    out = Path(out).resolve()
    report = json.loads((out / 'build.json').read_text())
    if report.get('schema') != 'natlang.vibestudio-sft-build/1' or report.get('episodes', 0) < 1:
        raise ValueError('registration requires a completed nonempty Vibestudio SFT build')
    for filename in ('chains.pkl', 'rows.jsonl.zst'):
        if report.get('outputs', {}).get(filename) != sha256_file(out / filename):
            raise ValueError(f'built output changed: {filename}')
    collections_ = report.get('inputs', {}).get('collections')
    if not isinstance(collections_, list) or not collections_:
        raise ValueError('build has no immutable collection snapshots')
    for collection in collections_:
        for field in ('manifest', 'episodes'):
            rel = collection.get(f'{field}_snapshot')
            if not isinstance(rel, str) or not rel or not (out / rel).resolve().is_relative_to(out):
                raise ValueError('snapshot path must stay inside the corpus')
            if sha256_file(out / rel) != collection.get(f'{field}_sha256'):
                raise ValueError(f'closed collection snapshot changed: {rel}')
    # External NVMe data is addressed through the same gitignored runs link the
    # rest of the corpus registry uses. Publication handles the link, never copies
    # machine-specific absolute storage paths into a transported manifest.
    try:
        relative = str(out.relative_to(REPO))
        links = {}
    except ValueError:
        relative = f'runs/{ident}'
        links = {relative: str(out)}
    entry = {'id': ident, 'owner': owner, 'path': relative,
             'include': ['build.json', 'chains.pkl', 'rows.jsonl.zst', 'collections/**/*'],
             'admission': 'sft-experiment-input',
             'license_provenance': 'Project-generated teacher execution trajectories on Vibestudio template programs '
                 'and fixture scenarios. Exact template revisions, teacher providers/model identities and source '
                 'collection manifests are retained in collections/ and per-row provenance; program source licensing '
                 'follows the corresponding pinned template repositories.',
             'stages': {'collected_episodes': report['funnel'].get('seen', 0),
                        'admitted_episodes': report['episodes'], 'rendered_chains': report['chains'],
                        'learner_exposure': 'not performed by collection, build or registration'},
             'source_collections': [{key: value for key, value in binding.items() if key.endswith('_sha256')}
                                    for binding in collections_],
             'note': f'Vibestudio verified teacher SFT: {report["episodes"]} admitted episodes, '
                 f'{report["calls"]} executor calls, {report["chains"]} chains, '
                 f'{report["weighted_target_tokens"]} weighted target tokens. Only train roots, quality 1, '
                 'independent semantic review, exact fixture replay, no failed tool actions; held-v3/v2 and '
                 'protected source groups excluded. exp0_sft.py train --chains input for the next experiment pin; '
                 'registration does not assert actual learner exposure.'}
    return entry, links


def register_build(out, ident, owner):
    from register_corpora import register
    entry, links = registration_entry(out, ident, owner)
    return register([entry], f'Register verified Vibestudio teacher corpus {ident}', links=links)


def main():
    from natlang_neuralese.common.paths import resolve_str
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', nargs='?', choices=('build', 'register'), default='build')
    parser.add_argument('--collection', action='append', default=[],
                        help='collection directory with manifest.json; repeat to build one corpus across suites')
    parser.add_argument('--episodes', help='explicit episode file (requires --manifest; default its bound path)')
    parser.add_argument('--manifest', help='collection manifest; may be combined with --collection')
    parser.add_argument('--out', required=True)
    parser.add_argument('--tokenizer', help='required for build; unused when registering a completed build')
    parser.add_argument('--register-id', help='publish immutable corpus registry/manifests after build, or with register')
    parser.add_argument('--register-owner', choices=('dgx', 'pop'),
                        help='actual machine holding the files; required with --register-id')
    parser.add_argument('--held-manifest', default=resolve_str('data_nvme', 'lfm3-rl-dgx-20261029',
                                                              'ops', 'held-v3', 'manifest.json'))
    parser.add_argument('--max-per-case', type=int, default=4)
    args = parser.parse_args()
    if bool(args.register_id) != bool(args.register_owner):
        parser.error('--register-id and --register-owner are required together')
    if args.command == 'register':
        if not args.register_id:
            parser.error('register requires --register-id and --register-owner')
        print(json.dumps({'registered': register_build(args.out, args.register_id, args.register_owner)}))
        return
    if args.max_per_case < 1:
        parser.error('--max-per-case must be positive')
    if args.episodes and not args.manifest:
        parser.error('--episodes requires --manifest')
    if not args.collection and not args.manifest:
        parser.error('at least one --collection or --manifest is required')
    if not args.tokenizer:
        parser.error('build requires --tokenizer')
    print(json.dumps(build(args), indent=2))
    if args.register_id:
        register_build(args.out, args.register_id, args.register_owner)


if __name__ == '__main__':
    main()
