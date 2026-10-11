#!/usr/bin/env python3
"""Typed decision cases and skill episodes from public classification and rating data. No model calls.

A decision case is one state (text) with one typed question, as in decision models such as Jev, Clef or Strands
Decider: `choice` (one of N options), `noul` (yes/no, answered with a probability) or `score` (a position on an ordered
scale). Gold answers come from the datasets; several are numeric (toxicity fractions, mean formality and hate-speech
ratings), so the targets are soft and the scores continuous:

  choice -> choice-brier (probabilities over the options)
  noul   -> binary-brier (a probability against a label or a target frequency)
  score  -> ordinal-rps  (a distribution, level or fractional position against a level or fractional target)

Outputs:
  decision-cases.jsonl     every case (train from the dataset's training split, held-out from its test/validation),
                           for distillation labels from decision models and for typed-decision training;
  decision-episodes.jsonl  skill episodes (4 support, 4 query, 4 transfer from another family of the same kind).
"""
import argparse
import ast
import glob
import hashlib
import json
import math
import os
import re
import sys
from collections import defaultdict

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'training', 'neuralese'))
from episode_lib import digest, group_commitment, run_gate  # noqa: E402
from decision_task_contracts import criteria_for  # noqa: E402
from natlang_neuralese.common.paths import resolve_str  # noqa: E402

ROOT = resolve_str('data_hdd', 'natlang-development-data', 'data', 'decision-sources')
SOURCE_FILES_READ = {}




MAX_ROWS = 60000


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def load_source_pins(path):
    """Load explicit upstream pins and require their shard hashes to bind the local bytes.

    Receipt shape: {"schema":"natlang.decision-source-pins/1","datasets":{
      "repo/name":{"revision":"full-immutable-revision","config":"optional",
        "shards":[{"local_path":"repo__name/train/shard.parquet",
                   "upstream_path":"data/train-....parquet","lfs_sha256":"64 hex chars"}]}}}.
    Local paths are relative to ROOT. Missing dataset receipts stay unpinned; a receipt for a dataset must bind
    every local shard encountered for it.
    """
    if not path:
        return None, None
    receipt_path = os.path.realpath(path)
    with open(receipt_path, 'rb') as stream:
        receipt_bytes = stream.read()
    receipt = json.loads(receipt_bytes)
    if receipt.get('schema') != 'natlang.decision-source-pins/1' or not isinstance(receipt.get('datasets'), dict):
        raise ValueError('source pin receipt must use natlang.decision-source-pins/1 with a datasets object')
    pins = {}
    for repo, dataset in receipt['datasets'].items():
        revision = dataset.get('revision') if isinstance(dataset, dict) else None
        if not isinstance(revision, str) or not re.fullmatch(r'[0-9a-fA-F]{40}', revision):
            raise ValueError(f'{repo}: source pin revision must be an immutable 40-character Hugging Face commit SHA')
        shards = dataset.get('shards')
        if not isinstance(shards, list):
            raise ValueError(f'{repo}: source pin needs a shards array')
        by_path = {}
        for shard in shards:
            if not isinstance(shard, dict) or not isinstance(shard.get('local_path'), str):
                raise ValueError(f'{repo}: each pinned shard needs local_path')
            local_path = _canonical_relative_path(shard['local_path'], f'{repo} local_path')
            upstream_path = shard.get('upstream_path')
            declared_shas = [shard[key] for key in ('lfs_sha256', 'sha256') if shard.get(key) is not None]
            if not isinstance(upstream_path, str):
                raise ValueError(f'{repo}:{local_path}: source pin needs upstream_path')
            upstream_path = _canonical_relative_path(upstream_path, f'{repo} upstream_path')
            if not declared_shas or any(not isinstance(value, str) or len(value) != 64 or
                                        any(c not in '0123456789abcdefABCDEF' for c in value)
                                        for value in declared_shas):
                raise ValueError(f'{repo}:{local_path}: source pin needs a 64-character SHA-256/LFS OID')
            if len({value.lower() for value in declared_shas}) != 1:
                raise ValueError(f'{repo}:{local_path}: sha256 and lfs_sha256 declarations disagree')
            pinned_sha = declared_shas[0]
            if local_path in by_path:
                raise ValueError(f'{repo}: duplicate pinned local shard {local_path}')
            by_path[local_path] = {**shard, 'local_path': local_path, 'pinned_sha256': pinned_sha.lower()}
        pins[repo] = {'revision': dataset['revision'], 'config': dataset.get('config'), 'shards': by_path}
    return pins, {'path': receipt_path, 'sha256': hashlib.sha256(receipt_bytes).hexdigest()}


def _canonical_relative_path(path, label):
    """Require a canonical slash-separated relative path without traversal or platform-specific aliases."""
    if not isinstance(path, str) or not path or path.startswith('/') or '\\' in path:
        raise ValueError(f'{label} must be a nonempty canonical relative path')
    parts = path.split('/')
    if any(part in ('', '.', '..') for part in parts):
        raise ValueError(f'{label} must not contain empty, dot, or parent path components')
    return path


def rows(repo, splits, source_pins=None):
    """Rows of the named splits, label metadata, and physical Parquet row references.

    At most MAX_ROWS rows per split, taken from row groups spread evenly over the file: some sources are sorted by
    label, and reading whole large shards into Python objects would cost gigabytes for no gain.
    """
    out, names, refs = [], {}, []
    for split in splits:
        for path in sorted(glob.glob(f'{ROOT}/{repo.replace("/", "__")}/**/{split}/*.parquet', recursive=True)):
            parquet = pq.ParquetFile(path)
            groups = parquet.metadata.num_row_groups
            per = max(1, parquet.metadata.num_rows // groups)
            wanted = min(groups, max(1, MAX_ROWS // per))
            picks = sorted({round(i * (groups - 1) / max(1, wanted - 1)) for i in range(wanted)}) if wanted > 1 else [0]
            meta = parquet.schema_arrow.metadata or {}
            if b'huggingface' in meta:
                for key, feature in json.loads(meta[b'huggingface']).get('info', {}).get('features', {}).items():
                    if isinstance(feature, dict) and feature.get('names'):
                        names[key] = feature['names']
            logical_root = os.path.abspath(ROOT)
            logical_path = os.path.abspath(path)
            if os.path.commonpath([logical_root, logical_path]) != logical_root:
                raise ValueError(f'{path}: parquet source is outside configured source root {ROOT}')
            local_path = _canonical_relative_path(os.path.relpath(logical_path, logical_root).replace(os.sep, '/'),
                                                  f'{repo} local_path')
            # Keep the logical cache path above, but hash the actual bytes reached through any symlink.
            byte_path = os.path.realpath(path)
            file_sha = _file_sha256(byte_path)
            pin = (source_pins or {}).get(repo)
            pinned_shard = (pin or {}).get('shards', {}).get(local_path)
            if pinned_shard:
                expected_sha = pinned_shard['pinned_sha256']
                if file_sha != expected_sha:
                    raise ValueError(f'{repo}:{local_path}: local bytes {file_sha} do not match pinned upstream shard {expected_sha}')
            elif pin:
                raise ValueError(f'{repo}:{local_path}: source pin receipt does not bind this local shard')
            source_file_key = (repo, local_path)
            source_file = SOURCE_FILES_READ.setdefault(source_file_key, {
                'dataset': repo, 'local_path': local_path, 'local_file_sha256': file_sha,
                'splits_read': [], 'sampled_row_groups': [], 'upstream': None,
                'upstream_metadata_status': 'unknown_requires_pinned_source_receipt'
            })
            if source_file['local_file_sha256'] != file_sha:
                raise ValueError(f'{repo}:{local_path}: local file changed while building the corpus')
            if split not in source_file['splits_read']:
                source_file['splits_read'].append(split)
            source_file['sampled_row_groups'] = sorted(set(source_file['sampled_row_groups']) | set(picks))
            if pin and pinned_shard:
                upstream = {
                    'revision': pin['revision'], 'config': pin.get('config'),
                    'path': pinned_shard['upstream_path'], 'lfs_sha256': pinned_shard['pinned_sha256']
                }
                if source_file['upstream'] not in (None, upstream):
                    raise ValueError(f'{repo}:{local_path}: conflicting upstream source pins')
                source_file['upstream'] = upstream
                source_file['upstream_metadata_status'] = 'pinned_and_byte_bound'
            physical_starts = []
            physical = 0
            for group in range(groups):
                physical_starts.append(physical)
                physical += parquet.metadata.row_group(group).num_rows
            for group in picks:
                group_rows = parquet.read_row_group(group).to_pylist()
                for row_in_group, item in enumerate(group_rows):
                    out.append(item)
                    ref = {
                        'dataset': repo,
                        'split': split,
                        'local_path': local_path,
                        'local_file_sha256': file_sha,
                        'row_group': group,
                        'row_in_group': row_in_group,
                        'physical_row': physical_starts[group] + row_in_group,
                        'upstream': None,
                        'upstream_metadata_status': 'unknown_requires_pinned_source_receipt'
                    }
                    if 'config' in item and isinstance(item['config'], (str, int, float, bool)):
                        ref['config'] = item['config']
                    if pin and pinned_shard:
                        ref['upstream'] = {
                            'revision': pin['revision'],
                            'config': pin.get('config'),
                            'path': pinned_shard['upstream_path'],
                            'lfs_sha256': pinned_shard['pinned_sha256']
                        }
                        ref['upstream_metadata_status'] = 'pinned_and_byte_bound'
                    refs.append(ref)
            if _file_sha256(byte_path) != file_sha:
                raise ValueError(f'{repo}:{local_path}: local file changed while reading sampled Parquet rows')
    return out, names, refs


def normalize_state(text):
    return ' '.join(str(text).split())


def legacy_identity_state(text, limit=1500):
    text = normalize_state(text)
    return text if len(text) <= limit else text[:limit] + ' ...'


# Each spec: kind, question, train/held-out splits, and a function (row, label names) -> (state, answer, options/levels)
# or None to skip the row. Choice answers are option names; noul answers are booleans or frequencies; score answers are
# level indices or fractional positions.
SPECS = {
    'banking77': ('legacy-datasets/banking77', 'choice', 'Which intent does this banking customer message express?',
                  ['train'], ['test'], lambda r, n: (r['text'], n['label'][int(r['label'])], n['label'])),
    'clinc-intent': ('clinc/clinc_oos', 'choice', 'Which intent does this assistant request express (oos means out of scope)?',
                     ['train'], ['test'], lambda r, n: (r['text'], n['intent'][int(r['intent'])], n['intent'])),
    'ag-news': ('fancyzhx/ag_news', 'choice', 'Which section does this news article belong to?',
                ['train'], ['test'], lambda r, n: (r['text'], n['label'][int(r['label'])], n['label'])),
    'dbpedia': ('fancyzhx/dbpedia_14', 'choice', 'What kind of entity does this encyclopedia entry describe?',
                ['train'], ['test'], lambda r, n: (f"{r['title']}: {r['content']}", n['label'][int(r['label'])], n['label'])),
    'emotion': ('dair-ai/emotion', 'choice', 'Which emotion does the writer express?',
                ['train'], ['test'], lambda r, n: (r['text'], n['label'][int(r['label'])], n['label'])),
    'trec-question': ('SetFit/TREC-QC', 'choice', 'What kind of answer does this question ask for?',
                      ['train'], ['test'], None),
    'newsgroups': ('SetFit/20_newsgroups', 'choice', 'Which newsgroup was this post written for?',
                   ['train'], ['test'], None),
    'massive-intent': ('mteb/amazon_massive_intent', 'choice', 'Which intent does this voice-assistant request express?',
                       ['train'], ['test'], None),
    'language-id': ('papluca/language-identification', 'choice', 'Which language is this text written in (ISO 639-1 code)?',
                    ['train'], ['test'], None),
    'yahoo-topic': ('community-datasets/yahoo_answers_topics', 'choice', 'Which topic does this question belong to?',
                    ['train'], ['test'], lambda r, n: (f"{r['question_title']} {r['question_content'] or ''}",
                                                       n['topic'][int(r['topic'])], n['topic'])),
    'pubmedqa': ('qiaojin/PubMedQA', 'choice', 'Given the abstract, is the answer to the research question yes, no or maybe?',
                 ['train'], ['train'], None),
    'boardgame-qa': ('tasksource/Boardgame-QA', 'choice',
                     'Given the facts, rules and preferences, is the goal proved, disproved, or unknown?',
                     ['train'], ['test'], lambda r, n: (r['example'], r['label'], ['proved', 'disproved', 'unknown'])),
    'vitaminc': ('tals/vitaminc', 'choice', 'Does the evidence support or refute the claim, or is there not enough information?',
                 ['train'], ['test'], lambda r, n: (f"Claim: {r['claim']}\nEvidence: {r['evidence']}", r['label'],
                                                    ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO'])),
    'boolq': ('google/boolq', 'noul', 'Is the answer to the question yes, according to the passage?',
              ['train'], ['validation'], lambda r, n: (f"Question: {r['question']}\nPassage: {r['passage']}",
                                                       str(r['answer']) == 'True', None)),
    'paws': ('google-research-datasets/paws', 'noul', 'Are the two sentences paraphrases of each other?',
             ['train'], ['test'], lambda r, n: (f"1: {r['sentence1']}\n2: {r['sentence2']}", int(r['label']) == 1, None)),
    'sms-spam': ('ucirvine/sms_spam', 'noul', 'Is this text message spam?',
                 ['train'], ['train'], lambda r, n: (r['sms'], int(r['label']) == 1, None)),
    'sarcasm': ('raquiba/Sarcasm_News_Headline', 'noul', 'Is this news headline sarcastic?',
                ['train'], ['test'], lambda r, n: (r['headline'], int(r['is_sarcastic']) == 1, None)),
    'subjectivity': ('SetFit/subj', 'noul', 'Is this sentence subjective (an opinion) rather than objective?',
                     ['train'], ['test'], lambda r, n: (r['text'], int(r['label']) == 1, None)),
    'ruletaker': ('tasksource/ruletaker', 'noul', 'Does the statement follow from the facts and rules?',
                  ['train'], ['test'], lambda r, n: (f"{r['context']}\nStatement: {r['question']}", r['label'] == 'entailment', None)),
    'toxicity': ('google/civil_comments', 'noul',
                 'Would a typical annotator consider this comment toxic? (Answer with the probability.)',
                 ['train'], ['test'], lambda r, n: (r['text'], float(r['toxicity']), None)),
    'sst5': ('SetFit/sst5', 'score', 'How positive is this movie review sentence?',
             ['train'], ['test'], lambda r, n: (r['text'], int(r['label']),
                                                ['very negative', 'negative', 'neutral', 'positive', 'very positive'])),
    'yelp-stars': ('Yelp/yelp_review_full', 'score', 'How many stars did the reviewer give?',
                   ['train'], ['test'], lambda r, n: (r['text'], int(r['label']), n['label'])),
    'app-stars': ('sealuzh/app_reviews', 'score', 'How many stars did the app reviewer give?',
                  ['train'], ['train'], lambda r, n: (r['review'], int(r['star']) - 1, ['1', '2', '3', '4', '5'])),
    'helpfulness': ('nvidia/HelpSteer2', 'score', 'How helpful is the response to the prompt?',
                    ['train'], ['validation'], lambda r, n: (f"Prompt: {r['prompt']}\nResponse: {r['response']}",
                                                             int(r['helpfulness']), ['0', '1', '2', '3', '4'])),
    'formality': ('osyvokon/pavlick-formality-scores', 'score', 'How formal is this sentence?',
                  ['train'], ['test'], lambda r, n: (r['sentence'], float(r['avg_score']) + 3,
                                                     ['very informal', 'informal', 'somewhat informal', 'neutral',
                                                      'somewhat formal', 'formal', 'very formal'])),
}


def special(name, repo, split_names, source_pins=None):
    """Rows for specs whose mapping needs the whole table (label texts, aggregation)."""
    data, names, refs = rows(repo, split_names, source_pins)
    out = []
    if name == 'trec-question':
        options = sorted({r['label_coarse_text'] for r in data})
        out = [(r['text'], r['label_coarse_text'], options, [ref]) for r, ref in zip(data, refs)]
    elif name == 'newsgroups':
        options = sorted({r['label_text'] for r in data})
        out = [(r['text'], r['label_text'], options, [ref]) for r, ref in zip(data, refs)]
    elif name == 'massive-intent':
        options = sorted({r['label_text'] for r in data})
        out = [(r['text'], r['label_text'], options, [ref]) for r, ref in zip(data, refs)]
    elif name == 'language-id':
        options = sorted({r['labels'] for r in data})
        out = [(r['text'], r['labels'], options, [ref]) for r, ref in zip(data, refs)]
    elif name == 'pubmedqa':
        for r, ref in zip(data, refs):
            context = r['context']
            if isinstance(context, str):
                try:
                    context = ast.literal_eval(context)
                except (ValueError, SyntaxError):
                    context = {'contexts': [context]}
            out.append((f"Question: {r['question']}\nAbstract: {' '.join(context.get('contexts', []))}",
                        r['final_decision'], ['yes', 'no', 'maybe'], [ref]))
    return out


def hate_speech(split_names, source_pins=None):
    """Mean annotator `hatespeech` rating (0 not hateful, 1 unclear, 2 hateful) per comment: a fractional target."""
    data, _, refs = rows('ucberkeley-dlab/measuring-hate-speech', split_names, source_pins)
    by = defaultdict(list)
    text = {}
    source_refs = defaultdict(list)
    for r, ref in zip(data, refs):
        by[r['comment_id']].append(float(r['hatespeech']))
        text[r['comment_id']] = r['text']
        source_refs[r['comment_id']].append(ref)
    return [(text[c], sum(v) / len(v), ['not hateful', 'unclear', 'hateful'], source_refs[c])
            for c, v in by.items() if len(v) >= 2]


def build_cases(name, spec, per_train, per_heldout, licenses, source_pins=None, source_catalog_provenance=None,
                identity_collisions=None):
    repo, kind, question, train_splits, held_splits, mapper = spec
    out = []
    shared = train_splits == held_splits  # one split only: hold out a hash-selected tail
    seen = set()  # across roles: a text in the training selection never reappears held out
    for role, split_names, limit in (('train', train_splits, per_train), ('heldout', held_splits, per_heldout)):
        if name == 'hate-speech':
            mapped = hate_speech(split_names, source_pins)
        elif mapper is None:
            mapped = special(name, repo, split_names, source_pins)
        else:
            data, names, refs = rows(repo, split_names, source_pins)
            mapped = [(m[0], m[1], m[2], [ref]) for r, ref in zip(data, refs)
                      if (m := mapper(r, names)) is not None]
        chosen = []
        identity_states = defaultdict(lambda: defaultdict(list))
        for state, answer, options, source_refs in sorted(mapped, key=lambda m: digest(f'{name}:{m[0]}')):
            # Preserve the complete mapped evidence for every newly built case.
            # Keep the historical normalized/clipped value only as the stable
            # identity and split key, so added suffix evidence does not create a
            # new source world or move an existing row between splits.
            full_state = state if name == 'pubmedqa' else normalize_state(state)
            identity_state = legacy_identity_state(state)
            key = digest(identity_state)
            identity_states[key][digest(full_state)].extend(source_refs)
            # Continue scanning after the selection limit so the collision
            # census covers every already-loaded mapped row. Do not mark these
            # unselected rows seen; the historical selection stops here.
            if len(chosen) >= limit or not identity_state or key in seen:
                continue
            if shared and (int(key[:8], 16) % 8 == 0) != (role == 'heldout'):
                continue
            if isinstance(answer, float) and math.isnan(answer):
                continue
            seen.add(key)
            chosen.append((full_state, identity_state, answer, options, source_refs))
        if identity_collisions is not None:
            for identity_sha, full_states in identity_states.items():
                if len(full_states) > 1:
                    identity_collisions.append({
                        'family': f'decision:{name}', 'role': role,
                        'identity_state_sha256': identity_sha,
                        'distinct_full_state_sha256': sorted(full_states),
                        'full_states': [
                            {'full_state_sha256': full_sha, 'source_refs': refs}
                            for full_sha, refs in sorted(full_states.items())
                        ]
                    })
        for state, identity_state, answer, options, source_refs in chosen:
            case = {'version': 'natlang.decision-case/1', 'id': 'dc-' + digest(f'{name}:{identity_state}')[:20], 'source': repo,
                    'family': f'decision:{name}', 'role': role, 'kind': kind, 'question': question, 'state': state,
                    'group': 'g-' + digest(f'decision:{identity_state}')[:24], 'license': licenses.get(repo),
                    'identity_state_sha256': digest(identity_state), 'full_state_sha256': digest(state),
                    'source_refs': source_refs}
            if source_catalog_provenance:
                case['license_provenance'] = {
                    'legacy_catalog_value': licenses.get(repo),
                    'catalog_path': source_catalog_provenance['path'],
                    'catalog_sha256': source_catalog_provenance['sha256'],
                    'status': 'local_catalog_value_not_bound_to_upstream_revision',
                    'upstream_license_status': 'unknown'
                }
            if kind == 'choice':
                case.update(options=list(options), answer=answer)
                derived_criteria, contract_provenance = criteria_for(
                    family=case['family'], source=repo, kind=kind, labels=list(options),
                    where=f"{case['family']}:{case['id']}")
                if contract_provenance is not None:
                    case['criteria'] = derived_criteria
            elif kind == 'noul':
                case['answer'] = answer if isinstance(answer, bool) else round(float(answer), 6)
            else:
                case.update(levels=list(options), answer=answer if isinstance(answer, int) else round(float(answer), 6))
            out.append(case)
    return out


TARGETS = {
    'choice': ({'question': 'string', 'text': 'string', 'options': 'string[]'}, 'Record<string, number>',
               'Answer the question about the text with a probability for every option, summing to 1. The answer is '
               'scored by the Brier score, so put weight where the evidence is and spread it where it is uncertain.'),
    'noul': ({'question': 'string', 'text': 'string'}, 'number',
             'Return the probability (0 to 1) that the answer to the yes/no question about the text is yes. The answer '
             'is scored by the Brier score against the label or the share of annotators who said yes.'),
    'score': ({'question': 'string', 'text': 'string', 'levels': 'string[]'}, '{ probabilities: Record<string, number> }',
              'Rate the text on the ordered levels (lowest first): return a probability for every level, summing to 1. '
              'The answer is scored by the ranked probability score, so nearby levels count as nearly right.'),
}
METRICS = {'choice': 'choice-brier', 'noul': 'binary-brier', 'score': 'ordinal-rps'}


def target(kind):
    args, returns, prompt = TARGETS[kind]
    return {'kind': 'improvement-case', 'entry': 'solve.nl', 'exportName': 'default',
            'source': {'schema': 'natlang.skill-decision-target/1', 'id': f'decision-{kind}-v1'},
            'files': {'solve.nl': '---\nargs:\n' + ''.join(f'  {k}: {v}\n' for k, v in args.items())
                      + f'returns: {returns}\n---\n{prompt}\n'}}


def episode_case(case):
    if case['kind'] == 'choice':
        args, expected = [case['question'], case['state'], case['options']], {'kind': 'choice', 'answer': case['answer'], 'options': case['options']}
    elif case['kind'] == 'noul':
        args, expected = [case['question'], case['state']], {'kind': 'binary', 'answer': case['answer']}
    else:
        args, expected = [case['question'], case['state'], case['levels']], {'kind': 'ordinal', 'levels': case['levels'], 'answer': case['answer']}
    return {'id': 'case-' + case['id'][3:], 'group': case['group'], 'args': args, 'expected': expected}


def build_episodes(cases, per_family):
    by = defaultdict(lambda: {'train': [], 'heldout': []})
    for case in cases:
        by[case['family']][case['role']].append(case)
    episodes = []
    for kind in ('choice', 'noul', 'score'):
        families = sorted(f for f, v in by.items() if v['train'] and v['train'][0]['kind'] == kind)
        for index, family in enumerate(families):
            other = families[(index + 1) % len(families)]
            for role, split_of in (('train', lambda n: 'train'), ('heldout', lambda n: 'validation' if n % 2 == 0 else 'test')):
                own, theirs = by[family][role], by[other][role]
                count = per_family if role == 'train' else max(2, per_family // 5)
                for n in range(count):
                    chosen = own[8 * n: 8 * n + 8]
                    transfer = theirs[len(theirs) - 4 * (n + 1): len(theirs) - 4 * n]
                    if len(chosen) < 8 or len(transfer) < 4 or 8 * count + 4 * count > len(own) + len(theirs) * 2:
                        break
                    support, query, moved = ([episode_case(c) for c in chosen[:4]], [episode_case(c) for c in chosen[4:]],
                                             [episode_case(c) for c in transfer])
                    groups = sorted(c['group'] for c in support + query + moved)
                    metric = {'schema': 'natlang.skill-graded/1', 'kind': METRICS[kind]}
                    slug = family.split(':', 1)[1]
                    episodes.append({'version': 'natlang.skill-episode/1', 'id': f'skill-decision-{slug}-{role}-{n}',
                                     'family': family, 'split': split_of(n),
                                     'source_groups': [group_commitment(groups)],
                                     'license': chosen[0]['license'] or 'see source dataset',
                                     'target': target(kind), 'library': {'kind': 'empty', 'skills': {}},
                                     'support': {'cases': support}, 'query': {'cases': query},
                                     'transfer': {'family': other, 'target': target(kind), 'cases': moved},
                                     'operations': ['create', 'revise', 'select', 'test'], 'limits': {'maxSteps': 6},
                                     'provenance': {'generator': 'natlang.skill-decision-episodes/1', 'source': chosen[0]['source'],
                                                    'decision_kind': kind, 'transfer_family': other,
                                                    'metric': metric, 'transfer_metric': metric}})
    return episodes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', required=True)
    parser.add_argument('--per-train', type=int, default=2000)
    parser.add_argument('--per-heldout', type=int, default=300)
    parser.add_argument('--episodes-per-family', type=int, default=20)
    parser.add_argument('--source-pins', help=(
        'optional natlang.decision-source-pins/1 JSON mapping dataset repos to explicit upstream revisions and '
        'local shard paths/upstream paths/SHA-256 or LFS OIDs. Every pinned local shard must match its declared bytes; '
        'unprovided upstream provenance remains explicitly unknown.'))
    args = parser.parse_args()
    SOURCE_FILES_READ.clear()
    source_catalog_path = os.path.realpath(f'{ROOT}/MANIFEST.json')
    with open(source_catalog_path, 'rb') as stream:
        source_catalog_bytes = stream.read()
    source_catalog = json.loads(source_catalog_bytes)
    builder_path = os.path.realpath(__file__)
    builder_sha256 = _file_sha256(builder_path)
    source_catalog_provenance = {
        'path': os.path.relpath(source_catalog_path, os.path.realpath(ROOT)).replace(os.sep, '/'),
        'sha256': hashlib.sha256(source_catalog_bytes).hexdigest()
    }
    licenses = {repo: (info.get('license') if isinstance(info.get('license'), str) else json.dumps(info.get('license')))
                for repo, info in source_catalog.items() if 'error' not in info}
    source_pins, source_pin_receipt = load_source_pins(args.source_pins)
    specs = dict(SPECS)
    specs['hate-speech'] = ('ucberkeley-dlab/measuring-hate-speech', 'score', 'How hateful is this comment?',
                            ['train'], ['train'], None)
    cases, counts, identity_collisions = [], {}, []
    for name, spec in specs.items():
        built = build_cases(name, spec, args.per_train, args.per_heldout, licenses, source_pins,
                            source_catalog_provenance, identity_collisions)
        counts[name] = {role: sum(1 for c in built if c['role'] == role) for role in ('train', 'heldout')}
        cases += built
        print(name, counts[name], flush=True)
    ids = [c['id'] for c in cases]
    assert len(ids) == len(set(ids)), 'case ids repeat'
    episodes = build_episodes(cases, args.episodes_per_family)
    source_files = {}
    unknown_repos = set()
    for key, entry in SOURCE_FILES_READ.items():
        source_files[key] = {
            **entry,
            'splits_read': sorted(entry['splits_read']),
            'sampled_row_groups': sorted(entry['sampled_row_groups'])
        }
        if entry['upstream_metadata_status'] != 'pinned_and_byte_bound':
            unknown_repos.add(entry['dataset'])
    os.makedirs(args.out, exist_ok=True)
    bodies = {}
    for name, items in (('decision-cases.jsonl', cases), ('decision-episodes.jsonl', episodes)):
        bodies[name] = ''.join(json.dumps(item, sort_keys=True) + '\n' for item in items)
        with open(os.path.join(args.out, name), 'x') as stream:
            stream.write(bodies[name])
    source_provenance = {
        'schema': 'natlang.decision-source-provenance/1',
        'builder_path': builder_path,
        'builder_sha256': builder_sha256,
        'source_catalog_path': os.path.relpath(source_catalog_path, os.path.realpath(ROOT)).replace(os.sep, '/'),
        'source_catalog_sha256': hashlib.sha256(source_catalog_bytes).hexdigest(),
        'source_catalog_license_status': 'local_catalog_values_recorded_but_not_bound_to_upstream_revision',
        'decision_task_contracts': {
            'path': os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                '..', 'training', 'decision_task_contracts.json')),
            'sha256': _file_sha256(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                '..', 'training', 'decision_task_contracts.json')),
            'loader_path': os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                'decision_task_contracts.py')),
            'loader_sha256': _file_sha256(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                'decision_task_contracts.py')),
            'schema': 'natlang.decision-task-contracts/1',
            'applied_families': ['decision:trec-question'],
            'meaning': 'Derived criteria are an explicit model-visible taxonomy gloss; upstream labels, golds, IDs, groups, and source row refs remain unchanged.'
        },
        'source_pin_receipt': source_pin_receipt,
        'source_files_read': [source_files[k] for k in sorted(source_files)],
        'upstream_metadata_unknown_datasets': sorted(unknown_repos),
        'conversion_obligation': (
            'For every dataset with unknown upstream metadata, recover an authoritative immutable dataset revision, '
            'upstream shard path and upstream shard SHA-256/LFS OID; match local bytes and publish a new source-bound '
            'derivative. Do not infer a commit from repository name, cache path, or current remote state.'
            if unknown_repos else None
        )
    }
    summary = {'schema': 'natlang.decision-data/1', 'cases': len(cases), 'episodes': len(episodes), 'per_dataset': counts,
               'episode_splits': {s: sum(1 for e in episodes if e['split'] == s) for s in ('train', 'validation', 'test')},
               'kinds': {k: sum(1 for c in cases if c['kind'] == k) for k in ('choice', 'noul', 'score')},
               'sha256': {name: digest(body) for name, body in bodies.items()}, 'model_calls': 0,
               'source_provenance': source_provenance,
               'identity_collision_census_scope': 'all_loaded_mapped_rows_per_family_and_role',
               'identity_collisions': identity_collisions,
               'licenses': licenses}
    with open(os.path.join(args.out, 'decision-data.manifest.json'), 'x') as stream:
        stream.write(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: summary[k] for k in ('cases', 'episodes', 'episode_splits', 'kinds')}, indent=2))
    gate = run_gate([os.path.join(args.out, 'decision-episodes.jsonl')],
                    report=os.path.join(args.out, 'decision-episodes.audit.json'))
    print(json.dumps({'gate': 'passed', 'warnings': gate['warning_count']}))


if __name__ == '__main__':
    main()
