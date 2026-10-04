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
from collections import defaultdict

import pyarrow.parquet as pq

ROOT = '/mnt/external/natlang-development-data/data/decision-sources'


def digest(value):
    return hashlib.sha256((value if isinstance(value, str) else json.dumps(value, sort_keys=True)).encode()).hexdigest()


MAX_ROWS = 60000


def rows(repo, splits):
    """Rows of the named splits (first shard each), with their label-name metadata.

    At most MAX_ROWS rows per split, taken from row groups spread evenly over the file: some sources are sorted by
    label, and reading whole large shards into Python objects would cost gigabytes for no gain.
    """
    out, names = [], {}
    for split in splits:
        for path in sorted(glob.glob(f'{ROOT}/{repo.replace("/", "__")}/**/{split}/*.parquet', recursive=True)):
            parquet = pq.ParquetFile(path)
            groups = parquet.metadata.num_row_groups
            per = max(1, parquet.metadata.num_rows // groups)
            wanted = min(groups, max(1, MAX_ROWS // per))
            picks = sorted({round(i * (groups - 1) / max(1, wanted - 1)) for i in range(wanted)}) if wanted > 1 else [0]
            table = parquet.read_row_groups(picks)
            meta = parquet.schema_arrow.metadata or {}
            if b'huggingface' in meta:
                for key, feature in json.loads(meta[b'huggingface']).get('info', {}).get('features', {}).items():
                    if isinstance(feature, dict) and feature.get('names'):
                        names[key] = feature['names']
            out += table.to_pylist()
    return out, names


def clip(text, limit=1500):
    text = ' '.join(str(text).split())
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


def special(name, repo, split_names):
    """Rows for specs whose mapping needs the whole table (label texts, aggregation)."""
    data, names = rows(repo, split_names)
    out = []
    if name == 'trec-question':
        options = sorted({r['label_coarse_text'] for r in data})
        out = [(r['text'], r['label_coarse_text'], options) for r in data]
    elif name == 'newsgroups':
        options = sorted({r['label_text'] for r in data})
        out = [(r['text'], r['label_text'], options) for r in data]
    elif name == 'massive-intent':
        options = sorted({r['label_text'] for r in data})
        out = [(r['text'], r['label_text'], options) for r in data]
    elif name == 'language-id':
        options = sorted({r['labels'] for r in data})
        out = [(r['text'], r['labels'], options) for r in data]
    elif name == 'pubmedqa':
        for r in data:
            context = r['context']
            if isinstance(context, str):
                try:
                    context = ast.literal_eval(context)
                except (ValueError, SyntaxError):
                    context = {'contexts': [context]}
            out.append((f"Question: {r['question']}\nAbstract: {' '.join(context.get('contexts', []))}",
                        r['final_decision'], ['yes', 'no', 'maybe']))
    return out


def hate_speech(split_names):
    """Mean annotator `hatespeech` rating (0 not hateful, 1 unclear, 2 hateful) per comment: a fractional target."""
    data, _ = rows('ucberkeley-dlab/measuring-hate-speech', split_names)
    by = defaultdict(list)
    text = {}
    for r in data:
        by[r['comment_id']].append(float(r['hatespeech']))
        text[r['comment_id']] = r['text']
    return [(text[c], sum(v) / len(v), ['not hateful', 'unclear', 'hateful']) for c, v in by.items() if len(v) >= 2]


def build_cases(name, spec, per_train, per_heldout, licenses):
    repo, kind, question, train_splits, held_splits, mapper = spec
    out = []
    shared = train_splits == held_splits  # one split only: hold out a hash-selected tail
    seen = set()  # across roles: a text in the training selection never reappears held out
    for role, split_names, limit in (('train', train_splits, per_train), ('heldout', held_splits, per_heldout)):
        if name == 'hate-speech':
            mapped = hate_speech(split_names)
        elif mapper is None:
            mapped = special(name, repo, split_names)
        else:
            data, names = rows(repo, split_names)
            mapped = [m for m in (mapper(r, names) for r in data) if m is not None]
        chosen = []
        for state, answer, options in sorted(mapped, key=lambda m: digest(f'{name}:{m[0]}')):
            state = clip(state)
            key = digest(state)
            if not state or key in seen:
                continue
            if shared and (int(key[:8], 16) % 8 == 0) != (role == 'heldout'):
                continue
            if isinstance(answer, float) and math.isnan(answer):
                continue
            seen.add(key)
            chosen.append((state, answer, options))
            if len(chosen) >= limit:
                break
        for state, answer, options in chosen:
            case = {'version': 'natlang.decision-case/1', 'id': 'dc-' + digest(f'{name}:{state}')[:20], 'source': repo,
                    'family': f'decision:{name}', 'role': role, 'kind': kind, 'question': question, 'state': state,
                    'group': 'g-' + digest(f'decision:{state}')[:24], 'license': licenses.get(repo)}
            if kind == 'choice':
                case.update(options=list(options), answer=answer)
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
                                     'source_groups': ['group-commitment:sha256:' + digest(groups)],
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
    args = parser.parse_args()
    manifest = json.load(open(f'{ROOT}/MANIFEST.json'))
    licenses = {repo: (info.get('license') if isinstance(info.get('license'), str) else json.dumps(info.get('license')))
                for repo, info in manifest.items() if 'error' not in info}
    specs = dict(SPECS)
    specs['hate-speech'] = ('ucberkeley-dlab/measuring-hate-speech', 'score', 'How hateful is this comment?',
                            ['train'], ['train'], None)
    cases, counts = [], {}
    for name, spec in specs.items():
        built = build_cases(name, spec, args.per_train, args.per_heldout, licenses)
        counts[name] = {role: sum(1 for c in built if c['role'] == role) for role in ('train', 'heldout')}
        cases += built
        print(name, counts[name], flush=True)
    ids = [c['id'] for c in cases]
    assert len(ids) == len(set(ids)), 'case ids repeat'
    episodes = build_episodes(cases, args.episodes_per_family)
    os.makedirs(args.out, exist_ok=True)
    bodies = {}
    for name, items in (('decision-cases.jsonl', cases), ('decision-episodes.jsonl', episodes)):
        bodies[name] = ''.join(json.dumps(item, sort_keys=True) + '\n' for item in items)
        with open(os.path.join(args.out, name), 'x') as stream:
            stream.write(bodies[name])
    summary = {'schema': 'natlang.decision-data/1', 'cases': len(cases), 'episodes': len(episodes), 'per_dataset': counts,
               'episode_splits': {s: sum(1 for e in episodes if e['split'] == s) for s in ('train', 'validation', 'test')},
               'kinds': {k: sum(1 for c in cases if c['kind'] == k) for k in ('choice', 'noul', 'score')},
               'sha256': {name: digest(body) for name, body in bodies.items()}, 'model_calls': 0,
               'licenses': licenses}
    with open(os.path.join(args.out, 'decision-data.manifest.json'), 'x') as stream:
        stream.write(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: summary[k] for k in ('cases', 'episodes', 'episode_splits', 'kinds')}, indent=2))


if __name__ == '__main__':
    main()
