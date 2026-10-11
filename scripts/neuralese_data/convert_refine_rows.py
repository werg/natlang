#!/usr/bin/env python3
"""Refinement corpora → native teacher-turn rows for render_native_turn_corpus.py.

- ``natlang.decision-prompt/1`` (refine-judge-exact): the judge's system+user prompt as recorded (it mirrors
  ts-host decisionJudge), target = the gold option as the assistant reply. Train rows only (held-out predicate and
  family rows stay out); source group = the predicate id, so a predicate never spans splits.
- ``natlang.refine-author/1`` (refine-author-static): the user request and the assistant's rewrite (2 messages);
  the pair's id is its source group; all train.

Rows are marked approved under the registry's training admission (inventory section H, 2026-10-11); the system
prompt is the program's own prompt, so render with ``--program-header-as-full``.
Usage: convert_refine_rows.py --judge FILE --author FILE --out FILE
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def turn(rid, messages, target, split, groups, source, admission):
    return {'version': 'natlang.teacher_training_turn.native/1', 'id': rid, 'split': split,
            'source_groups': groups, 'source': source, 'messages': messages, 'target': target, 'tools': [],
            'training_admission': {'approved': True, 'kind': admission}, 'source_ref': {'trajectory_id': rid}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--judge', type=Path, required=True)
    p.add_argument('--author', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    n = {'judge_train': 0, 'judge_heldout_left_out': 0, 'author': 0}
    with open(a.out, 'w') as out:
        for line in a.judge.read_text().splitlines():
            r = json.loads(line)
            if r['split'] != 'train':
                n['judge_heldout_left_out'] += 1
                continue
            answer = r['options'][max(range(len(r['gold'])), key=lambda i: r['gold'][i])]
            out.write(json.dumps(turn(r['id'], r['messages'], {'role': 'assistant', 'content': answer}, 'train',
                                      ['refine-judge-predicate:' + r['predicate_id']], 'refine-judge-exact',
                                      'crisp-checker-label (refine-judge-exact-20261009-v1)'), ensure_ascii=False) + '\n')
            n['judge_train'] += 1
        for line in a.author.read_text().splitlines():
            r = json.loads(line)
            messages, target = r['messages'][:-1], r['messages'][-1]
            if target['role'] != 'assistant':
                raise SystemExit(f"{r['id']}: last message is not the assistant rewrite")
            out.write(json.dumps(turn(r['id'], messages, {'role': 'assistant', 'content': target['content']}, 'train',
                                      [r['id']], 'refine-author-static',
                                      'static authored refinement (refine-author-static-20261009-v1)'),
                                 ensure_ascii=False) + '\n')
            n['author'] += 1
    print(json.dumps(n))


if __name__ == '__main__':
    main()
