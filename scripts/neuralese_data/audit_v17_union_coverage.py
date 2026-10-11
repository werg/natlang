#!/usr/bin/env python3
"""Exact coverage audit: every V17 train decision re-rendered (LFM2.5, gold_text_rows) vs its 4962 union document (Pop pop-9e9b)."""
import json, random, sys
sys.path[:0] = ['training/neuralese', 'scripts', 'scripts/neuralese_data']
from build_neuralese_gold_text_delta import load_pinned_tokenizer
from natlang_neuralese.serve.chat import bind_history_reasoning
from natlang_neuralese.data.text_corpus import gold_text_rows
tok = load_pinned_tokenizer('models/hf/hub/models--LiquidAI--LFM2.5-350M/snapshots/9e6c6ccf47cd318696e137d381a7ded8fe4df09f')
bind_history_reasoning(tok, require_declared=True)
L = 'runs/luna-v17-actual-quality-proposal-20261010-v9-portable-context-view-v1/assembled-v3/'
pieces = [json.loads(l) for l in open(L + 'native-pieces.jsonl')]
recs = []
for l in open(L + 'native-records.jsonl', 'rb'):
    r = json.loads(l)
    if r['split'] == 'train':
        import hashlib; r['_source_record_sha256'] = hashlib.sha256(l.rstrip(b'\n')).hexdigest(); recs.append(r)
union = {}
for l in open('runs/neuralese-foundation-text-input-4962-root-adopted-union-20261009-v1/text-data.jsonl'):
    r = json.loads(l)
    for m in r.get('source_record_ids') or [r['id']]:
        union[m] = r
# render all train records (whole set: writer hydration needs the full record list)
rows, receipt, omitted, prov = gold_text_rows(recs, pieces, tokenizer=tok, require_independent_splits=False)
res = {'train_records': len(recs), 'rendered_docs': len(rows), 'omitted': len(omitted), 'exact_doc_equal': 0,
       'exact_prompt_and_action': 0, 'differs': 0, 'examples_differ': []}
for row in rows:
    for rid in row['source_record_ids']:
        u = union.get(rid)
        if u is None:
            res['differs'] += 1; continue
        if (u['text'], u['token_ids'], u['supervised_suffix_start']) == (row['text'], row['token_ids'], row['supervised_suffix_start']):
            res['exact_doc_equal'] += 1
        elif (u['token_ids'][:u['supervised_suffix_start']] == row['token_ids'][:row['supervised_suffix_start']] and
              u['token_ids'][u['supervised_suffix_start']:] == row['token_ids'][row['supervised_suffix_start']:]):
            res['exact_prompt_and_action'] += 1
        else:
            res['differs'] += 1
            if len(res['examples_differ']) < 5: res['examples_differ'].append(rid)
res['omission_reasons'] = sorted({o['reason'] for o in omitted})
print(json.dumps(res))
