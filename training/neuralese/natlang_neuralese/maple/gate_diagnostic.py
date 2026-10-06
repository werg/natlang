"""Where does a foundation projection disagree with the teacher? Held positions bucketed by the teacher's top-1
probability, top-k containment, bf16 near-ties, and the untrained (copied readout) baseline.

    PYTHONPATH=training/neuralese python -m natlang_neuralese.maple.gate_diagnostic RUN_DIR HEADS RECORDS PIECES CONTEXTS CUTOFF OUT
"""
import json
import sys
from pathlib import Path

import torch

from natlang_neuralese.model.causal_feedback import CausalFeedbackProjection, load_projection_state
from natlang_neuralese.serve import load_engine
from natlang_neuralese.train.bootstrap_data import context_ids
from natlang_neuralese.train.output_embedding_projection import source_texts

run, heads, records, pieces, contexts_n, cutoff, out = sys.argv[1:8]
engine = load_engine(heads_checkpoint=heads, device='cuda')
backbone = engine.backbone
backbone.ffn_chunk_tokens = 256
projection = CausalFeedbackProjection(backbone).to('cuda').eval()
baseline = CausalFeedbackProjection(backbone).to('cuda').eval()
state = torch.load(Path(run) / 'best-checkpoint.pt', map_location='cpu', weights_only=False)
load_projection_state(projection, state['projection'])
texts = source_texts(records)
held = [t for t in texts['test'] if t not in set(texts['train'])]
contexts, _ = context_ids(engine, records, pieces, int(contexts_n), 1024)
examples = {'source': [engine._tokens(t)[:1024] for t in held], 'context': [list(i) for i in contexts['test']]}
buckets = [0.0, 0.3, 0.5, 0.7, 0.9, 1.01]
report = {}
with torch.no_grad():
    for name, sequences in examples.items():
        stats = {'n': 0, 'agree': 0, 'top5': 0, 'base_agree': 0, 'ties': 0, 'tie_disagree': 0,
                 'buckets': {f'{lo}-{hi}': [0, 0] for lo, hi in zip(buckets, buckets[1:])}}
        for ids in sequences:
            if engine.tokenizer.bos_token_id is not None:
                ids = [engine.tokenizer.bos_token_id] + ids
            o = backbone.forward_ids(torch.tensor([ids], device='cuda'), cutoff=int(cutoff), logits=False)
            for b in range(0, o['h_cut'].shape[1], 128):
                h, f = o['h_cut'][0, b:b + 128], o['h_final'][0, b:b + 128]
                teacher, student, base = backbone.logits(f), projection.logits(h), baseline.logits(h)
                t1, s1 = teacher.argmax(-1), student.argmax(-1)
                p = teacher.float().softmax(-1)
                ptop = p.max(-1).values
                top2 = teacher.topk(2, -1).values
                tie = top2[:, 0] == top2[:, 1]
                agree = s1 == t1
                stats['n'] += len(t1); stats['agree'] += int(agree.sum())
                stats['base_agree'] += int((base.argmax(-1) == t1).sum())
                stats['top5'] += int((teacher.topk(5, -1).indices == s1[:, None]).any(-1).sum())
                stats['ties'] += int(tie.sum()); stats['tie_disagree'] += int((tie & ~agree).sum())
                for lo, hi in zip(buckets, buckets[1:]):
                    m = (ptop >= lo) & (ptop < hi)
                    stats['buckets'][f'{lo}-{hi}'][0] += int(m.sum()); stats['buckets'][f'{lo}-{hi}'][1] += int((m & agree).sum())
        n = stats['n']
        report[name] = {'positions': n, 'agreement': stats['agree'] / n, 'untrained_agreement': stats['base_agree'] / n,
                        'student_in_teacher_top5': stats['top5'] / n, 'teacher_exact_bf16_ties': stats['ties'] / n,
                        'disagreements_at_exact_ties': stats['tie_disagree'] / max(1, n - stats['agree']),
                        'by_teacher_top1_probability': {k: {'share': v[0] / n, 'agreement': v[1] / max(1, v[0])} for k, v in stats['buckets'].items()},
                        'teacher_logit_dtype': str(teacher.dtype)}
Path(out).write_text(json.dumps(report, indent=1) + '\n')
print(json.dumps(report, indent=1))
