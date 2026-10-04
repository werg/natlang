#!/usr/bin/env python3
"""Label decision cases with a decision model's probabilities: soft targets for distillation and a teacher baseline.

Reads `decision-cases.jsonl` (scripts/build_decision_cases.py) and appends one row per case to the output:
the teacher's full distribution (choice: per option; noul: P(yes); score: per level plus the fractional score).
Resumable: cases already in the output are skipped. Backends: `decider` (Strands Decider checkpoints, through the
strands_decider package) and `clef` (Cloudflare Clef releases, through the `systemone` function shipped with the
weights). Both answer Jev/SystemOne-shaped questions. The CUDA allocation is capped because memory is shared with
the rest of the machine.
"""
import argparse
import hashlib
import json
import os
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--teacher', required=True, help='name recorded with every label, e.g. strands-decider-2B-hobson-v19')
    parser.add_argument('--backend', choices=['decider', 'clef'], default='decider')
    parser.add_argument('--memory-gb', type=float, default=8)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()

    import torch
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.memory_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory))
    from strands_decider.infer import load_engine
    from strands_decider.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as stream:
            done = {json.loads(line)['id'] for line in stream if line.strip()}
    manifest_path = args.out + '.manifest.json'
    checkpoint_files = sorted(os.path.join(root, f) for root, _, files in os.walk(args.checkpoint) for f in files
                              if f.endswith(('.safetensors', '.json')))
    identity = {'schema': 'natlang.decision-labels/1', 'teacher': args.teacher, 'checkpoint': os.path.abspath(args.checkpoint),
                'checkpoint_sha256': hashlib.sha256(b''.join(hashlib.sha256(open(f, 'rb').read()).digest()
                                                              for f in checkpoint_files)).hexdigest(),
                'cases_sha256': hashlib.sha256(open(args.cases, 'rb').read()).hexdigest()}
    if os.path.exists(manifest_path):
        if json.load(open(manifest_path)) != identity:
            raise SystemExit('output was labelled by another teacher or from other cases; use a new output')
    else:
        json.dump(identity, open(manifest_path, 'w'), indent=2)

    if args.backend == 'decider':
        engine = load_engine(args.checkpoint, device='cuda')

        def ask(case, question):
            return engine.ask(case['state'], {'q': question}).answers['q'].model_dump()
    else:
        import sys
        sys.path.insert(0, os.path.abspath(args.checkpoint))
        from joint_schema_model import load_release_model, systemone
        model, processor = load_release_model(args.checkpoint, device='cuda')

        def ask(case, question):
            body = question.model_dump(exclude_none=True)
            return systemone(model, processor, {'model': args.teacher, 'state': case['state'], 'questions': {'q': body}})['answers']['q']
    started, count = time.time(), 0
    with open(args.cases) as cases, open(args.out, 'a') as out:
        for line in cases:
            case = json.loads(line)
            if case['id'] in done:
                continue
            if case['kind'] == 'choice':
                question = ChoiceQuestion(instructions=case['question'], criteria={o: '' for o in case['options']})
            elif case['kind'] == 'noul':
                question = NoulQuestion(instructions=case['question'])
            else:
                question = ScoreQuestion(instructions=case['question'], criteria=case['levels'])
            try:
                answer = ask(case, question)
            except Exception as error:  # a case the teacher cannot take is recorded, not skipped silently
                answer = {'error': repr(error)[:300]}
            out.write(json.dumps({'id': case['id'], 'family': case['family'], 'teacher': args.teacher, 'answer': answer}) + '\n')
            count += 1
            if count % 500 == 0:
                out.flush()
                print(json.dumps({'labelled': count, 'seconds': round(time.time() - started), 'peak_gb':
                                  round(torch.cuda.max_memory_allocated() / 2**30, 2)}), flush=True)
            if args.limit and count >= args.limit:
                break
    print(json.dumps({'labelled': count, 'seconds': round(time.time() - started)}), flush=True)


if __name__ == '__main__':
    main()
