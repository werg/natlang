#!/usr/bin/env python3
"""One command for the LFM3 joint-SFT learner inputs from what is converted and admitted now.

Steps (CPU, DGX; the learners get pre-tokenized, pre-packed inputs and do no tokenization):
  1. ``score-teachers``: idiom scores of every verified LFM3 teacher twin's source trajectories (once per source set);
  2. ``build``: the mixture config from (a) graphed LFM3 renders of runs/lfm3-renders/queue.json whose sources all have
     idiom scores, (b) passed teacher twins under runs/lfm3-teacher-graphs, (c) communication graphs listed in
     runs/lfm3-communication/outputs.json when present; then build_idiom_mixture.py under the declared tier policy,
     the own-outputs/1 CE policy (build_native_ce_supervision_policy.py) and the exact packed-capacity geometry
     sidecar (build_native_packed_geometry.py) of the train manifest.
Outputs in OUT: config.json, the mixture (cohorts/, held/, train-sources.json, held-sources.json, mixture.json),
ce-policy.json, geometry/ and learner-inputs.json (the exact bindings a native_state_sft launch consumes).

  build_lfm3_joint_sft.py score-teachers
  build_lfm3_joint_sft.py build --out runs/lfm3-mixture/joint-sft-v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'training' / 'neuralese'))
sys.path.insert(0, str(REPO / 'scripts' / 'neuralese_data'))
from natlang_neuralese.common import line_backbones  # noqa: E402
from natlang_neuralese.common.hashing import sha256_file_hex  # noqa: E402

PY = str(REPO / '.venv-neuralese/bin/python')
POLICY = 'training/neuralese/mixtures/lfm3-joint-sft-mixture-policy-v3.json'
SCORES = REPO / 'runs/lfm3-mixture/scores'
QUEUE = REPO / 'runs/lfm3-renders/queue.json'
TEACHERS = REPO / 'runs/lfm3-teacher-graphs'
COMMUNICATION = REPO / 'runs/lfm3-communication/outputs.json'
HELD_PACKETS = ['runs/natlang-dev-task-packet-v1-20261028/cases.jsonl',
                'runs/natlang-held-task-packet-v3-20261012/cases.jsonl']


def run(cmd):
    print(json.dumps({'run': cmd[:6]}), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO)


def score_file(sources: list[str], label: str) -> Path:
    """The idiom score file of a source set (runs/lfm3-mixture/score-all.py naming)."""
    digest = hashlib.sha1(json.dumps(sources).encode()).hexdigest()[:10]
    return SCORES / f'{label}-{digest}.jsonl.zst'


def teacher_twins():
    for receipt_path in sorted(TEACHERS.glob('*/lfm3*/receipt.json')):  # lfm3 (full) and lfm3-program-header
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get('verification') or {}).get('passed'):
            yield receipt_path.parent, receipt


def teacher_score_file(source: str) -> Path:
    return SCORES / 'teacher-twins' / f'{hashlib.sha1(source.encode()).hexdigest()[:16]}.jsonl.zst'


def score_teachers(args):
    """One small score file per twin source (a preempted unit loses at most one source)."""
    for source in sorted({receipt['source']['path'] for _, receipt in teacher_twins()}):
        out = teacher_score_file(source)
        if not out.exists():
            run([PY, 'scripts/neuralese_data/idiom_scores.py', '--kind', 'trajectories', '--corpus',
                 'lfm3-teacher-twins', '--out', str(out), '--source', source])


def render_inputs():
    queue = json.loads(QUEUE.read_text())
    renders, scores, waiting = [], [], []
    for item in queue['items'].values():
        if item['status'] not in ('graphed', 'registered'):
            continue
        command = item['command']
        tail = command[command.index('--source') + 1:]
        sources = tail[:next((i for i, w in enumerate(tail) if w.startswith('--')), len(tail))]
        label = command[command.index('--corpus-id') + 1]
        candidates = sorted(SCORES.glob(f'*-{hashlib.sha1(json.dumps(sources).encode()).hexdigest()[:10]}.jsonl.zst'))
        if not candidates:
            waiting.append(item['id'])
            continue
        renders.append(item['out'])
        scores.append(str(candidates[0]))
    return renders, sorted(set(scores)), waiting


def build(args):
    out = (REPO / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    renders, scores, waiting = render_inputs()
    # The serving mode is program-header (policy serving_mode): twins rendered with the full system prompt stay out
    # until they are re-rendered; --include-teachers takes program-header twins only.
    twins = [(d, r) for d, r in teacher_twins() if teacher_score_file(r['source']['path']).exists()
             and args.include_teachers and r.get('prompt_rendering') == 'program-header']
    scores += sorted({str(teacher_score_file(r['source']['path'])) for _, r in twins})
    teacher_graphs = [str((d / 'train/native-graphs.jsonl.zst').relative_to(REPO)) for d, _ in twins]
    teacher_test = [str((d / 'test/native-graphs.jsonl.zst').relative_to(REPO)) for d, _ in twins]
    communication = []
    if COMMUNICATION.exists():
        listed = json.loads(COMMUNICATION.read_text())
        for item in listed.get('outputs', listed) if isinstance(listed, dict) else listed:
            if item.get('split', 'train') == 'train' and '/test/' not in item['path']:
                communication.append({'kind': item['kind'], 'path': str(Path(item['path']).resolve().relative_to(REPO))
                                      if Path(item['path']).resolve().is_relative_to(REPO) else item['path']})
    import student_tokenizers
    tokenizer = str(student_tokenizers.resolve('lfm3')['path'])
    fingerprint = line_backbones.tokenizer_contract('lfm3')['expanded_fingerprint']
    config = {'policy': POLICY, 'renders': sorted(renders), 'scores': scores, 'teacher_graphs': teacher_graphs,
              'teacher_graphs_test': teacher_test, 'communication': communication, 'held_packets': HELD_PACKETS,
              'objective_tokenizer': tokenizer, 'expanded_tokenizer_fingerprint': fingerprint,
              'renders_waiting_for_scores': waiting,
              'prior': [str((REPO / prior).resolve().relative_to(REPO)) for prior in args.prior]}
    mixture = out / 'mixture'
    if (mixture / 'mixture.json').exists():  # a resumed build keeps the config its mixture was built from
        bound = json.loads((mixture / 'mixture.json').read_text())['config']
        (out / 'config.json').write_text(json.dumps(bound['value'], indent=1) + '\n')
        if sha256_file_hex(out / 'config.json') != bound['sha256']:
            raise SystemExit('config.json differs from the one the existing mixture was built from')
    else:
        (out / 'config.json').write_text(json.dumps(config, indent=1) + '\n')
    if not (mixture / 'mixture.json').exists():
        run([PY, 'scripts/neuralese_data/build_idiom_mixture.py', '--config', str(out / 'config.json'),
             '--out', str(mixture)])
    policy = json.loads((REPO / POLICY).read_text())
    objective = policy['objective']
    held = json.loads((mixture / 'held-sources.json').read_text())['sources'][0]
    if not (out / 'ce-policy.json').exists():
        run([PY, 'scripts/neuralese_data/build_native_ce_supervision_policy.py',
             '--source', str(mixture / 'train-sources.json'),
             # An empty held view (inputs with no test rows) has nothing to declare; held CE then comes from later
             # versions and task evaluation uses the dev packet.
             *(['--held-source', str(mixture / held['path'])]
               if json.loads((mixture / 'mixture.json').read_text())['held']['rows'] else []),
             '--legacy-pair-context-weight', '0.05', '--objective-reasoning-share', str(objective['reasoning_share']),
             '--objective-tokenizer', tokenizer, '--objective-tokenizer-fingerprint', fingerprint,
             '--objective-family-balance', json.dumps(objective['family_balance']), '--out', str(out / 'ce-policy.json')])
    role_ids = [str(i) for i in line_backbones.backbone('lfm3')['role_token_ids']] \
        if 'role_token_ids' in line_backbones.backbone('lfm3') else ['125033', '125034', '125035', '125036']
    run([PY, 'scripts/neuralese_data/build_native_packed_geometry.py', '--source', str(mixture / 'train-sources.json'),
         '--tokenizer-fingerprint', fingerprint, '--role-ids', *role_ids, '--out', str(out / 'geometry')])
    geometry = json.loads((out / 'geometry' / 'packed-geometry-build.json').read_text())
    run([PY, 'scripts/coreweave/prepare_sft_capacity_descriptor.py',
         '--source', str(mixture / 'train-sources.json'), '--geometry', geometry['geometry']['path'],
         '--out', str(out / 'sft-capacity-descriptor.json')])
    bind = lambda p: {'path': str(p), 'sha256': sha256_file_hex(p)}
    inputs = {'schema': 'natlang.lfm3-joint-sft-learner-inputs/1', 'model_id': line_backbones.model_id('lfm3'),
              'expanded_tokenizer_fingerprint': fingerprint, 'role_token_ids': [int(r) for r in role_ids],
              'native_graph_sources': bind(mixture / 'train-sources.json'),
              'held_sources': bind(mixture / 'held-sources.json'),
              'ce_supervision_policy': bind(out / 'ce-policy.json'),
              'sft_packed_geometry': geometry['geometry'],
              'memory_qualification': bind(out / 'sft-capacity-descriptor.json'), 'mixture': bind(mixture / 'mixture.json'),
              'policy': bind(REPO / POLICY), 'config': bind(out / 'config.json'),
              'note': 'pre-tokenized native graph rows and exact packed geometry: learners do no tokenization; '
                      'held/dev rows never in the train manifest; later admissions append as a new version'}
    (out / 'learner-inputs.json').write_text(json.dumps(inputs, indent=1) + '\n')
    print(json.dumps(inputs, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('score-teachers')
    b = sub.add_parser('build')
    b.add_argument('--out', required=True)
    b.add_argument('--include-teachers', action='store_true',
                   help='include teacher twins whose receipt declares prompt_rendering program-header')
    b.add_argument('--prior', action='append', default=[],
                   help='earlier version mixture dir (e.g. runs/lfm3-mixture/joint-sft-v1/mixture): an append keeps '
                        'its train sources unchanged and its rows/decisions/held groups taken')
    a = p.parse_args()
    {'score-teachers': score_teachers, 'build': build}[a.cmd](a)


if __name__ == '__main__':
    main()
