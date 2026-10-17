#!/usr/bin/env python3
"""Report protected task quality using an already-owned live native actor."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from natlang_neuralese.common.jsonio import write_json_durable
from natlang_neuralese.eval.task_cadence import TaskEval, parse_task_eval
from natlang_neuralese.rl.grpo import Server


def evaluate(*, endpoint, checkpoint_sha256, spec_path, spec_sha256, out, step):
    raw = Path(spec_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != spec_sha256:
        raise ValueError('held actor task specification changed')
    spec = parse_task_eval(raw.decode())
    if not spec or not spec.get('neuralese_server') or spec.get('async'):
        raise ValueError('held actor monitoring needs synchronous native transport')
    server = Server(endpoint, timeout=30)
    def identity():
        info = json.loads(server.get_bytes('/v1/neuralese/info'))
        policy = info.get('rl_policy_binding') or {}
        native = info.get('native_state') or {}
        if (policy.get('checkpoint_sha256') != checkpoint_sha256 or
                native.get('checkpoint_sha256') != checkpoint_sha256 or not native.get('enabled') or
                info.get('prompt_rendering') != spec['prompt_rendering']):
            raise ValueError('held task actor differs from the exact current native policy')
        return info
    before = identity()
    models = json.loads(server.get_bytes('/v1/models'))['data']
    if len(models) != 1 or not isinstance(models[0].get('id'), str):
        raise ValueError('held actor needs one explicit model identity')
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    contract = {'checkpoint_sha256': checkpoint_sha256, 'spec_sha256': spec_sha256,
                'step': step, 'endpoint': endpoint, 'model': models[0]['id'],
                'native_identity': before['rl_policy_binding']}
    with (out/'evaluation.lock').open('a') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        completed = out/'completed.json'
        if completed.exists():
            saved = json.loads(completed.read_text())
            if saved['contract'] != contract:
                raise ValueError('held actor monitoring directory was rebound')
            return saved
        intent = out/'intent.json'
        if intent.exists() and json.loads(intent.read_text()) != contract:
            raise ValueError('held actor monitoring intent was rebound')
        write_json_durable(intent, contract)
        engine = SimpleNamespace(native_state_enabled=True, model_name=models[0]['id'],
                                 prompt_rendering=spec['prompt_rendering'], backbone=None)
        evaluator = TaskEval(spec, out, engine, prompt_rendering=spec['prompt_rendering'],
                             external_actor={'url': endpoint})
        try:
            summary = evaluator.run(step)
            result = {'status': 'finished', 'summary': summary}
        except Exception as error:
            # Task quality remains report-only; preserve an actual monitoring
            # failure rather than stopping useful TRAIN work for collector IO.
            result = {'status': 'monitoring-failed', 'eligible': False,
                      'error_type': type(error).__name__, 'error': str(error)}
        after = identity()
        if after != before:
            raise ValueError('held actor identity changed during evaluation')
        receipt = {'contract': contract, 'actor_info': before, **result,
                   'training_admission': False, 'separate_held_outputs': True}
        write_json_durable(completed, receipt)
        return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--spec', required=True, type=Path)
    parser.add_argument('--spec-sha256', required=True)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--step', required=True, type=int)
    args = parser.parse_args(argv)
    print(json.dumps(evaluate(endpoint=args.endpoint, checkpoint_sha256=args.checkpoint_sha256,
                              spec_path=args.spec, spec_sha256=args.spec_sha256,
                              out=args.out, step=args.step)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
