"""Exact native-state forward/capture/splice verification on an existing model."""
from __future__ import annotations

import copy
from collections import Counter
import hashlib
import json
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def _execution_receipt(summary, identity, *, checkpoint_sha256, score, path, digest):
    """Separate exact observed execution from successful result-file export."""
    from .task_packet import TASK_OUTCOMES, accounted
    errors = []
    details = (score or {}).get('cases_detail')
    cases, indices = summary.get('cases'), summary.get('case_indices')
    outcome_counts = summary.get('outcome_counts')
    outcome_counts = outcome_counts if isinstance(outcome_counts, dict) else {}
    if score is None or {key: value for key, value in score.items() if key != 'cases_detail'} != summary:
        errors.append('saved score differs from the evaluated summary')
    if (type(cases) is not int or cases <= 0 or not isinstance(indices, list)
            or len(indices) != cases or any(type(index) is not int or index < 0 for index in indices)
            or len(set(indices)) != len(indices)):
        errors.append('selected cases are missing or inconsistent')
    if not isinstance(summary.get('packet_sha256'), str) or re.fullmatch(r'[0-9a-f]{64}', summary['packet_sha256']) is None:
        errors.append('evaluation packet lacks its exact digest')
    if (not isinstance(details, list) or len(details) != cases
            or any(not isinstance(item, dict) or not isinstance(item.get('id'), str)
                   or not item['id'] or item.get('outcome') not in TASK_OUTCOMES for item in details)):
        errors.append('saved score lacks all selected case outcomes')
    else:
        counts = Counter(item['outcome'] for item in details)
        if (len({item['id'] for item in details}) != cases
                or any(outcome_counts.get(name) != counts[name] for name in TASK_OUTCOMES)):
            errors.append('case identities or detailed outcome counts differ')
    if not accounted(summary):
        errors.append('evaluation has unresolved or infrastructure outcomes')
    successes = summary.get('task_successes')
    if type(successes) is not int or successes != outcome_counts.get('task_success'):
        errors.append('task successes differ from observed outcomes')
    if summary.get('historical') is True or summary.get('qualifies_current_weights') is False:
        errors.append('historical evaluation cannot qualify current weights')
    if (not isinstance(checkpoint_sha256, str) or re.fullmatch(r'[0-9a-f]{64}', checkpoint_sha256) is None
            or not isinstance(identity, dict) or identity.get('checkpoint_sha256') != checkpoint_sha256):
        errors.append('evaluation is not bound to the current checkpoint')
    receipt = {'schema': 'natlang.native-task-execution-receipt/2',
        'checkpoint_sha256': (identity if isinstance(identity, dict) else {}).get('checkpoint_sha256'), 'native_identity': identity,
        'packet_sha256': summary.get('packet_sha256'), 'case_indices': summary.get('case_indices'),
        'cases': cases, 'complete': not errors, 'eligible': summary.get('eligible'),
        'result_export_complete': summary.get('complete'), 'outcome_counts': summary.get('outcome_counts'),
        'task_successes': summary.get('task_successes'), 'task_successes_are_qualification_barrier': False,
        'score_path': str(path.resolve()) if path is not None else None,
        'score_sha256': digest, 'execution_errors': errors}
    receipt['passed'] = bool(identity and receipt['checkpoint_sha256'] == checkpoint_sha256 and
                             receipt['complete'] and receipt['eligible'] is True and digest)
    return receipt


def _read_task_score(path):
    try:
        payload = path.read_bytes()
        score = json.loads(payload)
        if not isinstance(score, dict):
            return None, None
        return score, hashlib.sha256(payload).hexdigest()
    except (OSError, ValueError):
        return None, None


def task_execution_receipt(summary, identity, *, checkpoint_sha256):
    """One exact-harness receipt contract for native SFT and joint training."""
    path = Path(summary['run'])/'task-score.json' if isinstance(summary.get('run'), str) and summary['run'] else None
    score, digest = _read_task_score(path) if path is not None else (None, None)
    return _execution_receipt(summary, identity, checkpoint_sha256=checkpoint_sha256,
                              score=score, path=path, digest=digest)


def validate_task_execution_receipt(receipt, identity):
    """Recheck v2 evidence; retain the original v1 completion contract."""
    if not isinstance(receipt, dict) or not isinstance(identity, dict):
        return False
    if (receipt.get('passed') is not True or receipt.get('native_identity') != identity
            or receipt.get('complete') is not True or receipt.get('eligible') is not True):
        return False
    if receipt.get('schema') == 'natlang.native-task-execution-receipt/1':
        return True
    if (receipt.get('schema') != 'natlang.native-task-execution-receipt/2'
            or not isinstance(receipt.get('score_path'), str) or not receipt['score_path']):
        return False
    path = Path(receipt['score_path'])
    score, digest = _read_task_score(path)
    if (score is None or digest != receipt.get('score_sha256')
            or not isinstance(score.get('run'), str) or not score['run']):
        return False
    if (Path(score['run'])/'task-score.json').resolve() != path.resolve():
        return False
    summary = {key: value for key, value in score.items() if key != 'cases_detail'}
    expected = _execution_receipt(summary, identity, checkpoint_sha256=identity['checkpoint_sha256'],
                                   score=score, path=path, digest=digest)
    return expected['passed'] and all(receipt.get(key) == value for key, value in expected.items())


def qualify_existing(model, tokenizer, roles, config, *, checkpoint_sha256, forward_atol, gradient_atol,
                     task_execution_receipt=None, backbone_optimizer_updates=0,
                     gradient_parameters=None, gradient_rtol=1e-4,before_graph=None,gradient_observer=None,minimum_source_tokens=0):
    """Verify the live trained instance without reloading a second backbone or changing training state.

    The check compares graph execution with the explicit cached reference, compares
    target CE and adapter gradients, then captures a producer span and consumes it
    by exact native splice. The trainer's optimizer/weights/RNG remain untouched.
    """
    from ..model.native_model import native_identity
    from ..model.shared_state import (Invocation, ImportSpan, capture_block, channel_checkpoint,
                                      compile_graph, native_step, restore_channel_checkpoint, splice_block)

    if type(minimum_source_tokens)is not int or minimum_source_tokens<0:
        raise ValueError('runtime reference minimum source tokens must be nonnegative')
    start = time.monotonic()
    identity = native_identity(model, config, checkpoint_sha256=checkpoint_sha256)
    def probe_tokens(text):
        tokens = tuple(tokenizer(text, add_special_tokens=False).input_ids)
        special = set(getattr(tokenizer, 'all_special_ids', ()))
        if tokens and any(token not in special for token in tokens):
            return tokens
        # A tokenizer with a deliberately tiny lexical vocabulary can map the
        # diagnostic sentence entirely to its zero/padding unknown embedding.
        # Use actual round-tripped lexical text, never synthetic payload rows.
        for _, token in sorted(tokenizer.get_vocab().items(), key=lambda item: item[1]):
            if token in special:
                continue
            text = tokenizer.decode([token], skip_special_tokens=False)
            candidate = tuple(tokenizer(text, add_special_tokens=False).input_ids)
            if candidate and any(value not in special for value in candidate):
                return candidate
        raise ValueError('native runtime reference needs a round-tripped lexical token')
    source = probe_tokens('Write a useful state: the account balance is 42 after adding seven units.')
    target = probe_tokens('Read the state and return the account balance as an integer.')
    if not source or not target:
        raise ValueError('native runtime reference needs nonempty source and consumer tokenizations')
    if minimum_source_tokens>len(source):
        # Repeat existing round-tripped lexical tokens to exercise the selected
        # token-local execution boundary; this is reference input, never training data.
        source=(source*((minimum_source_tokens+len(source)-1)//len(source)))[:minimum_source_tokens]
    payload_width = min(4, len(source))
    lo = len(source) - payload_width + 1
    source = source[:-payload_width] + (roles.produce_open,) + source[-payload_width:] + (roles.produce_close,)
    hi = lo + payload_width
    prefix_width = min(2, len(target)-1)
    target = target[:prefix_width] + (roles.consume_open, roles.consume_close) + target[prefix_width:]
    import_at = prefix_width + 1
    plan = compile_graph([Invocation('source', source, producing=((lo, hi),)),
                          Invocation('target', target, imports=(ImportSpan(import_at, 'source', lo, hi),))])
    positions = torch.arange(len(source) + import_at, plan.token_count - 1, device=model.input_rows.device)
    labels = torch.tensor(target[import_at+1:], device=model.input_rows.device)
    previous_mode, previous_checkpointing = model.training, model.port.checkpoint_layers
    complete=None
    adapter = model.adapter.up.weight
    try:
        model.eval()
        model.port.checkpoint_layers = True
        if gradient_parameters is not None:
            from .native_gradients import production_inventory
            named,gradient_roles=production_inventory(model,gradient_parameters)
            ids=plan.tensor([token for node in plan.invocations for token in node.token_ids],model.input_rows.device)
            from ..train.loop import capture_training_rng_state
            graph_rng=capture_training_rng_state(model.input_rows.device)
            flat_inputs=model.embed(ids[None])
            if before_graph is not None:before_graph()
            flat=model(plan,embeddings=flat_inputs)
            reference=None
        else:
            flat = model(plan)
            reference = model(plan, reference=True)

        def objective(result):
            logits = model.logits(result['h_final'][:, positions]).float()
            return F.cross_entropy(logits.flatten(0, 1), labels)

        flat_loss = objective(flat)
        reference_loss = None if reference is None else objective(reference)
        if gradient_parameters is not None:
            from .native_gradients import compare_complete_gradients,validate_complete_proof
            from ..common.hashing import canonical_json_sha256_hex
            from ..model.native_graph_plan import invocation_dict
            from ..train.loop import restore_training_rng_state
            # The cached forward must not mutate backend state retained by flat VJP.
            # Construct it only after the flat derivatives have been collected.
            def cached_forward():
                nonlocal reference,reference_loss
                if before_graph is not None:before_graph()
                restore_training_rng_state(graph_rng,model.input_rows.device)
                reference_inputs=model.embed(ids[None])
                reference=model(plan,embeddings=reference_inputs,reference=True)
                reference_loss=objective(reference)
                return reference_loss,reference_inputs
            complete=compare_complete_gradients(flat_loss,None,named,flat_inputs,None,
                atol=gradient_atol,rtol=gradient_rtol,cached_factory=cached_forward,gradient_observer=gradient_observer)
            case={'invocations':[invocation_dict(node)for node in plan.invocations],
                'positions':positions.tolist(),'labels':labels.tolist(),'loss':'consumer-cross-entropy/mean'}
            complete.update(native_identity=identity,roles=gradient_roles,case=case,
                case_sha256=canonical_json_sha256_hex(case),model_config=config,
                model_config_sha256=canonical_json_sha256_hex(config),
                training_backend_execution={'configuration':{'enabled':False},'executed_projections':{},'passed':True},
                frozen_vocabulary_io={'embedding_requires_grad':model.port.embedding_weight.requires_grad,
                    'readout_requires_grad':model.port.output_weight.requires_grad})
            if complete['passed']:
                try:validate_complete_proof(complete,identity,gradient_atol=gradient_atol)
                except ValueError as error:complete.update(passed=False,error=str(error))
            adapter_name=next(name for name,value in named if value is adapter)
            adapter_comparison=complete['comparisons'][adapter_name]
            gradient_error=adapter_comparison.get('max_abs',float('inf'))
            gradient_nonzero=adapter_comparison.get('actual_l2',0)>0
            gradient_finite=adapter_comparison.get('finite')is True
        else:
            flat_gradient = torch.autograd.grad(flat_loss, adapter, retain_graph=False)[0]
            reference_gradient = torch.autograd.grad(reference_loss, adapter, retain_graph=False)[0]
            gradient_error = float((flat_gradient - reference_gradient).abs().max())
            gradient_nonzero=bool(flat_gradient.abs().sum()>0)
            gradient_finite=bool(torch.isfinite(flat_gradient).all())
        forward_error = float((flat['h_final'] - reference['h_final']).detach().abs().max())
        ce_error = abs(float(flat_loss.detach()) - float(reference_loss.detach()))
        del flat, reference, flat_loss, reference_loss

        with torch.no_grad():
            if before_graph is not None:before_graph()
            result = model(plan)
            block = capture_block(result, 'source', lo, hi,
                text=tokenizer.decode(source[lo:hi], skip_special_tokens=False), **identity)
            if before_graph is not None:before_graph()
            prefix = native_step(model, torch.tensor([target[:import_at]], device=model.input_rows.device))
            cache = splice_block(model, prefix['cache'], block,
                **{key: value for key, value in identity.items() if key != 'architecture'})
            suffix = native_step(model, torch.tensor([target[import_at:]], device=model.input_rows.device), cache)
            incremental_error = float((suffix['h_final'] - result['h_final'][:, len(source)+import_at:]).abs().max())
            saved_channel = copy.deepcopy(channel_checkpoint(model))
            before = {key: value.detach().clone() for key, value in saved_channel['adapter'].items()}
            restore_channel_checkpoint(model, saved_channel)
            restore_exact = all(torch.equal(model.adapter.state_dict()[key], value)
                                for key, value in before.items())
        del result, prefix, suffix, cache, block
    finally:
        model.port.checkpoint_layers = previous_checkpointing
        model.train(previous_mode)
    passed = (forward_error <= forward_atol and incremental_error <= forward_atol and
              ce_error <= forward_atol and gradient_error <= gradient_atol and
              gradient_finite and gradient_nonzero and restore_exact and (complete is None or complete['passed']))
    return {'schema': 'natlang.native-runtime-certificate/1', 'passed': passed,
            'checkpoint_sha256': checkpoint_sha256, 'native_identity': identity,
            'forward_max_abs': forward_error, 'consumer_ce_abs': ce_error,
            'consumer_gradient_max_abs': gradient_error, 'incremental_max_abs': incremental_error,
            'channel_restore_exact': restore_exact, 'forward_atol': forward_atol,
            'gradient_atol': gradient_atol, 'task_execution_receipt': task_execution_receipt,
            'gradient_scope':'adapter-channel-only'if complete is None else'complete-production-trainables',
            'complete_gradient_qualification':complete,
            'task_successes_are_qualification_barrier': False,
            'backbone_optimizer_tested': backbone_optimizer_updates > 0,
            'backbone_optimizer_updates': int(backbone_optimizer_updates),
            'task_reward_evaluated': bool(task_execution_receipt),
            'elapsed_seconds': time.monotonic() - start}
