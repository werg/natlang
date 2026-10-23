"""Real-size serving/sampling/backward validation, never production completion.

A controlled positive gradient coefficient checks machinery, not task reward or
model improvement. Optional harness execution is separately reported with the
real shared verifier. No production checkpoint is changed or promoted.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path


def validate_value_channel(engine, out):
    """Real-width direct writer, durable block transport and consumer backward."""
    import torch
    from ..model.neuralese_projection import NeuraleseProjection
    from ..serve.engine import GenerationRequest
    from ..serve.grad import GradSession
    from ..serve.store import encode_block
    from ..common.hashing import sha256_file_hex
    from ..train.trajectory_state import atomic_checkpoint
    from .graph_replay import save_bundle, restore_bundle, captured_producer_leaves
    from .sampling import captured_logps
    for parameter in list(engine.backbone.parameters()) + list(engine.heads.parameters()):
        parameter.requires_grad_(False)
    width = engine.backbone.config.hidden_size
    projection = NeuraleseProjection(width, embedding_width=engine.backbone.embedding_weight.shape[1], hidden=64)
    projection = projection.to(device=engine.device, dtype=engine.backbone.embedding_weight.dtype)
    with torch.no_grad():
        projection.linear.weight.normal_(0, .001)
    previous = getattr(engine, 'neuralese_projection', None)
    engine.neuralese_projection = projection
    try:
        parent = GenerationRequest(messages=[{'role': 'user', 'content': 'Write a compact state for adding two and two.'}],
            forced=[{'neuralese': 'write'}], neuralese_length=2, max_tokens=8,
            temperature=1., seed=901, rl_capture=True)
        written = engine.generate(parent)
        block = written['neuralese']['blocks'][0]['id']
        reader = GenerationRequest(messages=[{'role': 'user', 'content': [
            {'type': 'neuralese', 'id': block}, {'type': 'text', 'text': 'Answer using the state above.'}]}],
            max_tokens=8, temperature=1., seed=902, rl_capture=True)
        read = engine.generate(reader)
        producer = {'messages': parent.messages, 'reply': written['choices'][0]['message'],
                    'sampling': written['x_natlang_rl_capture']}
        from ..common.jsonio import write_json_durable
        write_json_durable(out / 'value-channel-transcript.json', {'writer': written, 'reader': read})
        projection_path = out / 'diagnostic-writer.pt'
        atomic_checkpoint(projection_path, projection.state_dict())
        policy = 'diagnostic-writer:' + sha256_file_hex(projection_path)
        save_bundle([producer, read], out / 'value-blocks', policy_id=policy,
                    fetch=lambda bid: encode_block(engine.store.get(bid)))
        restore_bundle(engine, out / 'value-blocks/manifest.json', policy_id=policy)
        session = GradSession(engine)
        with torch.no_grad():
            detached, _ = captured_logps(session, read['x_natlang_rl_capture'])
        leaves = captured_producer_leaves(session, read['x_natlang_rl_capture'], [producer])
        chained, _ = captured_logps(session, read['x_natlang_rl_capture'], leaves=leaves)
        forward_delta = float((chained.detach() - detached).abs().max())
        (-chained.mean()).backward()
        gradient = projection.linear.weight.grad
        if gradient is None or not torch.isfinite(gradient).all() or not bool(gradient.abs().sum() > 0):
            raise RuntimeError('real-width Neuralese consumer has no finite writer gradient')
        return {'width': width, 'vectors': engine.lookup(block).length,
            'forced_writer_sampled_actions': len(written['x_natlang_rl_capture']['samples']),
            'reader_sampled_actions': chained.numel(), 'recorded_forward_logp_max_abs': forward_delta,
            'writer_gradient_norm': float(gradient.float().norm()),
            'durable_bundle': str(out / 'value-blocks/manifest.json'),
            'projection_is_unqualified_diagnostic': True}
    finally:
        engine.neuralese_projection = previous


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--heads', required=True)
    p.add_argument('--base')
    p.add_argument('--checkpoint')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--prompt-rendering', default='full')
    p.add_argument('--diagnose-fp32', action='store_true', help='small LFM numerical localization only')
    p.add_argument('--sdpa-backend', choices=['default', 'flash', 'math'], default='flash')
    p.add_argument('--runtime', type=Path)
    p.add_argument('--wired-value-harness', action='store_true', help='forced protocol fixture, never task reward')
    p.add_argument('--cases', type=Path)
    p.add_argument('--port', type=int, default=8096)
    p.add_argument('--harness-rows', type=int, default=2)
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    import torch
    from ..common.jsonio import write_json_durable
    from ..serve import load_engine
    from ..serve.engine import GenerationRequest
    from ..serve.grad import GradSession
    from ..train.backbone_policy import configure_backbone_training, plain_named_tensors
    from ..train.text_warmup import copy_restored_values
    from .sampling import captured_logps
    from .policy_math import clipped_token_loss
    torch.set_num_threads(4)
    if args.device != 'cpu':
        from ..common.attention_backend import configure_sdpa_backend
        configure_sdpa_backend(args.sdpa_backend)
    engine = load_engine(base=args.base, heads_checkpoint=args.heads, device=args.device,
                         dtype=torch.bfloat16, prompt_rendering=args.prompt_rendering)
    if args.checkpoint:
        state = torch.load(args.checkpoint, mmap=True, map_location='cpu', weights_only=False)
        copy_restored_values({'backbone.' + n: q for n, q in plain_named_tensors(engine.backbone.hf).items()},
                             state['student_parameters'])
        state = None
    report = {'schema': 'natlang.rl-local-validation/1', 'scope': 'diagnostic machinery only',
              'heads': args.heads, 'checkpoint': args.checkpoint, 'base': args.base,
              'production_complete': False, 'task_improvement_claimed': False,
              'sdpa_backend': args.sdpa_backend}
    request = GenerationRequest(messages=[{'role': 'user', 'content': 'What is two plus two?'}],
                                max_tokens=16, temperature=1., seed=876, rl_capture=True)
    response = engine.generate(request)
    write_json_durable(args.out / 'sampling-response.json', response)
    session = GradSession(engine)
    with torch.no_grad():
        current, sampled = captured_logps(session, response['x_natlang_rl_capture'])
    report['sampled_tokens'] = current.numel()
    report['sampling_replay_max_abs'] = float((current - sampled).abs().max())
    ratio = (current - sampled).exp()
    report['initial_importance_ratios'] = {'min': float(ratio.min()), 'max': float(ratio.max()),
        'clipped_fraction': float(((ratio < .8) | (ratio > 1.28)).float().mean())}
    if args.diagnose_fp32:
        active = configure_backbone_training(engine.backbone, 'latent')
        differentiable, _ = captured_logps(session, response['x_natlang_rl_capture'])
        report['parallel_grad_vs_no_grad_max_abs'] = float((differentiable.detach() - current).abs().max())
        del differentiable
        for _, parameter in active:
            parameter.requires_grad_(False)
    # Replay the same IDs through the actual serving step path. This separates
    # capture mistakes from parallel teacher-force/cache execution differences.
    capture = response['x_natlang_rl_capture']
    if all(kind == 'tok' for kind, _ in capture['items']):
        original_next, original_plan = engine._next_token, engine._prompt_plan
        forced_logps = []
        def exact_next(seq):
            index = len(forced_logps)
            if index == len(capture['samples']):
                return None
            event = capture['samples'][index]
            forced_logps.append(float(torch.log_softmax(seq.logits[0].float(), -1)[event['token']]))
            return event['token']
        try:
            engine._next_token = exact_next
            engine._prompt_plan = lambda *_: capture['prompt']
            engine.generate(GenerationRequest(messages=request.messages, tools=request.tools,
                max_tokens=len(capture['samples']) + 1, temperature=1., seed=876))
        finally:
            engine._next_token, engine._prompt_plan = original_next, original_plan
        reference = torch.tensor(forced_logps, device=sampled.device)
        report['serving_step_replay_max_abs'] = float((reference - sampled).abs().max())
        report['parallel_vs_step_logp_deltas'] = (current - reference).cpu().tolist()
    report['value_channel'] = validate_value_channel(engine, args.out)
    if args.wired_value_harness:
        from ..eval.task_cadence import restore_neuralese_projections
        from .value_harness import validate as validate_harness
        previous = getattr(engine, 'neuralese_projection', None)
        restore_neuralese_projections(engine, {'neuralese_projection': torch.load(
            args.out / 'diagnostic-writer.pt', weights_only=True), 'identity': {}})
        try:
            report['wired_value_harness'] = validate_harness(engine, args.out / 'value-harness', args.runtime)
        finally:
            engine.neuralese_projection = previous
    write_json_durable(args.out / 'validation.json', report)
    if args.cases:
        # Complete real harness rollouts first; no updated weights are served
        # while collecting. The production collector/reward implementation is reused.
        from .production import collect
        from ..common.hashing import sha256_file_hex
        policy = sha256_file_hex(args.checkpoint or args.heads)
        engine.rl_policy_binding = {'checkpoint_sha256': policy, 'neuralese': False}
        engine.start()
        from ..serve.http import serve
        server = serve(engine, '127.0.0.1', args.port)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            report['harness'] = collect(cases=args.cases, out=args.out / 'harness',
                endpoints=[f'http://127.0.0.1:{args.port}'], policy_id=policy, rows=args.harness_rows,
                offset=0, group=2, workers=2, root_mode='student',
                runtime=args.runtime, seed_base=770000, neuralese_server=False)
        finally:
            server.shutdown()
            server.server_close()
            engine.stop()
        write_json_durable(args.out / 'validation.json', report)
    # This coefficient tests gradient plumbing; it is NOT a fabricated task
    # reward and is never admitted to production training data.
    named = configure_backbone_training(engine.backbone, 'latent')
    engine.backbone.checkpoint_layers = True
    for q in engine.heads.parameters():
        q.requires_grad_(False)
    # Same optimizer implementation and in-backward partitioning as production;
    # the diagnostic starts fresh optimizer slots and never replaces a run.
    from ..train.optim import PortMuonAdamW, latent_entries, latent_split, latent_options
    named = [('backbone.' + n, q) for n, q in named]
    latent_named, other = latent_split(named, 'muonsr')
    config = engine.backbone.hf.config
    head_dim = getattr(config, 'head_dim', None) or config.hidden_size // config.num_attention_heads
    latent = latent_entries(latent_named, 'muonsr', 1e-6, head_dim)
    optimizer = PortMuonAdamW(other, lr=1e-6, vocab_size=int(engine.backbone.embedding_weight.shape[0]),
        latent=latent, latent_optimizer='muonsr',
        latent_options=latent_options('muonsr', weight_decay=0., skip_idle=True,
                                     cautious=False, ns='quintic', betas=(.9, .999)))
    optimizer.latent.step_in_backward(gated=True)
    tick = time.monotonic()
    with optimizer.latent.in_backward() as armed:
        logps, old = captured_logps(session, response['x_natlang_rl_capture'])
        loss = clipped_token_loss(logps, old, 1.).mean()
        loss.backward()
    square = float(armed.norm()) ** 2 + sum(float(q.grad.float().square().sum()) for _, q in other if q.grad is not None)
    if not 0 < square < float('inf'):
        raise RuntimeError('full-size RL backward has zero/nonfinite gradient')
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    report.update(backward_gradient_norm=square ** .5, controlled_update_seconds=time.monotonic() - tick,
                  controlled_update_only=True, gradient_coefficient_is_task_reward=False)
    if args.diagnose_fp32:
        # Same rounded weights and actual sampled IDs for both reference paths;
        # math SDPA is required for true FP32 (Flash kernels require BF16/FP16).
        from ..common.attention_backend import configure_sdpa_backend
        configure_sdpa_backend('math')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        for parameter in engine.backbone.parameters():
            parameter.requires_grad_(False)
        engine.backbone.float()
        engine.heads.float()
        with torch.no_grad():
            fp32_parallel, _ = captured_logps(session, capture)
        original_next, original_plan = engine._next_token, engine._prompt_plan
        fp32_steps = []
        def fp32_next(seq):
            index = len(fp32_steps)
            if index == len(capture['samples']):
                return None
            token = capture['samples'][index]['token']
            fp32_steps.append(float(torch.log_softmax(seq.logits[0].float(), -1)[token]))
            return token
        try:
            engine._next_token = fp32_next
            engine._prompt_plan = lambda *_: capture['prompt']
            engine.generate(GenerationRequest(messages=request.messages, tools=request.tools,
                max_tokens=len(capture['samples']) + 1, temperature=1., seed=876))
        finally:
            engine._next_token, engine._prompt_plan = original_next, original_plan
        report['fp32_math_parallel_vs_step_max_abs'] = float((fp32_parallel - torch.tensor(
            fp32_steps, device=fp32_parallel.device)).abs().max())
        report['fp32_comparison_scope'] = 'same post-diagnostic-update weights and captured IDs; no new sample'
    write_json_durable(args.out / 'validation.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
