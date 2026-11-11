"""A policy (RL) stage entered from SFT-fed momentum starts its first moments at zero (w5 RL rounds 2-3, 2026-11-11)."""
import torch

from natlang_neuralese.train.optim import PortMuonAdamW, momentum_reset_needed


def _optimizer():
    torch.manual_seed(3)
    head = torch.nn.Parameter(torch.randn(8) * .1)
    q = torch.nn.Parameter((torch.randn(32, 64) * .02).to(torch.bfloat16))
    optimizer = PortMuonAdamW([('heads.h', head)], lr=1e-3, vocab_size=1000, latent=[('backbone.q', q, 1e-4, 1)],
                              latent_optimizer='muonsr', latent_options={'rounding': 'kahan'})
    return optimizer, head, q


def _step(optimizer, head, q, scale, seed):
    generator = torch.Generator().manual_seed(seed)
    head.grad = torch.randn(head.shape, generator=generator) * scale
    q.grad = (torch.randn(q.shape, generator=generator) * scale).to(torch.bfloat16)
    optimizer.step()


def test_reset_clears_first_moments_only():
    optimizer, head, q = _optimizer()
    for seed in range(4):
        _step(optimizer, head, q, 10., seed)
    aux, latent = optimizer.auxiliary.state[head], optimizer.latent.state[q]
    assert aux['exp_avg'].abs().sum() > 0 and latent['momentum'].abs().sum() > 0
    second, step, compensation = aux['exp_avg_sq'].clone(), aux['step'].clone(), latent['compensation'].clone()
    cleared = optimizer.reset_momentum()
    assert cleared['buffers'] == 2 and cleared['elements'] == head.numel() + q.numel()
    assert aux['exp_avg'].abs().sum() == 0 and latent['momentum'].abs().sum() == 0
    assert torch.equal(aux['exp_avg_sq'], second) and torch.equal(aux['step'], step)
    assert torch.equal(latent['compensation'], compensation)


def test_after_reset_a_small_policy_gradient_steers_muon_instead_of_the_stale_momentum():
    """Muon orthogonalizes g + beta*m: the step has full size whatever g is, so an SFT-sized buffer owns its direction."""
    def run(reset):
        optimizer, head, q = _optimizer()
        for seed in range(6):                               # SFT-sized gradients leave a large momentum
            _step(optimizer, head, q, 10., seed)
        stale = optimizer.latent.state[q]['momentum'].float().clone()
        if reset:
            optimizer.reset_momentum()
        before = (q.float() + optimizer.latent.state[q].get('compensation', 0)).clone()
        generator = torch.Generator().manual_seed(99)
        gradient = (torch.randn(q.shape, generator=generator) * 1e-3).to(torch.bfloat16)   # policy-sized gradient
        for _ in range(3):
            q.grad = gradient.clone()
            head.grad = torch.zeros_like(head)
            optimizer.step()
        moved = (q.float() + optimizer.latent.state[q]['compensation']) - before
        cosine = lambda a, b: float((a.flatten() @ b.flatten()) / (a.norm() * b.norm()))
        return cosine(moved, -stale), cosine(moved, -gradient.float())
    stale_cos, gradient_cos = run(reset=False)
    assert abs(stale_cos) > 5 * abs(gradient_cos) and stale_cos > .1, (stale_cos, gradient_cos)
    stale_cos, gradient_cos = run(reset=True)
    assert gradient_cos > .2 and abs(stale_cos) < gradient_cos, (stale_cos, gradient_cos)


def test_only_entering_a_policy_stage_from_other_momentum_resets():
    assert momentum_reset_needed(None, 'policy') and momentum_reset_needed('other', 'policy')
    assert not momentum_reset_needed('policy', 'policy')           # a round continues the previous round's momentum
    assert not momentum_reset_needed('policy', 'other') and not momentum_reset_needed('other', 'other')
    assert not momentum_reset_needed('other', 'policy', keep=True)


def test_actual_trainer_resets_on_entering_policy_stage_and_records_it(tmp_path, monkeypatch):
    """The real trainer resume: CE keeps momentum, the first policy-stage resume zeroes it and records the transition,
    a policy round after a policy round keeps its own momentum, a later CE stage keeps the history."""
    import json
    from test_native_source_fetch_recovery import source
    from natlang_neuralese.rl import native_replay
    from natlang_neuralese.train import native_state_sft as trainer
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE', str(tmp_path / 'cache'))
    base = tmp_path / 'base'; paths = source(base, 'lfm'); row = json.loads(paths['train'].read_text())
    paths['train'].write_text(''.join(json.dumps({**row, 'id': str(i), 'source_groups': [str(i)]}) + '\n' for i in range(5)))
    out = tmp_path / 'run'
    args = ['--base', str(base), '--dtype', 'bfloat16', '--rank', '4', '--device', 'cpu',
        '--native-graphs', str(paths['train']), '--held-native-graphs', str(paths['test']), '--epochs', '1',
        '--batch-graphs', '1', '--accumulate-batches', '1', '--eval-every', '1', '--checkpoint-every', '1',
        '--checkpoint-seconds', '0', '--seed', '23', '--refresh-after-updates', '1', '--out', str(out)]

    resets = []

    def run(policy_stage=False, *extra):
        with monkeypatch.context() as patch:
            original = PortMuonAdamW.reset_momentum
            def spy(self):
                held = sum(float(state['momentum'].float().abs().sum()) for state in self.latent.state.values() if 'momentum' in state)
                result = original(self)
                resets.append((held, sum(float(state['momentum'].float().abs().sum()) for state in self.latent.state.values()
                                         if 'momentum' in state)))
                return result
            patch.setattr(PortMuonAdamW, 'reset_momentum', spy)
            if policy_stage:  # a CE run stands in for the policy stage: the trainer asks isinstance(objective, policy)
                patch.setattr(native_replay, 'NativePolicyObjective', type(None))
            assert trainer.main([*args, *extra]) == 0
        training = torch.load(out / 'checkpoint.pt', weights_only=False)['training']
        return training['options'], training['optimizer']['muonsr']['state']

    def momentum(states):
        return sum(float(state['momentum'].float().abs().sum()) for state in states.values() if 'momentum' in state)

    options, states = run()
    assert options['optimizer_momentum_stage'] == 'other' and options['optimizer_momentum_history'] == []
    before = momentum(states); assert before > 0
    options, states = run()                                                  # CE after CE: nothing reset
    assert options['optimizer_momentum_history'] == []
    options, states = run(True)                                              # first policy-stage resume
    assert options['optimizer_momentum_stage'] == 'policy' and len(options['optimizer_momentum_history']) == 1
    record = options['optimizer_momentum_history'][0]
    assert record['from_stage'] == 'other' and record['cleared']['buffers'] > 0
    assert len(resets) == 1 and resets[0][0] > 0 and resets[0][1] == 0.        # nonzero momentum entered, zero left
    options, _ = run(True)                                                   # policy after policy: carried, no new reset
    assert len(options['optimizer_momentum_history']) == 1
    assert len(resets) == 1
    options, _ = run()                                                       # a later CE stage keeps the history
    assert options['optimizer_momentum_stage'] == 'other' and len(options['optimizer_momentum_history']) == 1
    options, _ = run(True, '--keep-optimizer-momentum')                      # opt-out: no reset record
    assert len(options['optimizer_momentum_history']) == 1
    assert len(resets) == 1
