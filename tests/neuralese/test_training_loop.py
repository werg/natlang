"""train/loop.py reproduces the loop idioms the trainers carried inline before C1 (parity with those idioms)."""
import os
import random
import signal

import pytest
import torch

from natlang_neuralese.train.loop import (
    Cadence, StopSignal, TrainingLoop, accumulate_gradients, capture_training_rng_state,
    commit_optimizer_step, restore_training_rng_state)


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def old_text_warmup_loop(start, total, stop_at=None, qualify_at=None):
    """The text warm-up's loop as written inline: `for _ in range(step, steps): if stop[0]: break ...`."""
    stop, step, ran = [False], start, []
    for _ in range(step, total):
        if stop[0]:
            break
        ran.append(step)
        step += 1
        if step == stop_at:
            stop[0] = True
        if step == qualify_at:
            break
    return ran, stop[0]


def old_trajectory_loop(start, total, stop_at=None, stopped_first=False):
    """The trajectory trainer's: `for step in range(start, steps): ... if stop_requested[0]: break`."""
    stop, ran = [stopped_first], []
    for step in range(start, total):
        ran.append(step)
        if step + 1 == stop_at:
            stop[0] = True
        if stop[0]:
            break
    return ran, stop[0]


@pytest.mark.parametrize('start,total,stop_at,qualify_at', [
    (0, 5, None, None), (3, 5, None, None), (5, 5, None, None), (0, 6, 3, None), (0, 6, None, 4), (0, 6, 2, 2)])
def test_text_warmup_style_loop_matches_the_inline_loop(start, total, stop_at, qualify_at):
    expected, expected_stop = old_text_warmup_loop(start, total, stop_at, qualify_at)
    stop, ran = StopSignal(), []
    loop = TrainingLoop(start, total, stop)
    for step in loop:
        ran.append(step)
        if step + 1 == stop_at:
            stop.request()
        if step + 1 == qualify_at:
            loop.finish('qualified')
            break
    assert (ran, stop.requested) == (expected, expected_stop)


@pytest.mark.parametrize('start,total,stop_at,stopped_first', [
    (0, 4, None, False), (2, 4, None, False), (0, 6, 3, False), (0, 6, None, True), (4, 4, None, True)])
def test_trajectory_style_loop_observes_a_stop_only_after_a_step_has_run(start, total, stop_at, stopped_first):
    expected, _ = old_trajectory_loop(start, total, stop_at, stopped_first)
    stop, ran = StopSignal(), []
    stop.requested = stopped_first
    loop = TrainingLoop(start, total, stop, stop_before_step=False)
    for step in loop:
        ran.append(step)
        if step + 1 == stop_at:
            stop.request()
    assert ran == expected
    if stop.requested and ran:
        assert loop.reason == 'signal' or loop.reason == 'complete'


def test_a_loop_names_why_it_ended():
    loop = TrainingLoop(0, 2)
    assert list(loop) == [0, 1] and loop.reason == 'complete'
    stop = StopSignal()
    loop = TrainingLoop(0, 9, stop)
    for step in loop:
        if step == 1:
            stop.request()
    assert loop.reason == 'signal'


def test_cadences_match_the_inline_expressions_of_both_trainers():
    clock = Clock()
    # text warm-up: step % checkpoint_every == 0 or monotonic() - last_save >= 60 * minutes
    cadence = Cadence(4, 10., clock=clock)
    last = clock()
    for step in range(1, 30):
        clock.now += 90.
        expected = step % 4 == 0 or clock() - last >= 60 * 10.
        assert cadence.due(step) == expected
        if expected:
            cadence.mark()
            last = clock()
    # trajectories: (step + 1) % every == 0 or stop or last step
    cadence = Cadence(5)
    for completed in range(1, 21):
        for stop in (False, True):
            assert cadence.due(completed, force=stop or completed == 20) == (completed % 5 == 0 or stop or completed == 20)
    assert not Cadence(0).due(7) and not Cadence(None, None).due(7)
    assert Cadence(0, 0., clock=clock).due(1)  # zero minutes means every step, as the inline expression did


def test_gradient_accumulation_matches_the_inline_idiom():
    def inline(parameters, gradients):
        for param, gradient in zip(parameters, gradients):
            if gradient is not None:
                if param.grad is None:
                    param.grad = gradient
                else:
                    param.grad.add_(gradient)

    def run(apply):
        torch.manual_seed(3)
        parameters = [torch.nn.Parameter(torch.randn(3)) for _ in range(3)]
        parameters[1].grad = torch.ones(3)
        for _ in range(2):
            apply(parameters, [torch.randn(3), torch.randn(3), None])
        return [None if p.grad is None else p.grad.clone() for p in parameters]

    expected, actual = run(inline), run(accumulate_gradients)
    assert expected[2] is None and actual[2] is None
    for want, got in zip(expected[:2], actual[:2]):
        assert torch.equal(want, got)


def test_the_optimizer_commit_runs_the_failure_hook_and_reraises():
    class Broken:
        def step(self):
            raise RuntimeError('boom')

    seen = []
    with pytest.raises(RuntimeError, match='boom'):
        commit_optimizer_step(Broken(), on_failure=lambda: seen.append('cleared'))
    assert seen == ['cleared']
    parameter = torch.nn.Parameter(torch.ones(1))
    parameter.grad = torch.ones(1)
    optimizer = torch.optim.SGD([parameter], lr=.5)
    commit_optimizer_step(optimizer)
    assert parameter.item() == .5


def test_stop_signal_sets_a_flag_and_restores_the_previous_handler():
    previous = signal.getsignal(signal.SIGUSR2)
    stop = StopSignal().install((signal.SIGUSR2,))
    assert not stop.requested
    os.kill(os.getpid(), signal.SIGUSR2)
    assert stop.requested
    stop.restore()
    assert signal.getsignal(signal.SIGUSR2) == previous


def test_rng_capture_and_restore_replay_the_same_draws():
    state = capture_training_rng_state('cpu')
    first = (random.random(), torch.rand(2))
    restore_training_rng_state(state, 'cpu')
    second = (random.random(), torch.rand(2))
    assert first[0] == second[0] and torch.equal(first[1], second[1])
    assert state['cuda_rng'] == []
