"""Program-adaptation and learned-optimizer recipe stages (owner 2026-10-28 "LFM3 focus", 2026-11-03 "firmly include
tuning and meta-learning in the recipe").

Two stage kinds wrap the existing tuning and learned-optimizer machinery; they add no trainer:

``program_adaptation`` (module ``natlang_neuralese.tune.stage``)
    Per program of a supply manifest: privileged training references, the untuned development arm, ``natlang tune
    --rounds`` (privileged self-teaching, SFT/RL/RFT rounds selected on development), the tuned development arm, one
    confirmation run of each release with measured development gain, and the distillation export of verified
    training-world executions for ordinary source admission. Entry: the bound backbone parent shows headroom
    (0 < untuned development success < 1) on at least ``entry_min_headroom_programs`` programs. Exit: at least
    ``exit_min_programs`` programs with a paired development gain of ``min_development_gain`` roots, spanning at least
    ``exit_min_structural_shapes`` structurally different program shapes (``tune.structure``), each with its one
    confirmation run. Program tunes never advance production lineage exposure.

``learned_optimizer`` (module ``natlang_neuralese.tune.learned.stage``)
    On a completed program-adaptation stage: family split into corpus and held programs, bank-init, corpus, outer
    training cycles with ``train_basis``/``train_bank``/``train_keys`` explicitly enabled (AGENTS.md "Learned LoRA
    mixtures"), outer meta-RL on real task outcomes, meta-examples, the controller program tune, planted controls and
    the held arms. Promotion: on held programs the learned release matches or beats both the crisp baseline and the
    program's own gradient-fit release on development at no more adaptation compute than the gradient fit, with one
    confirmation run each.

``bank_metalearning`` (module ``natlang_neuralese.tune.learned.bank_stage``; the meta-learning main line, owner
2026-11-03, plans/neuralese/CONTINUOUS_BANK_METALEARNING_20261103.md)
    Continuous first-order meta-learning of the shared LoRA bank and basis over an episode pool of task families with
    verified trajectories: coefficient-only inner fits on support worlds, outer updates of the bank/basis from query
    worlds, with ``train_basis``/``train_bank`` explicitly enabled. It runs on the latest promoted parent while
    backbone training continues and follows each new parent by a recorded transition (``max_parent_lag_steps``).
    Promotion: on held families' development worlds the coefficient fit over the meta-trained bank has a lower loss
    than the same fit over the identical un-meta-trained bank, a plain per-program LoRA fit at matched fit passes and
    the unadapted parent; confirmed once on confirmation worlds. It needs no completed program adaptation; a
    ``program_adaptation`` manifest, when bound (``adaptation``), supplies per-program fits as further baselines.
    ``learned_optimizer`` remains the later optional refinement (a natlang updater over the trained bank).

``program_adaptation`` reruns per promoted parent, warm-started from the previous run's releases where the tuner can
continue them; its fits are baselines for the bank stage and distillation sources for ordinary source admission.

All bind an earned backbone checkpoint (``continue_from``), either their native predecessor's completed artifact or
an explicit behaviour-selected parent (``parent_selection``, ``train.promotion``), so they run as soon as their entry
criterion holds while backbone training continues (``recipe.py --stage``). Their reports are gate receipts; saved
artifacts and passing stages are not capability claims.
"""
from __future__ import annotations

import math

PROGRAM_ADAPTATION = 'program_adaptation'
LEARNED_OPTIMIZER = 'learned_optimizer'
BANK_METALEARNING = 'bank_metalearning'
ADAPTATION_KINDS = frozenset({PROGRAM_ADAPTATION, LEARNED_OPTIMIZER, BANK_METALEARNING})
BANK_METALEARNING_REPORT = 'natlang.bank-metalearning-report/1'
PROGRAM_ADAPTATION_REPORT = 'natlang.program-adaptation-report/1'
LEARNED_OPTIMIZER_REPORT = 'natlang.learned-optimizer-report/1'

PROGRAM_ADAPTATION_DEFAULTS = {
    'entry_min_headroom_programs': 3,
    'exit_min_programs': 12,
    'exit_min_structural_shapes': 6,
    'min_development_gain': 1,
    'confirmation': True,
    'distill': True,
    'concurrency': 4,
}
LEARNED_OPTIMIZER_DEFAULTS = {
    'held_fraction': .25,
    'min_corpus_programs': 2,
    'min_held_programs': 1,
    'train_basis': True,
    'train_bank': True,
    'train_keys': True,
    'outer_rl_per_context': 4,
    'planted_controls': True,
}

BANK_METALEARNING_DEFAULTS = {
    'held_fraction': .25,
    'outer_steps': 200,
    'eval_every': 50,
    'meta_batch': 2,
    'rank': 32,
    'components': 64,
    'component_rank': 16,
    'inner_steps': 12,
    'layer_groups': 8,
    'residual_rank': 2,
    'train_bank': True,
    'train_basis': True,
    'min_held_families': 4,
    'promotion_margin': 0.,
    'max_parent_lag_steps': 200,
}

HANDLERS = {
    PROGRAM_ADAPTATION: {
        'module': 'natlang_neuralese.tune.stage',
        'required_inputs': {'programs', 'protection'},
        'optional_inputs': {'continue_from', 'parent_selection', 'base_snapshot'},
        'argument_roles': ('programs', 'protection', 'parent_selection', 'base_snapshot'),
        'parameters': set(PROGRAM_ADAPTATION_DEFAULTS) | {'program_names', 'tuner', 'program_hours'},
        'result': 'adaptation-manifest.json'},
    LEARNED_OPTIMIZER: {
        'module': 'natlang_neuralese.tune.learned.stage',
        'required_inputs': {'protection'},
        'optional_inputs': {'continue_from', 'parent_selection', 'base_snapshot'},
        'argument_roles': ('protection', 'parent_selection', 'base_snapshot'),
        'parameters': set(LEARNED_OPTIMIZER_DEFAULTS) | {'spec_overrides'},
        'result': 'optimizer-manifest.json'},
    BANK_METALEARNING: {
        'module': 'natlang_neuralese.tune.learned.bank_stage',
        'required_inputs': {'episodes', 'protection'},
        'optional_inputs': {'continue_from', 'parent_selection', 'base_snapshot', 'adaptation'},
        'argument_roles': ('episodes', 'protection', 'parent_selection', 'base_snapshot', 'adaptation'),
        'parameters': set(BANK_METALEARNING_DEFAULTS) | {'recipe_overrides'},
        'result': 'bank-manifest.json'},
}


def _positive_int(parameters, key, minimum=1):
    if key in parameters and (type(parameters[key]) is not int or parameters[key] < minimum):
        raise ValueError(f'{key} must be an integer >= {minimum}')


def validate_parameters(kind, parameters):
    if kind == PROGRAM_ADAPTATION:
        for key in ('entry_min_headroom_programs', 'exit_min_programs', 'exit_min_structural_shapes',
                    'min_development_gain', 'concurrency'):
            _positive_int(parameters, key)
        for key in ('confirmation', 'distill'):
            if key in parameters and type(parameters[key]) is not bool:
                raise ValueError(f'{key} must be true or false')
        if parameters.get('confirmation') is False:
            raise ValueError('program adaptation exits only with one confirmation run per selected release')
        names = parameters.get('program_names')
        if names is not None and (not isinstance(names, list) or not names or len(set(names)) != len(names)
                                  or not all(isinstance(name, str) and name for name in names)):
            raise ValueError('program_names is a nonempty list of distinct program names')
        if 'tuner' in parameters and not isinstance(parameters['tuner'], dict):
            raise ValueError('tuner is the natlang tune settings object')
        hours = parameters.get('program_hours')
        if hours is not None and (type(hours) not in (int, float) or not 0 < hours <= 168):
            raise ValueError('program_hours is a positive wall budget per program (at most a week)')
    elif kind == LEARNED_OPTIMIZER:
        fraction = parameters.get('held_fraction', .25)
        if type(fraction) not in (int, float) or not 0 < fraction < 1:
            raise ValueError('held_fraction must be in (0, 1): promotion needs held families')
        for key in ('min_corpus_programs', 'min_held_programs', 'outer_rl_per_context'):
            _positive_int(parameters, key)
        for key in ('train_basis', 'train_bank', 'train_keys'):
            if parameters.get(key, True) is not True:
                raise ValueError(f'{key} stays explicitly enabled in the operational recipe (AGENTS.md "Learned LoRA '
                                 'mixtures and bases")')
        if 'planted_controls' in parameters and type(parameters['planted_controls']) is not bool:
            raise ValueError('planted_controls must be true or false')
        if 'spec_overrides' in parameters and not isinstance(parameters['spec_overrides'], dict):
            raise ValueError('spec_overrides maps a lane spec name or "*" to settings')
    elif kind == BANK_METALEARNING:
        fraction = parameters.get('held_fraction', .25)
        if type(fraction) not in (int, float) or not 0 < fraction < 1:
            raise ValueError('held_fraction must be in (0, 1): promotion needs held families')
        for key in ('outer_steps', 'eval_every', 'meta_batch', 'rank', 'components', 'component_rank', 'inner_steps',
                    'layer_groups', 'min_held_families', 'max_parent_lag_steps'):
            _positive_int(parameters, key)
        _positive_int(parameters, 'residual_rank', 0)
        for key in ('train_bank', 'train_basis'):
            if parameters.get(key, True) is not True:
                raise ValueError(f'{key} stays explicitly enabled in the operational recipe (AGENTS.md "Learned LoRA '
                                 'mixtures and bases")')
        margin = parameters.get('promotion_margin', 0.)
        if type(margin) not in (int, float) or not math.isfinite(margin) or margin < 0:
            raise ValueError('promotion_margin is a nonnegative loss margin')
        if 'recipe_overrides' in parameters and not isinstance(parameters['recipe_overrides'], dict):
            raise ValueError('recipe_overrides maps meta_bank.MetaRecipe fields to values')
    else:
        raise ValueError('not an adaptation stage kind: ' + str(kind))


def require_gate(report, kind):
    """The completed receipt of an adaptation stage, including its task-behaviour exit criteria."""
    if kind == PROGRAM_ADAPTATION:
        if report.get('schema') != PROGRAM_ADAPTATION_REPORT:
            raise ValueError('program adaptation lacks its report')
        if report.get('production_exposure_advanced') is not False:
            raise ValueError('program adaptation must not advance production lineage exposure')
        entry, exit_ = report.get('entry') or {}, report.get('exit') or {}
        if entry.get('met') is not True:
            raise ValueError('program adaptation entry criterion not met: ' + str(entry.get('reason')))
        if report.get('completed') is not True:
            raise ValueError('program adaptation has not completed: ' + str(exit_.get('reason')))
        if exit_.get('met') is not True:
            raise ValueError('program adaptation exit criterion not met: ' + str(exit_.get('reason')))
        if exit_.get('gained_programs', 0) < exit_.get('min_programs', 1) \
                or exit_.get('structural_shapes', 0) < exit_.get('min_structural_shapes', 1) \
                or exit_.get('confirmations_missing'):
            raise ValueError('program adaptation exit record is inconsistent with its criteria')
        return
    if kind == LEARNED_OPTIMIZER:
        if report.get('schema') != LEARNED_OPTIMIZER_REPORT or report.get('completed') is not True:
            raise ValueError('learned optimizer has not completed')
        if report.get('production_exposure_advanced') is not False:
            raise ValueError('the learned optimizer must not advance production lineage exposure')
        executed = report.get('outer_training') or {}
        if not all(executed.get(key) is True for key in ('train_basis', 'train_bank', 'train_keys')):
            raise ValueError('the outer training did not train the shared basis, bank and keys')
        if (executed.get('steps') or 0) < 1:
            raise ValueError('the learned optimizer earned no outer update')
        held = report.get('held') or {}
        if held.get('met') is not True:
            raise ValueError('learned optimizer held-family promotion not met: ' + str(held.get('reason')))
        return
    if kind == BANK_METALEARNING:
        if report.get('schema') != BANK_METALEARNING_REPORT or report.get('completed') is not True:
            raise ValueError('bank meta-learning has not completed')
        if report.get('production_exposure_advanced') is not False:
            raise ValueError('bank meta-learning must not advance production lineage exposure')
        executed = report.get('outer_training') or {}
        if not all(executed.get(key) is True for key in ('train_basis', 'train_bank')) or (executed.get('steps') or 0) < 1:
            raise ValueError('bank meta-learning earned no outer update of the bank and basis')
        if (report.get('held') or {}).get('met') is not True:
            raise ValueError('bank meta-learning held-family promotion not met: ' + str((report.get('held') or {}).get('reason')))
        if (report.get('confirmation') or {}).get('met') is not True:
            raise ValueError('bank meta-learning was not confirmed on confirmation worlds')
        return
    raise ValueError('no adaptation gate adapter for stage: ' + str(kind))
