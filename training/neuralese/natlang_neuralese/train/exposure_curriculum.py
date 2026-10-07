"""Readiness-gated sequence-depth and greedy-rollout exposure schedule.

This state machine is deliberately separate from optimizer plateaus and update
counts. A caller advances only after its matched per-stratum crisp-control
report clears the declared thresholds. Integrations must checkpoint its
``state_dict`` alongside model, optimizer, scheduler, and RNG state.
"""
from __future__ import annotations

import copy
import math


EXPOSURE_STAGES = ('gold_passes_3', 'gold_passes_5', 'gold_passes_8', 'greedy_rollout')
PASS_COUNT = {'gold_passes_3': 3, 'gold_passes_5': 5, 'gold_passes_8': 8}


class ExposureCurriculum:
    SCHEMA = 'natlang.text-warmup-exposure-curriculum/1'

    def __init__(self, *, min_tokens_per_stratum=128, max_ce_delta=.1,
                 max_embedding_mse_delta=.25, min_argmax_agreement=.9):
        if type(min_tokens_per_stratum) is not int or min_tokens_per_stratum < 1:
            raise ValueError('minimum per-stratum token count must be positive')
        for name, value in (('max_ce_delta', max_ce_delta),
                            ('max_embedding_mse_delta', max_embedding_mse_delta),
                            ('min_argmax_agreement', min_argmax_agreement)):
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(value) or value < 0 or
                    (name == 'min_argmax_agreement' and value > 1)):
                raise ValueError('invalid exposure readiness threshold: ' + name)
        self.thresholds = {
            'min_tokens_per_stratum': min_tokens_per_stratum,
            'max_ce_delta': float(max_ce_delta),
            'max_embedding_mse_delta': float(max_embedding_mse_delta),
            'min_argmax_agreement': float(min_argmax_agreement),
        }
        self.stage = EXPOSURE_STAGES[0]
        self.transitions = []
        self.greedy_completion = None

    @property
    def passes(self):
        return PASS_COUNT.get(self.stage)

    def training_passes(self, foundation_schedule_passes):
        """Use the foundation ramp through three, then only readiness promotions."""
        if type(foundation_schedule_passes) is not int or not 1 <= foundation_schedule_passes <= 3:
            raise ValueError('foundation pass ramp must be between one and three')
        if self.stage == 'greedy_rollout':
            return None
        target = self.passes
        return foundation_schedule_passes if target == 3 else max(foundation_schedule_passes, target)

    def _readiness(self, report, pass_count):
        if not isinstance(report, dict) or not isinstance(report.get('strata'), dict):
            return {'ready': False, 'reason': 'missing-strata-report', 'strata': {}}
        prefix = f'pass-{pass_count - 1}-'
        matching = {name: row for name, row in report['strata'].items()
                    if isinstance(name, str) and name.startswith(prefix)}
        if not matching:
            return {'ready': False, 'reason': 'missing-final-pass-strata', 'strata': {}}
        results = {}
        required = ('tokens', 'ce_delta', 'embedding_mse_delta', 'text_argmax_agreement', 'text_ce')
        for name, row in matching.items():
            if not isinstance(row, dict) or any(key not in row for key in required):
                results[name] = {'ready': False, 'reason': 'missing-matched-control-metrics'}
                continue
            values = {key: row[key] for key in required}
            valid = all(isinstance(value, (int, float)) and not isinstance(value, bool)
                        and math.isfinite(value) for value in values.values())
            ready = (valid and values['tokens'] >= self.thresholds['min_tokens_per_stratum'] and
                     values['ce_delta'] <= self.thresholds['max_ce_delta'] and
                     values['embedding_mse_delta'] <= self.thresholds['max_embedding_mse_delta'] and
                     values['text_argmax_agreement'] >= self.thresholds['min_argmax_agreement'])
            results[name] = {'ready': ready, 'metrics': values}
        return {'ready': all(item['ready'] for item in results.values()),
                'reason': 'all-final-pass-strata-cleared' if all(item['ready'] for item in results.values())
                          else 'one-or-more-final-pass-strata-not-ready',
                'pass_count': pass_count, 'strata': results}

    def observe_alignment(self, report, *, foundation_schedule_passes):
        """Promote 3→5→8→greedy only from held matched-control readiness."""
        if self.stage == 'greedy_rollout':
            raise ValueError('gold-pass readiness cannot advance the greedy-rollout stage')
        if (type(foundation_schedule_passes) is not int or
                not 1 <= foundation_schedule_passes <= 3):
            raise ValueError('foundation pass ramp must be between one and three')
        if foundation_schedule_passes < 3:
            return {'advanced': False, 'stage': self.stage,
                    'reason': 'foundation-three-pass-ramp-incomplete'}
        readiness = self._readiness(report, self.passes)
        if not readiness['ready']:
            return {'advanced': False, 'stage': self.stage, 'readiness': readiness}
        previous = self.stage
        next_index = EXPOSURE_STAGES.index(previous) + 1
        self.stage = EXPOSURE_STAGES[next_index]
        receipt = {'from': previous, 'to': self.stage,
                   'report_step': report.get('step'), 'readiness': readiness}
        self.transitions.append(receipt)
        return {'advanced': True, 'stage': self.stage, 'transition': copy.deepcopy(receipt)}

    def observe_greedy(self, report):
        """Record actual greedy exposure without certifying tasks or stopping quality."""
        if self.stage != 'greedy_rollout':
            raise ValueError('greedy rollout is not the active exposure stage')
        if not isinstance(report, dict) or report.get('greedy_autoregressive') is not True:
            return {'completed': False, 'reason': 'actual-greedy-rollout-evidence-required'}
        if report.get('generated_tokens', 0) < self.thresholds['min_tokens_per_stratum']:
            return {'completed': False, 'reason': 'insufficient-greedy-rollout-evidence'}
        if type(report.get('examples')) is not int or report['examples'] < 1:
            return {'completed': False, 'reason': 'greedy-example-count-required'}
        if type(report.get('max_capacity_tokens')) is not int or report['max_capacity_tokens'] < 1:
            return {'completed': False, 'reason': 'greedy-capacity-evidence-required'}
        # Completion here means only that the declared exposure was exercised;
        # downstream task/runtime qualification remains a separate gate.
        self.greedy_completion={key:report[key] for key in
            ('generated_tokens','examples','max_capacity_tokens','real_close_targets','truncated_examples')
            if key in report}
        return {'completed': True, 'stage': self.stage,
                'scope': 'greedy autoregressive exposure only; no task or runtime admission'}

    def state_dict(self):
        return copy.deepcopy({'schema': self.SCHEMA, 'thresholds': self.thresholds,
                              'stage': self.stage, 'transitions': self.transitions,
                              'greedy_completion':self.greedy_completion})

    def load_state_dict(self, state):
        if not isinstance(state, dict) or state.get('schema') != self.SCHEMA:
            raise ValueError('invalid exposure curriculum state')
        if state.get('thresholds') != self.thresholds:
            raise ValueError('exposure readiness thresholds changed')
        stage = state.get('stage')
        transitions = state.get('transitions')
        if stage not in EXPOSURE_STAGES or not isinstance(transitions, list):
            raise ValueError('invalid exposure stage or transition history')
        expected = EXPOSURE_STAGES[:EXPOSURE_STAGES.index(stage)]
        if len(transitions) != len(expected):
            raise ValueError('exposure transition history does not match the current stage')
        for index, event in enumerate(transitions):
            if (not isinstance(event, dict) or event.get('from') != EXPOSURE_STAGES[index] or
                    event.get('to') != EXPOSURE_STAGES[index + 1] or
                    not isinstance(event.get('readiness'), dict) or
                    event['readiness'].get('ready') is not True):
                raise ValueError('invalid exposure promotion receipt')
        greedy_completion=state.get('greedy_completion')
        if stage!='greedy_rollout' and greedy_completion is not None:
            raise ValueError('greedy evidence appears before the greedy stage')
        if greedy_completion is not None and (not isinstance(greedy_completion,dict) or
                type(greedy_completion.get('generated_tokens')) is not int or
                greedy_completion['generated_tokens']<self.thresholds['min_tokens_per_stratum'] or
                type(greedy_completion.get('examples')) is not int or greedy_completion['examples']<1):
            raise ValueError('invalid greedy exposure receipt')
        self.stage = stage
        self.transitions = copy.deepcopy(transitions)
        self.greedy_completion=copy.deepcopy(greedy_completion)
