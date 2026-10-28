"""One program's adaptation problem, read from the tuner's own plan and build (plan "Turning existing training into
optimizer training data"): the same ``natlang.tune-plan/v1`` that `tune.build` fits, its admitted training-split turns
lowered exactly as the tuner lowers them (reply supervision only), the same initial soft states, the same scope tree,
and, for a finished build, the tuned node and soft-state versions at every committed update.

`apply` (episodes and services), `targets` (labels) and `collect` (policy records) all start here, so a tuner fit, an
optimizer target and an optimizer release are one representation: native soft states keyed like `tune.soft.soft_key`,
and per-scope ``xs`` adapters that slice the bank's basis (a fit with ``recipe.basis`` set to the bank basis) or, for
older fits on another basis, run natively and are compared through their exact effective factors.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import LearnedOptimizerError
from ..build import import_archive, load_plan, persist_artifacts, _write

PROGRAM_SCHEMA = 'natlang.lopt.program-context/1'


def default_scopes(manifest, ranks):
    """Every scope level the fit trains (rank > 0) gets a view of the whole bank by default."""
    return {level: None for level, rank in (ranks or {'tree': 8, 'folder': 4, 'lambda': 2}).items() if rank}


class ProgramContext:
    """``engine`` is the exact student the plan names (checked against the manifest); ``scopes`` maps scope levels to
    view sizes (None: the full bank); ``out`` holds this context's own initialization when no build supplies one;
    ``occurrences``: an optional subset of training occurrences (support episodes); ``soft``: the soft-state
    initialization of the program's full context (episodes of one program share it)."""

    def __init__(self, engine, bank, plan_path, *, out, scopes=None, build=None, occurrences=None, soft=None):
        from ..rows import TURN_SCHEMA, lower_rows
        from .. import TuneMathError
        from .evidence import build_view, demonstration_facts
        from .fit import SupportSet, placeholder_adapters
        from .views import select_view
        self.engine, self.bank, self.plan_path = engine, bank, Path(plan_path)
        self.plan, self.manifest, self.turns, _ = load_plan(plan_path)
        base = self.manifest['base']
        if base != {key: engine.native_identity[key] for key in base}:
            raise LearnedOptimizerError('the program plan was prepared for another exact student')
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.build = Path(build) if build else None
        import_archive(engine,
            [*(self.plan_path.parent / item for item in self.plan.get('artifacts', [])),
             *([self.build / 'artifacts'] if self.build else []), self.out / 'artifacts'],
            [*((self.plan_path.parent / item).parent / 'states' for item in self.plan.get('artifacts', [])),
             *([self.build / 'states'] if self.build else []), self.out / 'states'])
        self.soft = soft if soft is not None else self._soft_initialization()
        initial = {key: record['state'] for key, record in self.soft.items()}
        turns = [turn for turn in self.turns if occurrences is None or _occurrence(turn) in occurrences]
        self.rows = lower_rows(engine, turns, prompt_weight=0., soft=initial)
        self.support = SupportSet(engine, self.manifest, self.rows)
        levels = {node['id']: node['level'] for node in self.manifest['tree']}
        scopes = scopes if scopes is not None else default_scopes(self.manifest, (self.plan.get('recipe') or {}).get('ranks'))
        self.scopes = sorted(node for node, level in levels.items() if level in scopes
                             and any(node in binding['ancestry'] for binding in self.manifest['bindings']))
        self.placeholders = placeholder_adapters(engine, bank, self.scopes)
        self.views = {}
        for scope in self.scopes:
            size = scopes[levels[scope]] or len(bank.components)
            self.views[scope] = (select_view(bank, scope, levels[scope], size, policy='full-bank-v1') if len(bank.components) <= size
                                 else select_view(bank, scope, levels[scope], size, policy='key-similarity-v1',
                                                  key_scores=[0.] * len(bank.components)))
        items = []
        for turn in turns:
            if turn.get('schema') != TURN_SCHEMA:
                continue  # already-lowered graphs carry no demonstration record
            try:
                items.append(demonstration_facts(turn, 'support'))
            except (LearnedOptimizerError, TuneMathError):
                continue  # unverified turns are context, never demonstrations
        self.evidence = build_view(self.manifest['program']['id'], 'incoming', items)
        self.contract = json.dumps({'program': self.manifest['program'], 'bindings': [b['id'] for b in self.manifest['bindings']]})
        self.texts = {key: engine.store.get_native_state(record['state']).block.text for key, record in self.soft.items()}

    def _soft_initialization(self):
        """The build's initial soft states when a build exists (the fit's own incoming bundle), else this context's."""
        for directory in [*([self.build] if self.build else []), self.out]:
            path = directory / 'soft-init.json'
            if path.exists():
                return json.loads(path.read_text())
        if not self.plan.get('softStates'):
            return {}
        from ..soft import initialize_soft_states
        soft = initialize_soft_states(self.engine, self.turns)
        persist_artifacts(self.engine, [record['state'] for record in soft.values()], self.out / 'artifacts', self.out / 'states')
        _write(self.out / 'soft-init.json', soft)
        return soft

    # -- the optimizer's request for one incoming bundle --------------------------------------------------------
    def request(self, *, incoming_states=None, incoming_adapters=None, focus=None, digests=(), tier=None, labels=None,
                lineage=None, key_evidence=None):
        """The assembled request (a target without labels when ``labels`` is None): incoming soft states (default: the
        initial states), incoming adapters per scope (default: zero) and the bound bank views."""
        from .assemble import training_target
        states = {key: record['state'] for key, record in self.soft.items()}
        states.update(incoming_states or {})
        adapters = {scope: adapter for scope, adapter in (incoming_adapters or {}).items() if scope in self.scopes}
        for scope, adapter in adapters.items():
            self.bank.adapter_cores(self.engine, adapter)  # exact coordinates of this bank, or an error
        labels = labels or {}
        return training_target(
            self.engine, self.bank, manifest=self.manifest, rows=self.rows, contract=self.contract, view=self.evidence,
            soft_texts=self.texts, incoming_states=states, incoming_adapters=adapters, bank_views=self.views,
            target_mixtures=labels.get('mixtures'),
            target_adapters=labels.get('adapters', {}), target_states=labels.get('states', {}),
            target_factors=labels.get('factors'),
            placeholders=self.placeholders, focus=focus, digests=digests, tier=tier, program=self.program,
            lineage=lineage, key_evidence=key_evidence)

    @property
    def program(self):
        program = self.manifest['program']
        return {'id': program['id'], 'buildHash': program.get('buildHash')}

    def scope_arrays(self, incoming_adapters=None):
        from .fit import ScopeArray
        return [ScopeArray(scope, self.placeholders[scope], list(self.views[scope].slots), list(self.views[scope].available),
                           self.bank.adapter_cores(self.engine, incoming_adapters[scope])
                           if incoming_adapters and scope in incoming_adapters else None)
                for scope in self.scopes]

    def persist(self, ids):
        persist_artifacts(self.engine, [ident for ident in ids if ident], self.out / 'artifacts', self.out / 'states')


def _occurrence(turn):
    """The tuner's occurrence key of a turn (`tune.rows.turn_graph`) or of an already-lowered row."""
    if 'occurrence' in turn:
        return turn['occurrence']
    return f"{turn['root']}:{turn.get('call') or 'root'}:{turn['turn']}"


def read_build(build):
    """A finished or running `tune.build` output: its trainer state, release fragment and update boundaries."""
    build = Path(build)
    state = json.loads((build / 'state.json').read_text())
    result = json.loads((build / 'result.json').read_text()) if (build / 'result.json').exists() else None
    boundaries = [json.loads(line) for line in (build / 'boundaries.jsonl').read_text().splitlines() if line.strip()] \
        if (build / 'boundaries.jsonl').exists() else []
    return {'state': state, 'result': result, 'boundaries': boundaries}
