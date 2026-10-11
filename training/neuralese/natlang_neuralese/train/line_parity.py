"""Line parity: the Mellum (DGX) and LFM (Pop) lines run each Neuralese stage with the same data and settings.

Owner, 2026-10-11: one recipe, data and code for both lines; only declared differences, each either fit-only
(memory/hardware/time) or backbone-inherent, with its reason (plans/neuralese/LINE_PARITY.md). This is an exact
checker, crisp by design: it resolves the effective configuration of each line's declared recipe for a stage and
compares it with the stage's shared base, failing on any difference that ``training/neuralese/line-parity.json``
does not list. It reads only recipe JSON and the corpus registry (no data files), so it runs in check-main --quick.

Effective configuration, flattened to dotted keys:
- ``parameters.*``: the stage's resolved parameters (recipe defaults merged with the stage's own);
- ``inputs.<role>``: each data input's sha256 (identity, not location);
- ``quantization.*``: the quantization component, if any (without its description);
- ``derived.supervision``: the amount of supervision of a neuralese_direct stage, which must stay equal across lines:
  its ``epochs`` over the target-bearing windows of its (twin) data (the update count is derived from the data and
  each line's fit-only packing), or for a legacy steps-budgeted stage steps x target_tokens x batch.
- A recipe of several stages (phases of one lineage, all of one kind) prefixes every key with ``stages.<id>.``;
  declared differences and lineage keys name the unprefixed key and apply to every stage.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

PARITY_SCHEMA = 'natlang.neuralese-line-parity/1'
KINDS = ('fit-only', 'backbone-inherent')
LINES = ('mellum', 'lfm')


def _repo():
    return Path(__file__).resolve().parents[4]


def parity_path(repo=None):
    return Path(repo or _repo()) / 'training' / 'neuralese' / 'line-parity.json'


def load_parity(repo=None):
    parity = json.loads(parity_path(repo).read_text())
    if parity.get('schema') != PARITY_SCHEMA or not isinstance(parity.get('stages'), dict):
        raise ValueError('line-parity.json needs schema ' + PARITY_SCHEMA + ' and stages')
    for entry in parity.get('allowed_differences', []):
        if (set(entry) - {'stage', 'lines', 'key', 'kind', 'reason', 'check'} or entry.get('kind') not in KINDS
                or entry.get('stage') not in parity['stages'] or not entry.get('reason')
                or not set(entry.get('lines') or ()) or set(entry['lines']) - set(LINES)):
            raise ValueError('an allowed difference is {stage, lines, key, kind (fit-only|backbone-inherent), '
                             'reason, check?}: ' + json.dumps(entry))
    return parity


def _flatten(prefix, value, out):
    if isinstance(value, dict) and value:
        for key, item in value.items():
            _flatten(prefix + '.' + key, item, out)
    else:
        out[prefix] = value


def _recipe_file(recipe_id, repo):
    directory = Path(repo) / 'training' / 'neuralese' / 'recipes'
    found = [p for p in sorted(directory.glob('*.json')) if json.loads(p.read_text()).get('id') == recipe_id]
    if len(found) != 1:
        raise ValueError('line parity names an unknown recipe: ' + recipe_id)
    return found[0]


def _stage_view(kind, parameters, inputs, quantization=None):
    out = {}
    _flatten('parameters', parameters, out)
    for role, binding in inputs.items():
        out['inputs.' + role] = binding.get('sha256', binding.get('artifact'))
    if quantization:
        _flatten('quantization', {k: v for k, v in quantization.items() if k != 'description'}, out)
    if kind == 'neuralese_direct':
        out['derived.supervision'] = (f"{parameters['epochs']} epochs" if 'epochs' in parameters else
                                      f"{parameters.get('steps', 0) * parameters.get('target_tokens', 0) * parameters.get('batch', 1)} positions")
    return out


STAGE_PREFIX = re.compile(r'^stages\.[^.]+\.')


def effective_config(spec, repo=None):
    """``spec`` is {"recipe": id} (a multi-stage or a direct-stage recipe) or {"config": {kind, inputs, parameters}};
    returns (kind, flat configuration)."""
    from .recipe import DIRECT_STAGE_SCHEMA, resolve_recipe_data
    repo = Path(repo or _repo())
    if 'config' in spec:
        config = spec['config']
        return config['kind'], _stage_view(config['kind'], config.get('parameters', {}), config.get('inputs', {}))
    recipe = resolve_recipe_data(_recipe_file(spec['recipe'], repo))
    return recipe_config(recipe)


def recipe_config(recipe):
    from .recipe import DIRECT_STAGE_SCHEMA
    if recipe.get('schema') == DIRECT_STAGE_SCHEMA:
        kind = recipe['stage']['kind']
        return kind, _stage_view(kind, recipe.get('parameters', {}), recipe.get('inputs', {}))
    stages = [s for s in recipe['stages'] if s.get('admitted', True) is not False]
    if not stages or len({s['kind'] for s in stages}) != 1:
        raise ValueError('line parity compares recipes whose stages share one kind: ' + recipe['id'])
    catalog = recipe.get('input_bindings', {})
    views = {}
    for stage in stages:
        parameters = {**recipe.get('stage_parameter_defaults', {}).get(stage['kind'], {}), **stage.get('parameters', {})}
        inputs = {role: catalog[name] for role, name in stage.get('inputs', {}).items()}
        views[stage['id']] = _stage_view(stage['kind'], parameters, inputs, recipe.get('quantization'))
    if len(views) == 1:
        return stages[0]['kind'], next(iter(views.values()))
    return stages[0]['kind'], {f'stages.{sid}.{key}': value for sid, view in views.items() for key, value in view.items()}


def _twin_registered(base_sha, line_sha, repo):
    registry = json.loads((Path(repo) / 'training' / 'neuralese_corpora.json').read_text())
    for corpus in registry.get('corpora', registry if isinstance(registry, list) else []):
        twin = corpus.get('tokenizer_twin_of') if isinstance(corpus, dict) else None
        if twin and twin.get('sha256') == base_sha and twin.get('twin_sha256') == line_sha:
            return True
    return False


def _allowed(parity, stage, line, key):
    key = STAGE_PREFIX.sub('', key)
    for entry in parity.get('allowed_differences', []):
        if entry['stage'] == stage and line in entry['lines'] and (key == entry['key'] or
                                                                  key.startswith(entry['key'] + '.')):
            return entry
    return None


def _lineage(parity, stage, key):
    key = STAGE_PREFIX.sub('', key)
    return any(key == k or key.startswith(k + '.') for k in parity.get('lineage_keys', {}).get(stage, []))


def compare(parity, stage, line, kind, config, repo=None):
    """Errors (one sentence each) for ``line``'s configuration of ``stage`` against the shared base."""
    repo = Path(repo or _repo())
    base_kind, base = effective_config(parity['stages'][stage]['base'], repo)
    errors, used = [], set()
    if kind != base_kind:
        return [f'{stage}/{line}: stage kind {kind} differs from the shared base\'s {base_kind}'], set()
    for key in sorted(set(base) | set(config)):
        if base.get(key, '<absent>') == config.get(key, '<absent>') or _lineage(parity, stage, key):
            continue
        entry = _allowed(parity, stage, line, key)
        if entry is None:
            errors.append(f'{stage}/{line}: {key} is {json.dumps(config.get(key, "<absent>"))} but the shared base '
                          f'has {json.dumps(base.get(key, "<absent>"))}; move it into the base or declare it in '
                          f'line-parity.json')
            continue
        used.add(id(entry))
        if entry.get('check') == 'tokenizer_twin' and not _twin_registered(base.get(key), config.get(key), repo):
            errors.append(f'{stage}/{line}: {key} must be a registered tokenizer twin (tokenizer_twin_of) of the '
                          f'shared base\'s input {base.get(key)}')
    return errors, used


def check(repo=None):
    """All parity errors for the declared lines and every direct-stage recipe that declares a parity line."""
    from .recipe import DIRECT_STAGE_SCHEMA
    repo = Path(repo or _repo())
    parity = load_parity(repo)
    errors, used, checked = [], set(), set()
    for stage, declaration in parity['stages'].items():
        for line, spec in declaration.get('lines', {}).items():
            if line not in LINES:
                errors.append(f'{stage}: unknown line {line}')
                continue
            if not spec.get('recipe'):
                if not spec.get('pending'):
                    errors.append(f'{stage}/{line}: declare the line\'s recipe or why it is pending')
                continue
            kind, config = effective_config(spec, repo)
            found, seen = compare(parity, stage, line, kind, config, repo)
            errors += found
            used |= seen
            checked.add((stage, line))
    for path in sorted((repo / 'training' / 'neuralese' / 'recipes').glob('*.json')):
        recipe = json.loads(path.read_text())
        declared = recipe.get('parity')
        if recipe.get('schema') != DIRECT_STAGE_SCHEMA or not isinstance(declared, dict):
            continue
        if declared.get('stage') not in parity['stages'] or declared.get('line') not in LINES:
            errors.append(f'{path.name}: parity must name a stage of line-parity.json and a line (mellum|lfm)')
            continue
        kind, config = recipe_config(recipe)
        found, seen = compare(parity, declared['stage'], declared['line'], kind, config, repo)
        errors += [path.name + ': ' + e for e in found]
        used |= seen
        checked.add((declared['stage'], declared['line']))
    for entry in parity.get('allowed_differences', []):
        if id(entry) not in used and any((entry['stage'], line) in checked for line in entry['lines']):
            errors.append(f'{entry["stage"]}: declared difference {entry["key"]} ({", ".join(entry["lines"])}) no '
                          f'longer differs; remove it from line-parity.json')
    return errors


def check_declared_stage(recipe, repo=None):
    """For the direct-stage runner: refuse a fresh launch whose parity declaration does not hold (one sentence)."""
    declared = recipe.get('parity')
    if declared == 'diagnostic':
        return
    if not isinstance(declared, dict):
        raise ValueError('declare "parity": {"line": "mellum"|"lfm", "stage": "direct"|"e2e"} (or "diagnostic" for '
                         'a diagnostic run) so the launch is checked against the shared base (LINE_PARITY.md)')
    parity = load_parity(repo)
    if declared.get('stage') not in parity['stages'] or declared.get('line') not in LINES:
        raise ValueError('parity must name a stage of line-parity.json and a line (mellum|lfm)')
    kind, config = recipe_config(recipe)
    errors, _ = compare(parity, declared['stage'], declared['line'], kind, config, repo)
    if errors:
        raise ValueError(errors[0])
