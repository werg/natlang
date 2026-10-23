"""Exact Mellum/LFM parity checks for the current native production recipes."""
from __future__ import annotations

import datetime
import json
from pathlib import Path

PARITY_SCHEMA = 'natlang.neuralese-line-parity/1'
KINDS = ('fit-only', 'backbone-inherent')
LINES = ('mellum', 'lfm')
STAGE_KIND = 'native_state_sft'


def _repo():
    return Path(__file__).resolve().parents[4]


def parity_path(repo=None):
    return Path(repo or _repo()) / 'training' / 'neuralese' / 'line-parity.json'


def load_parity(repo=None):
    parity = json.loads(parity_path(repo).read_text())
    if (parity.get('schema') != PARITY_SCHEMA or set(parity.get('stages', {})) != {STAGE_KIND}
            or not isinstance(parity['stages'][STAGE_KIND].get('lines'), dict)):
        raise ValueError('line-parity.json must declare the single current native_state_sft stage')
    for entry in parity.get('allowed_differences', []):
        if (entry.get('stage') != STAGE_KIND or entry.get('kind') not in KINDS or not entry.get('reason')
                or not set(entry.get('lines') or ()) <= set(LINES) or not entry.get('lines')
                or not isinstance(entry.get('key'), str) or not entry['key']):
            raise ValueError('native allowed difference requires stage, lines, key, kind and reason')
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
        raise ValueError('line parity names an unknown native recipe: ' + recipe_id)
    return found[0]


def _stage_view(recipe):
    from .recipe import NATIVE_TRAINING_KINDS
    stages = [s for s in recipe.get('stages', []) if s.get('admitted') is not False]
    if not stages or any(s.get('kind') != STAGE_KIND or s.get('kind') not in NATIVE_TRAINING_KINDS for s in stages):
        raise ValueError('line parity accepts current native SFT stages only')
    catalog = recipe.get('input_bindings', {})
    flat = {}
    for stage in stages:
        parameters = {**recipe.get('stage_parameter_defaults', {}).get(stage['kind'], {}),
                     **stage.get('parameters', {})}
        _flatten('parameters', parameters, flat)
        for role, name in stage.get('inputs', {}).items():
            binding = catalog.get(name, {})
            flat['inputs.' + role] = binding.get('sha256', binding.get('artifact'))
    for key, value in recipe.items():
        if key not in {'schema', 'id', 'description', 'extends', 'overrides', 'stages',
                       'input_bindings', 'stage_parameter_defaults'}:
            _flatten(key, value, flat)
    return STAGE_KIND, flat


def effective_config(spec, repo=None):
    """Resolve a native recipe ID and return its exact flattened shared settings."""
    from .recipe import resolve_recipe_data
    repo = Path(repo or _repo())
    if 'recipe' not in spec:
        raise ValueError('line parity requires a declared native recipe ID')
    return _stage_view(resolve_recipe_data(_recipe_file(spec['recipe'], repo)))


def _native_source_twin(base_sha, line_sha, repo):
    """Confirm tokenizer-specific graph files share the admitted full-turn source and split contract."""
    registry = json.loads((Path(repo) / 'training' / 'neuralese_corpora.json').read_text())
    corpora = registry.get('corpora', registry if isinstance(registry, list) else [])
    def digests(value):
        if isinstance(value, dict):
            if isinstance(value.get('sha256'), str):
                yield value['sha256']
            for child in value.values():
                yield from digests(child)
        elif isinstance(value, list):
            for child in value:
                yield from digests(child)
    return any(isinstance(corpus, dict)
        and corpus.get('source_class') == 'static-role-aware-native-state-graph-conversion-of-admitted-full-turn-corpus'
        and corpus.get('derived_from')
        and 'Exact source train/test inheritance' in corpus.get('split', '')
        and base_sha in set(digests(corpus.get('renders')))
        and line_sha in set(digests(corpus.get('renders')))
        for corpus in corpora)


def _allowed(parity, line, key):
    return next((entry for entry in parity.get('allowed_differences', [])
                 if line in entry['lines'] and (key == entry['key'] or key.startswith(entry['key'] + '.'))), None)


def compare(parity, line, config, *, repo=None, other=None):
    """Compare one line against the shared native base, rejecting unexplained differences."""
    repo = Path(repo or _repo())
    base_id = parity['stages'][STAGE_KIND]['base']['recipe']
    _, base = effective_config({'recipe': base_id}, repo)
    errors, used = [], set()
    for key in sorted(set(base) | set(config)):
        before, actual = base.get(key, '<absent>'), config.get(key, '<absent>')
        if before == actual:
            continue
        entry = _allowed(parity, line, key)
        if entry is None:
            paired = f"; Mellum={other.get(key, '<absent>')}; LFM={config.get(key, '<absent>')}" if other else ''
            errors.append(f'{key} differs from the shared native recipe{paired}; declare a justified native line difference')
            continue
        used.add(id(entry))
        if entry.get('check') == 'native_source_twin' and not _native_source_twin(before, actual, repo):
            errors.append(f'{key} must retain the shared admitted source membership and exact train/held split')
    return errors, used


def check(repo=None, today=None):
    """Check both declared current production recipes against the shared native SFT base."""
    repo = Path(repo or _repo())
    parity = load_parity(repo)
    lines = parity['stages'][STAGE_KIND]['lines']
    if set(lines) != set(LINES):
        return ['native_state_sft must declare both Mellum and LFM']
    configs = {}
    for line in LINES:
        spec = lines[line]
        if not isinstance(spec.get('recipe'), str) or not spec['recipe']:
            return [f'native_state_sft/{line}: declare the current recipe']
        kind, config = effective_config({'recipe': spec['recipe']}, repo)
        if kind != STAGE_KIND:
            return [f"{line}: expected native stage {STAGE_KIND}, got {kind}"]
        configs[line] = config
    errors, used, checked = [], set(), set()
    for line in LINES:
        found, seen = compare(parity, line, configs[line], repo=repo,
                              other=configs['mellum' if line == 'lfm' else 'lfm'])
        errors.extend(f"{lines[line]['recipe']}: {error}" for error in found)
        used.update(seen)
        checked.add(line)
    for entry in parity.get('allowed_differences', []):
        if id(entry) not in used and any(line in checked for line in entry['lines']):
            errors.append(f"{entry['key']} no longer differs; remove the obsolete native line exception")
    return errors
