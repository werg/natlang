"""check-main --quick: the training-data registry stays the one place that names training data.

Static checks (no data access): the git layer is well formed, the roles the trainers resolve are complete, every
producer records what it writes and the trainers name no dataset paths of their own."""
import ast
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


# Scripts that write training data: each records its output in the registry (data_registry.record_dataset /
# declare_dataset / record_files). A new producer joins this list.
PRODUCERS = ['scripts/neuralese_data/grpo.py', 'scripts/neuralese_data/sft_sources.py', 'scripts/neuralese_data/teacher_gen.py',
             'scripts/neuralese_data/operator_rows.py', 'scripts/rl_tasks/assemble.py',
             'scripts/rl_tasks/assemble_selection_pool.py',
             'scripts/rl_tasks/neuralese_variants.py', 'training/neuralese/natlang_neuralese/rl/memetic_tasks.py',
             'training/neuralese/natlang_neuralese/rl/hindsight.py', 'training/neuralese/natlang_neuralese/rl/held_eval.py',
             'scripts/neuralese_data/lower_rollout_turns.py']
# Scripts that read training data for a run: each records the read (data_registry.consume) and so refuses
# unregistered data.
CONSUMERS = ['scripts/neuralese_data/grpo.py', 'scripts/neuralese_data/teacher_gen.py', 'scripts/neuralese_data/exp0_sft.py']
# The trainers' default data comes from registry roles, never from paths in their code.
TRAINERS = ['scripts/neuralese_data/grpo.py', 'scripts/neuralese_data/teacher_gen.py', 'scripts/neuralese_data/sft_sources.py',
            'training/neuralese/natlang_neuralese/rl/hindsight.py', 'training/neuralese/natlang_neuralese/rl/memetic_tasks.py']
DATASET_MARKERS = ('cases.jsonl', 'held-v', 'task-supply', 'memetic-tasks-', 'sft-teacher-demos-', 'sft-sources-')


def entries():
    sys.path.insert(0, str(REPO / 'training/neuralese'))
    from natlang_neuralese.common import corpus_registry
    return corpus_registry.load(REPO)['corpora']


def test_registry_ids_are_unique_path_components():
    ids = [e['id'] for e in entries()]
    assert len(ids) == len(set(ids)), 'duplicate registry ids'
    assert all(i and '/' not in i and i not in ('.', '..') for i in ids)


def test_roles_are_complete_and_ordered():
    roles = {}
    for e in entries():
        for role, spec in (e.get('roles') or {}).items():
            assert isinstance(spec, dict) and 'order' in spec, (e['id'], role)
            assert e.get('kind') and e.get('license'), f"{e['id']}: a role's dataset records its kind and license"
            assert not e.get('retired'), f"{e['id']}: a retired dataset keeps no role"
            roles.setdefault(role, []).append(spec['order'])
    for role in ('grpo-pool', 'grpo-memetic-tasks', 'grpo-sd-teacher', 'held-eval'):
        assert roles.get(role), f'no dataset has the role {role}'
        assert len(roles[role]) == len(set(roles[role])), f'{role}: two datasets share an order'
    assert len(roles['held-eval']) == 1 and len(roles['grpo-memetic-tasks']) == 1


def test_held_role_is_protected():
    held = [e for e in entries() if 'held-eval' in (e.get('roles') or {})]
    assert all(e.get('protected') and str(e.get('admission', '')).startswith('evaluation-only') for e in held)
    pools = [e for e in entries() if 'grpo-pool' in (e.get('roles') or {})]
    assert not any(e.get('protected') for e in pools)


def test_root_qualified_paths_name_known_roots():
    from importlib import util
    spec = util.spec_from_file_location('paths', REPO / 'training/neuralese/natlang_neuralese/common/paths.py')
    paths = util.module_from_spec(spec)
    spec.loader.exec_module(paths)
    for e in entries():
        head, sep, _ = e['path'].partition(':')
        assert not e['path'].startswith('/'), f"{e['id']}: registry paths are portable"
        if sep:
            assert head in paths.DEFAULTS, f"{e['id']}: unknown root {head}"


def _calls(path, names):
    tree = ast.parse((REPO / path).read_text())
    return {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr in names}


def test_producers_record_what_they_write():
    missing = [p for p in PRODUCERS if not _calls(p, {'record_dataset', 'declare_dataset', 'record_files'})]
    assert not missing, f'producers that do not record their output in the data registry: {missing}'


def test_consumers_record_what_they_read():
    missing = [p for p in CONSUMERS if not _calls(p, {'consume'})]
    assert not missing, f'consumers that do not record (and check) their inputs: {missing}'


def test_trainers_name_no_dataset_paths():
    offenders = []
    for path in TRAINERS:
        tree = ast.parse((REPO / path).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, 'id', getattr(node.func, 'attr', None)) in ('resolve_str', 'resolve'):
                literal = ' '.join(a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str))
                if any(m in literal for m in DATASET_MARKERS):
                    offenders.append(f'{path}:{node.lineno} {literal}')
    assert not offenders, 'training data named in code instead of a registry role: ' + '; '.join(offenders)


def test_registry_covers_what_the_catalog_found_before_it(monkeypatch):
    """Every dataset the ops catalog listed before the registry replaced its hand list is held by a registered dataset
    (an archive inventory does not count). Checked where the data lives (the DGX); elsewhere the paths are absent."""
    import os
    import sys
    sys.path[:0] = [str(REPO / 'scripts'), str(REPO / 'training/neuralese')]
    import data_registry as cli
    from natlang_neuralese.common import data_registry as dr
    paths = [p for p in cli.coverage_paths() if os.path.exists(p)]
    if not paths:
        import pytest
        pytest.skip('the baseline datasets are not on this machine')
    from natlang_neuralese.common.paths import root
    monkeypatch.delenv(dr.ALLOW_ENV, raising=False)
    monkeypatch.setenv('NATLANG_DATA_REGISTRY', str(root('data_nvme') / 'data-registry'))   # the machine's journal
    reg = dr.load(fresh=True)
    missing = [p for p in paths if (reg.find(p) or {'admission': dr.INVENTORY}).get('admission') == dr.INVENTORY]
    assert not missing, f'{len(missing)} datasets the catalog knew are not registered: {missing[:20]}'


def test_registry_is_one_file_per_entry():
    """The git layer stays split: no single registry document (every change would copy all of it into history), each
    entry file names its own id, and no entry file is huge (long file lists live in the manifests: include_manifest)."""
    assert not (REPO / 'training/neuralese_corpora.json').exists()
    directory = REPO / 'training/corpora'
    big = []
    for path in directory.glob('*.json'):
        entry = json.loads(path.read_text())
        assert entry['id'] == path.stem, path
        if path.stat().st_size > 1_000_000:
            big.append(path.name)
    assert not big, f'registry entries over 1 MB (move file lists into the manifest): {big}'


def test_registry_readers_use_the_loader():
    """Nothing reads or writes the retired single file: every reader goes through common.corpus_registry."""
    import subprocess
    hits = subprocess.run(['git', 'grep', '-l', 'neuralese_corpora.json', '--', '*.py', '*.mjs', '*.ts', '*.sh'],
                          cwd=REPO, capture_output=True, text=True).stdout.split()
    allowed = {'training/neuralese/natlang_neuralese/common/corpus_registry.py', 'tests/test_data_registry_ratchet.py'}
    assert not set(hits) - allowed, f'readers of the retired registry file: {sorted(set(hits) - allowed)}'
