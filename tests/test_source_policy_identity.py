from scripts.neuralese_data.source_policy_identity import scoped_python_policy, raw_protection_policy
import pytest

BASE = '''import json
from pathlib import Path
FLAG = 1
UNRELATED = 2
def file_rows(path: Path):
    return json.loads(path.read_text())
def current_protection(path):
    return file_rows(path), FLAG
def main():
    return UNRELATED
'''


def identity(tmp_path, text):
    p = tmp_path / 'policy.py'; p.write_text(text)
    return raw_protection_policy(p)


def test_unrelated_functions_and_constants_do_not_change_law(tmp_path):
    assert identity(tmp_path, BASE) == identity(tmp_path, BASE.replace('return UNRELATED', 'return UNRELATED + 3').replace('UNRELATED = 2', 'UNRELATED = 90'))


@pytest.mark.parametrize('before,after', [('FLAG = 1','FLAG = 2'), ('import json','import other as json'), ('return json.loads(path.read_text())','return json.loads(path.read_text().upper())'), ('from pathlib import Path','from other import Path')])
def test_actual_referenced_law_changes_identity(tmp_path, before, after):
    assert identity(tmp_path, BASE) != identity(tmp_path, BASE.replace(before, after))


def test_module_mutation_is_part_of_law(tmp_path):
    assert identity(tmp_path, BASE) != identity(tmp_path, BASE + '\njson.loads = str\n')


def test_unbound_global_fails_closed(tmp_path):
    with pytest.raises(ValueError, match='unbound source policy global'):
        identity(tmp_path, BASE.replace('return file_rows(path), FLAG', 'return missing(path)'))


def test_no_fake_file_hash(tmp_path):
    v = identity(tmp_path, BASE)
    assert v['schema'] == 'natlang.scoped-python-source-policy/1'
    assert 'sha256' not in v and 'bytes' not in v
    assert len(v['policy_sha256']) == 64


def test_actual_protection_module_has_resolved_identity():
    from pathlib import Path
    source = Path(__file__).resolve().parents[1] / 'scripts/neuralese_data/build_verified_teacher_trajectory_graphs.py'
    assert len(raw_protection_policy(source)['policy_sha256']) == 64


def test_module_alias_mutation_cannot_change_value_under_same_identity(tmp_path):
    base = '''CFG = {}
def file_rows(path): return CFG
def current_protection(path): return file_rows(path)
alias = CFG
alias.update({'held': 1})
'''
    assert identity(tmp_path, base) != identity(tmp_path, base.replace("'held': 1", "'held': 2"))


def test_module_mutator_function_is_transitive(tmp_path):
    base = '''CFG = {}
def file_rows(path): return CFG
def current_protection(path): return file_rows(path)
def mutate(): CFG.update({'held': 1})
mutate()
'''
    assert identity(tmp_path, base) != identity(tmp_path, base.replace("'held': 1", "'held': 2"))


def test_conditional_builtin_shadow_fails_closed(tmp_path):
    base = '''def file_rows(path): return len(path)
def current_protection(path): return file_rows(path)
if True:
    len = lambda value: 0
'''
    with pytest.raises(ValueError, match='conditional module mutation'):
        identity(tmp_path, base)


def test_dynamic_module_namespace_fails_closed(tmp_path):
    with pytest.raises(ValueError, match='dynamic module namespace'):
        identity(tmp_path, BASE + "globals().update({'FLAG': 99})\n")


def test_transitive_effects_preserve_original_module_order(tmp_path):
    before = '''CFG = {}
a = CFG
b = a
def mutate():
    global a
    a = {}
mutate()
b['x'] = 1
def file_rows(path): return CFG
def current_protection(path): return file_rows(path)
'''
    after = before.replace('b = a\n','').replace('mutate()\nb[', 'mutate()\nb = a\nb[')
    left, right = {}, {}; exec(before, left); exec(after, right)
    assert left['current_protection'](None) == {'x': 1}
    assert right['current_protection'](None) == {}
    assert identity(tmp_path, before) != identity(tmp_path, after)
