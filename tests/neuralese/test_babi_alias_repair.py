"""bAbI golds the instruction does not determine: task 15 plural species, task 19 spelled-out directions (task_packet)."""
import pytest

from natlang_neuralese.eval import task_packet


def result(family, expected, value, checks=None):
    return {'task': {'program_ir': {'source': 'external-rl:babi', 'family': family,
                                    'source_ids': ['facebook/babi_qa:train-15-1036:f3dd246b'],
                                    'semantics': {'oracle': 'normalized', 'expected': expected}}},
            'outcome': {'status': 'done', 'accepted': False, 'rejection_reasons': ['answer'],
                        'checks': checks or {'answer': False, 'status': True}, 'value': value}}


FINISH = {'status': 'complete', 'resource_limit_reason': None}


@pytest.mark.parametrize('family,expected,value,ok', [
    ('ext_babi_task15', 'wolf', 'wolves', True), ('ext_babi_task15', 'mouse', 'Mice', True),
    ('ext_babi_task15', 'wolf', 'mice', False), ('ext_babi_task15', 'cat', 'cat s', False),
    ('ext_babi_task19', 's,e', 'south, east', True), ('ext_babi_task19', 's,e', 'south then east', True),
    ('ext_babi_task19', 'n,w', 'north,west', True), ('ext_babi_task19', 's,e', 'east, south', False),
    ('ext_babi_task19', 's,e', 'south', False), ('ext_babi_task19', 's,e', 'south, east, north', False),
    ('ext_babi_task01', 'kitchen', 'the kitchen', True), ('ext_babi_task01', 'wolf', 'wolves', False),
])
def test_proved_babi_aliases(monkeypatch, family, expected, value, ok):
    monkeypatch.delenv('NATLANG_BABI_ALIAS_REPAIR', raising=False)
    assert task_packet.accepted_babi_article_answer(result(family, expected, value), FINISH) is ok
    assert (task_packet.classify_case(result(family, expected, value), FINISH) == 'task_success') is ok


def test_repair_never_waives_another_failed_check_and_opts_out(monkeypatch):
    failed_world = result('ext_babi_task15', 'wolf', 'wolves', {'answer': False, 'status': False})
    assert not task_packet.accepted_babi_article_answer(failed_world, FINISH)
    assert task_packet.classification_binding()['version'] == 'host-evidence-terminal-babi-alias/4'
    monkeypatch.setenv('NATLANG_BABI_ALIAS_REPAIR', '0')
    assert not task_packet.accepted_babi_article_answer(result('ext_babi_task15', 'wolf', 'wolves'), FINISH)
    assert task_packet.accepted_babi_article_answer(result('ext_babi_task01', 'kitchen', 'the kitchen'), FINISH)
    assert task_packet.classification_binding()['version'] == 'host-evidence-terminal-babi-article/3'
