from scripts.neuralese_data import acquired_agents, agents


def test_complete_session_export_refs_resolve_through_shared_corpora():
    expected = {
        'D13': 'team-cooperation-acquired',
        'D14': 'qwen-cooperation-acquired',
        'D17': 'cooperdata-acquired',
        'D19': 'agent-collusion-raw-acquired',
        'D20': 'swe-chat-acquired',
        'D22': 'open-swe-acquired',
        'D23': 'swe-zero-acquired',
        'D24': 'miroverse-acquired',
        'D25': 'swe-rebench-acquired',
        'D26': 'toucan-acquired',
        'D27': 'toolmind-acquired',
        'D28': 'nemotron-agentic-acquired',
        'D32': 'openresearcher-structured-acquired',
        'D35': 'nemotron-terminal-acquired',
        'D36': 'coderforge-acquired',
        'D37': 'orchard-acquired',
        'D38': 'ultradata-acquired',
    }
    assert acquired_agents.FULL_TRAJECTORY_CORPORA == expected
    for ref, name in expected.items():
        resolved_name, entry = acquired_agents.full_trajectory_corpus(ref)
        assert resolved_name == name
        assert entry is agents.CORPORA[name]


def test_unmapped_ref_fails_with_a_source_specific_reason():
    from scripts.neuralese_data.common import Reject

    try:
        acquired_agents.full_trajectory_corpus('D39')
    except Reject as error:
        assert 'D39' in str(error)
    else:
        raise AssertionError('unmapped source ref must not silently pick an adapter')
