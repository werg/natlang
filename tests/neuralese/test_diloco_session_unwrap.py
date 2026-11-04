from natlang_neuralese.train.diloco.session import unwrap_join


def test_unwrap_join_removes_every_earlier_session_wrapper():
    trainer = ['/py', '-m', 'natlang_neuralese.train.native_state_sft', '--diloco', '{}']
    s2 = ['/py', '/r/diloco-s2/bin/join.py', '--session', '/r/s2.json', '--worker', '1', '--', *trainer]
    s3 = ['/py', '/r/diloco-s3/bin/join.py', '--session', '/r/s3.json', '--worker', '1', '--', *s2]
    assert unwrap_join(trainer) == trainer
    assert unwrap_join(s2) == trainer
    assert unwrap_join(s3) == trainer
