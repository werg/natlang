import json
import urllib.error

from natlang_neuralese.rl import held_eval


class Flaky:
    def __init__(self, failures):
        self.failures, self.calls = failures, 0

    def get_bytes(self, path):
        self.calls += 1
        if self.calls <= self.failures:
            raise urllib.error.URLError(TimeoutError('timed out'))
        return json.dumps({'rl_policy_binding': {'policy_id': 'p'}}).encode()


def test_info_read_retries_connect_timeouts():
    client = Flaky(2)
    assert held_eval._read_info(client, attempts=3, wait=0)['rl_policy_binding']['policy_id'] == 'p'
    assert client.calls == 3


def test_info_read_gives_up_after_its_attempts():
    client = Flaky(5)
    try:
        held_eval._read_info(client, attempts=2, wait=0)
    except urllib.error.URLError:
        assert client.calls == 2
    else:
        raise AssertionError('expected the last error')
