import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

spec = importlib.util.spec_from_file_location('reclaim', Path(__file__).parents[1] / 'scripts/reclaim_file_cache.py')
reclaim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reclaim)


class CacheReadinessTests(unittest.TestCase):
    def run_cleanup(self, ready=None, error=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weight = root / 'weight.safetensors'
            weight.write_bytes(b'weights')
            receipt = root / 'receipt.json'
            argv = ['reclaim', '--root', str(weight), '--receipt', str(receipt), '--apply', '--min-file-age-seconds', '0']
            if ready is not None or error is not None:
                argv += ['--ready-url', 'http://127.0.0.1:8082/health']
            response = MagicMock()
            response.__enter__.return_value.status = ready
            with patch('sys.argv', argv), patch.object(reclaim.urllib.request, 'urlopen', return_value=response, side_effect=error) as probe, patch.object(reclaim.os, 'posix_fadvise') as advise, contextlib.redirect_stdout(io.StringIO()):
                reclaim.main()
                return json.loads(receipt.read_text()), advise.call_count, probe.call_count

    def test_ready_model_allows_advice(self):
        report, calls, probes = self.run_cleanup(200)
        self.assertEqual((report['status'], calls, probes), ('advised', 1, 1))

    def test_unready_response_preserves_weight_cache(self):
        report, calls, _ = self.run_cleanup(503)
        self.assertEqual((report['status'], calls, report['files']), ('model_not_ready', 0, 0))

    def test_connection_failure_preserves_weight_cache_and_receipt(self):
        report, calls, _ = self.run_cleanup(error=ConnectionRefusedError('loading'))
        self.assertEqual((report['status'], calls), ('model_not_ready', 0))
        self.assertIn('loading', report['readiness_error'])

    def test_data_cleanup_does_not_require_model(self):
        report, calls, probes = self.run_cleanup()
        self.assertEqual((report['status'], calls, probes), ('advised', 1, 0))


if __name__ == '__main__':
    unittest.main()
