"""The warm-up preflight frees page cache (the ledger's release-cache) before waiting on the GB10."""
from natlang_neuralese.train import memory_policy


def test_reclaimable_file_bytes_reads_meminfo():
    assert memory_policy.reclaimable_file_bytes() >= 0


def test_release_page_cache_runs_the_ledger_and_reports(monkeypatch, tmp_path):
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    (scripts / 'memory_ledger.py').write_text(
        'import json, sys\nassert sys.argv[1] == "release-cache"\nprint(json.dumps({"files": 3, "free_after_gb": 9.0}))\n')
    monkeypatch.setenv('NATLANG_REPO', str(tmp_path))
    assert memory_policy.release_page_cache() == {'files': 3, 'free_after_gb': 9.0}
    monkeypatch.setenv('NATLANG_REPO', str(tmp_path / 'missing'))
    assert 'skipped' in memory_policy.release_page_cache()
