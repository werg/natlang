import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.acquire_parquet_corpus import acquire_parquet, matching_parquet_files


class FakeBatch:
    def __init__(self, rows):
        self.rows = rows

    def to_pylist(self):
        return list(self.rows)


class FakeParquetFile:
    rows_by_path = {}

    def __init__(self, path):
        self.path = str(path)
        self.rows = self.rows_by_path[self.path]
        self.metadata = SimpleNamespace(num_rows=len(self.rows))

    def iter_batches(self, *, batch_size):
        for start in range(0, len(self.rows), batch_size):
            yield FakeBatch(self.rows[start:start + batch_size])


class FakeParquet:
    ParquetFile = FakeParquetFile


class FakeApi:
    sha = "a" * 40
    files = ["javascript/train/0000.parquet", "python/train/0000.parquet",
             "javascript/test/0000.parquet", "javascript/README.md"]

    def dataset_info(self, dataset, *, revision, files_metadata):
        assert dataset == "org/dataset"
        assert revision == "refs/convert/parquet"
        assert files_metadata is True
        return SimpleNamespace(sha=self.sha, siblings=[SimpleNamespace(
            rfilename=self.files[0], size=1234)])

    def list_repo_files(self, *, repo_id, repo_type, revision):
        assert repo_id == "org/dataset"
        assert repo_type == "dataset"
        assert revision == self.sha
        return self.files


def setup_fake(tmp_path, rows):
    parquet_path = tmp_path / "cached.parquet"
    parquet_path.write_bytes(b"fake parquet bytes")
    FakeParquetFile.rows_by_path[str(parquet_path.resolve())] = rows

    def download(**kwargs):
        assert kwargs["repo_id"] == "org/dataset"
        assert kwargs["filename"] == "javascript/train/0000.parquet"
        assert kwargs["revision"] == FakeApi.sha
        assert kwargs.get("token") in (None, "test-secret")
        return parquet_path

    return FakeApi(), download, FakeParquet


def test_file_selection_is_config_and_split_specific():
    files = ["javascript/train/00000.parquet", "javascript/test/00000.parquet",
             "python/train-00000.parquet", "javascript/valid.parquet", "train-00000.parquet"]
    assert matching_parquet_files(files, config="javascript", split="train") == [files[0]]
    assert matching_parquet_files(files, config="default", split="train") == [files[4]]


def test_acquisition_pins_revision_writes_shard_metadata_and_sanitizes_values(tmp_path):
    rows = [{"id": 1, "code": "const x = 1", "bytes": b"raw", "score": float("nan")},
            {"id": 2, "code": "const y = 2", "score": 0.5}]
    api, download, parquet = setup_fake(tmp_path, rows)
    out = tmp_path / "acquired"
    manifest = acquire_parquet(out=out, source="codesearchnet", dataset="org/dataset", config="javascript",
        limit=10, batch_rows=1, token="test-secret", api=api, downloader=download, parquet_module=parquet)
    assert manifest["status"] == "complete"
    assert manifest["source"] == "codesearchnet"
    assert manifest["dataset"] == "org/dataset"
    assert manifest["revision"] == FakeApi.sha
    assert [shard["rows"] for shard in manifest["shards"]] == [1, 1]
    assert [shard["source_row_start"] for shard in manifest["shards"]] == [0, 1]
    assert all(shard["path"] and shard["sha256"] for shard in manifest["shards"])
    first = json.loads((out / manifest["shards"][0]["path"]).read_text())
    assert first["bytes"] == {"__bytes_base64__": "cmF3"}
    assert first["score"] is None
    saved = (out / "manifest.json").read_text()
    assert "test-secret" not in saved


def test_sigterm_style_stop_commits_batch_then_resume_continues_next_row(tmp_path):
    rows = [{"id": i, "code": f"const n{i} = {i}"} for i in range(5)]
    api, download, parquet = setup_fake(tmp_path, rows)
    out = tmp_path / "resume"
    stopping = False
    def stop_after_first_shard():
        return (out / "part-00000000.jsonl").exists()
    paused = acquire_parquet(out=out, source="codesearchnet", dataset="org/dataset", config="javascript",
        limit=5, batch_rows=2, api=api, downloader=download, parquet_module=parquet, should_stop=stop_after_first_shard)
    assert paused["status"] == "paused"
    assert paused["rows_committed"] == 2
    assert paused["next_row_in_file"] == 2
    resumed = acquire_parquet(out=out, source="codesearchnet", dataset="org/dataset", config="javascript",
        limit=5, batch_rows=2, api=api, downloader=download, parquet_module=parquet)
    assert resumed["status"] == "complete"
    assert resumed["rows_committed"] == 5
    ids = [json.loads(line)["id"] for shard in resumed["shards"]
           for line in (out / shard["path"]).read_text().splitlines()]
    assert ids == list(range(5))


def test_increased_limit_can_resume_mid_batch_and_corruption_blocks_resume(tmp_path):
    rows = [{"id": i} for i in range(5)]
    api, download, parquet = setup_fake(tmp_path, rows)
    out = tmp_path / "increase"
    first = acquire_parquet(out=out, source="codesearchnet", dataset="org/dataset", config="javascript",
        limit=3, batch_rows=2, api=api, downloader=download, parquet_module=parquet)
    assert first["status"] == "complete"
    assert first["next_row_in_file"] == 3
    resumed = acquire_parquet(out=out, source="codesearchnet", dataset="org/dataset", config="javascript",
        limit=5, batch_rows=2, api=api, downloader=download, parquet_module=parquet)
    assert resumed["rows_committed"] == 5
    path = out / resumed["shards"][0]["path"]
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="hash mismatch"):
        acquire_parquet(out=out, source="codesearchnet", dataset="org/dataset", config="javascript",
            limit=5, batch_rows=2, api=api, downloader=download, parquet_module=parquet)


def test_token_file_mode_is_checked_without_recording_secret(tmp_path):
    from scripts.acquire_parquet_corpus import read_token_file
    token_file = tmp_path / "token"
    token_file.write_text("secret-value\n")
    token_file.chmod(0o600)
    assert read_token_file(token_file) == "secret-value"
    token_file.chmod(0o644)
    with pytest.raises(ValueError, match="permissions"):
        read_token_file(token_file)
