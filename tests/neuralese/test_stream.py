import json
import os

import pytest
import torch

from natlang_neuralese.data.fixtures import synthetic_records, write_jsonl
from natlang_neuralese.data.render import Renderer
from natlang_neuralese.data.stream import PortRecordStream


@pytest.fixture(scope="module")
def renderer(loaded):
    _, tokenizer, backbone = loaded
    return Renderer(tokenizer, backbone.controls)


def _families(tmp_path):
    rows = synthetic_records(24, heldout_every=1000)
    for i, row in enumerate(rows):
        row["split_groups"] = [f"group:{i % 4}"]  # four groups of six records
    other = [{**row, "id": row["id"] + ":b", "family": "fixture_b", "split_groups": [f"b:{i}"]}
             for i, row in enumerate(synthetic_records(6, seed=1, heldout_every=1000))]
    return {"a": write_jsonl(rows, tmp_path / "a.jsonl"), "b": write_jsonl(other, tmp_path / "b.jsonl")}


def _stream(paths, renderer, tmp_path, **options):
    return PortRecordStream(paths, renderer, index_dir=tmp_path / "index", seed=7, **options)


def test_stream_is_deterministic_group_balanced_and_weighted(renderer, tmp_path):
    paths = _families(tmp_path)
    first = [r.record_id for r in _stream(paths, renderer, tmp_path, weights={"a": 2, "b": 1}).take(12)]
    again = [r.record_id for r in _stream(paths, renderer, tmp_path, weights={"a": 2, "b": 1}).take(12)]
    assert first == again
    from_a = [i for i in first if not i.endswith(":b")]
    assert len(from_a) == 8, "weights 2:1 interleave deterministically"
    # One record per group per epoch: the first four from family a come from four different groups.
    groups = {int(i.rsplit(":", 1)[1]) % 4 for i in from_a[:4]}
    assert groups == {0, 1, 2, 3}


def test_stream_resumes_at_the_exact_next_record(renderer, tmp_path):
    paths = _families(tmp_path)
    stream = _stream(paths, renderer, tmp_path)
    stream.take(7)
    state = json.loads(json.dumps(stream.state_dict()))
    expected = [r.record_id for r in stream.take(9)]
    resumed = _stream(paths, renderer, tmp_path)
    resumed.load_state_dict(state)
    assert [r.record_id for r in resumed.take(9)] == expected
    with pytest.raises(ValueError, match="seed"):
        PortRecordStream(paths, renderer, index_dir=tmp_path / "index", seed=8).load_state_dict(state)


def test_oversized_records_are_held_not_truncated(renderer, tmp_path):
    paths = _families(tmp_path)
    log = tmp_path / "held.jsonl"
    stream = _stream(paths, renderer, tmp_path, max_target_tokens=0, held_log=log)
    with pytest.raises(ValueError, match="length bounds"):
        next(stream)
    assert sum(stream.held.values()) > 0 and log.read_text().count("\n") == sum(stream.held.values())


def test_a_changed_file_is_refused(renderer, tmp_path):
    paths = _families(tmp_path)
    stream = _stream(paths, renderer, tmp_path)
    stream.take(1)
    with open(paths["a"], "a") as handle:
        handle.write("\n")
    os.utime(paths["a"])
    with pytest.raises(ValueError, match="changed"):
        stream.take(30)


def test_trainer_checkpoints_the_stream_position(loaded, renderer, tmp_path):
    from natlang_neuralese.model.heads import PortHeads
    from natlang_neuralese.train.phases import Phase
    from natlang_neuralese.train.trainer import Trainer

    _, _, backbone = loaded
    torch.manual_seed(1)
    heads = PortHeads(backbone, cutoff=6, max_length=4)
    paths = _families(tmp_path)
    rows_before = backbone.control_rows.detach().clone()
    phases = [Phase("D", 1, batch_size=1, max_length=4), Phase("E", 1, batch_size=1, max_length=4)]
    trainer = Trainer(backbone, heads, phases[:1], tmp_path / "run", records_train=_stream(paths, renderer, tmp_path),
                      log=lambda *_: None)
    trainer.run()
    saved = torch.load(tmp_path / "run" / "checkpoint.pt", weights_only=False)
    assert sum(saved["record_stream"]["served"].values()) == 1
    resumed = Trainer(backbone, heads, phases, tmp_path / "run", records_train=_stream(paths, renderer, tmp_path),
                      log=lambda *_: None)
    assert sum(resumed.record_stream.served.values()) == 1
    with pytest.raises(ValueError, match="stream"):
        Trainer(backbone, heads, phases, tmp_path / "run", records_train=[], log=lambda *_: None)
    with torch.no_grad():
        backbone.control_rows.copy_(rows_before)
