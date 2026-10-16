import json
from pathlib import Path
import sys
import hashlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training" / "neuralese"))
sys.path.insert(0, str(ROOT / "scripts"))

from natlang_neuralese.data.native_graph_schema import parse_graph  # noqa: E402
from natlang_neuralese.data.port_record_graph import port_record_graph, consumer_target_authorization  # noqa: E402
from natlang_neuralese.data.records import read_record_rows  # noqa: E402
from scripts.neuralese_data.build_searchowl_port_record_graphs import load_reviewed_targets, _hold_references, load_hold_details, decision_hold_policy  # noqa: E402
from neuralese_data import bgkit  # noqa: E402
from test_neuralese_port_records import squad_row  # noqa: E402


class _Tokenizer:
    unk_token_id = -1
    special = {
        2: "<|neuralese_produce|>", 3: "<|/neuralese_produce|>",
        4: "<|neuralese_consume|>", 5: "<|/neuralese_consume|>",
        10: "<|system|>", 11: "<|user|>", 12: "<|assistant|>",
        13: "<|tool|>", 14: "<|end|>",
    }
    config = type("Config", (), {"model_type": "test"})()
    chat_template = "test"

    @property
    def all_special_tokens(self):
        return list(self.special.values())

    def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=False, **kwargs):
        pieces = []
        for message in messages:
            pieces.append(f"<|{message['role']}|>")
            if message.get("reasoning_content"):
                pieces.append("<think>" + message["reasoning_content"] + "</think>")
            content = message.get("content")
            if isinstance(content, list):
                for part in content:
                    pieces.append(part.get("text", "") if part.get("type") == "text"
                                  else f"<{part['type']}:{part['id']}>")
            elif content is not None:
                pieces.append(str(content))
            if message.get("tool_calls"):
                pieces.append(json.dumps(message["tool_calls"], sort_keys=True))
            pieces.append("<|end|>")
        if add_generation_prompt:
            pieces.append("<|assistant|>")
        return "".join(pieces)

    def __call__(self, text, *, add_special_tokens=False, split_special_tokens=False, **kwargs):
        ids, index = [], 0
        markers = sorted(self.special.items(), key=lambda pair: len(pair[1]), reverse=True)
        while index < len(text):
            match = next(((token, marker) for token, marker in markers
                          if not split_special_tokens and text.startswith(marker, index)), None)
            if match:
                ids.append(match[0])
                index += len(match[1])
            else:
                ids.append(ord(text[index]) + 1000)
                index += 1
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=False):
        reverse = {token: marker for token, marker in self.special.items()}
        return "".join(reverse[value] if value in reverse else chr(value - 1000) for value in ids)


def _source_row(tmp_path, *, label="teacher", split="train"):
    record = bgkit.convert_qa("qa_squad", 10, squad_row(split=split))
    record["family"] = "trajectory_continuation_research"
    record["sources"] = [{"role": "trajectory", "text": "user: evidence\nassistant: source", "exact_refs": []}]
    record["writer"]["result_type"] = "Neuralese<SessionHistory>"
    record["consumer"]["context"] = [{"role": "user", "content": "Question: who?"}]
    record["consumer"]["withheld"] = ["sources"]
    record["outcome"] = {"label": label, "checked": None if label == "teacher" else "exact reference"}
    record["split_groups"] = ["search-task:case-1"]
    path = tmp_path / "port-records.jsonl"
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    return next(read_record_rows([path], strict=True))


def _graph(row, admission=None):
    tokenizer = _Tokenizer()
    return port_record_graph(row, source_tokenizer=tokenizer,
        source_tokenizer_fingerprint="1" * 64, expanded_tokenizer_fingerprint="2" * 64,
        role_ids=(2, 3, 4, 5), corpus_id="search-owl-overcap", source_file_sha256="3" * 64,
        consumer_admission=admission)


def test_teacher_target_stays_context_while_source_is_a_real_consumer_import(tmp_path):
    row = _source_row(tmp_path)
    graph = _graph(row)
    parsed = parse_graph(graph, tokenizer_fingerprint="2" * 64)
    writer, consumer = graph["nodes"]
    assert writer["producing"]
    assert len(consumer["imports"]) == 1
    assert graph["conversion"]["consumer_target_positive_tokens"] == 0
    assert graph["conversion"]["consumer_target_context_tokens"] > 0
    assert graph["conversion"]["source_payload_tokens"] == len(row.record.source_text())
    assert all(weight == 0 for weight in consumer["target_weights"][-graph["conversion"]["consumer_target_context_tokens"]:])
    assert parsed.plan.token_count == len(writer["token_ids"]) + len(consumer["token_ids"])
    assert graph["conversion"]["declared_transformation"].startswith("static source-message serialization")


def test_gold_consumer_target_gets_positive_loss_and_test_split_is_preserved(tmp_path):
    graph = _graph(_source_row(tmp_path, label="gold", split="eval"))
    parse_graph(graph, tokenizer_fingerprint="2" * 64)
    consumer = graph["nodes"][1]
    count = graph["conversion"]["consumer_target_positive_tokens"]
    assert graph["split"] == "test" and count > 0
    assert all(weight > 0 for weight in consumer["target_weights"][-count:])
    assert graph["source_pairing"]["relationship"] == "static-transformation; not observed actor execution"


def test_same_external_id_with_distinct_source_rows_gets_distinct_graph_identity(tmp_path):
    first = _source_row(tmp_path, label="gold")
    one = _graph(first)
    second = _source_row(tmp_path, label="teacher")
    two = _graph(second)
    assert one["id"] != two["id"]
    assert one["conversion"]["original_id"] == two["conversion"]["original_id"]


def test_reviewed_teacher_target_requires_exact_source_and_target_bindings(tmp_path):
    row = _source_row(tmp_path, label="teacher")
    raw_sha = hashlib.sha256(row.raw_line).hexdigest()
    target_sha = hashlib.sha256(json.dumps(row.raw["target"], ensure_ascii=False,
        sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    proof = {"status": "approved", "id": row.record.id, "source_row_sha256": raw_sha,
        "target_sha256": target_sha, "decision_id": "decision-1", "proof_sha256": "4" * 64}
    assert consumer_target_authorization(row, reviewed_targets={raw_sha: proof})["kind"] == "source-reviewed-target"
    assert _graph(row, admission=proof)["conversion"]["consumer_target_positive_tokens"] > 0
    wrong_target = {**proof, "target_sha256": "5" * 64}
    try:
        _graph(row, admission=wrong_target)
    except ValueError as exc:
        assert "exact source occurrence" in str(exc)
    else:
        raise AssertionError("reviewed target admission accepted a mismatched target hash")


def test_structured_context_producer_and_tool_call_target_are_preserved(tmp_path):
    row = _source_row(tmp_path, label="gold")
    raw = dict(row.raw)
    raw["writer"] = {**raw["writer"], "context": [
        {"role": "assistant", "reasoning": "prepare the value", "content": ""}]}
    raw["consumer"] = {**raw["consumer"], "context": [
        {"role": "user", "content": "Use the supplied state."},
        {"role": "assistant", "reasoning": "I should inspect it", "content": "", "tool_calls": [
            {"id": "call-1", "type": "function", "function": {
                "name": "inspect", "arguments": {"field": "name"}}}]},
        {"role": "tool", "name": "inspect", "tool_call_id": "call-1", "content": "Ada"},
        {"role": "user", "content": "What is the name?"}],
        "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]}
    raw["target"] = {"kind": "message", "value": {
        "role": "assistant", "reasoning": "The tool returned Ada.", "content": "",
        "tool_calls": [{"id": "call-2", "type": "function", "function": {
            "name": "answer", "arguments": {"name": "Ada"}}}]}}
    path = tmp_path / "structured.jsonl"
    path.write_text(json.dumps(raw, ensure_ascii=False) + "\n", encoding="utf-8")
    graph = _graph(next(read_record_rows([path], strict=True)))
    parse_graph(graph, tokenizer_fingerprint="2" * 64)
    tokenizer = _Tokenizer()
    producer_text = tokenizer.decode(graph["nodes"][0]["token_ids"])
    consumer_text = tokenizer.decode(graph["nodes"][1]["token_ids"])
    assert "prepare the value" in producer_text
    assert "inspect" in consumer_text and '"field": "name"' in consumer_text
    assert "I should inspect it" in consumer_text
    assert "Ada" in consumer_text and "answer" in consumer_text
    assert "The tool returned Ada." in consumer_text
    assert graph["conversion"]["consumer_target_positive_tokens"] > 0


def test_source_curated_review_keeps_non_oracle_claims_explicit(tmp_path):
    review = {
        "schema": "natlang.external-curated-sft-target-review/1",
        "source_sha256": "a" * 64,
        "source_rows": 1,
        "status": "approved",
        "admission_class": "source-curated-third-party-SFT",
        "oracle_verified": False,
        "native_execution_verified": False,
        "observed_actor_state_exchange": False,
    }
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(review, sort_keys=True) + "\n")
    review_sha = hashlib.sha256(review_path.read_bytes()).hexdigest()
    source_sha = "b" * 64
    target_sha = "c" * 64
    targets_path = tmp_path / "targets.jsonl"
    targets_path.write_text(json.dumps({
        "id": "row-1", "source_row_sha256": source_sha, "target_sha256": target_sha,
        "proof_sha256": review_sha, "decision_id": "review:row-1",
        "review_kind": review["admission_class"], "status": "approved",
        "oracle_verified": False, "native_execution_verified": False,
    }) + "\n")
    loaded = load_reviewed_targets(targets_path, review_path, source_sha256="a" * 64)
    proof = loaded[source_sha]
    assert proof["review_sha256"] == review_sha
    assert proof["oracle_verified"] is False
    assert proof["native_execution_verified"] is False


def test_exact_decision_hold_zeroes_only_bound_target_and_keeps_context(tmp_path):
    row = _source_row(tmp_path, label="gold")
    hold = {"id": "decision:held-target", "reason": "exact target hold"}
    # Rebuild through the normal API with an exact matched target-decision hold.
    tokenizer = _Tokenizer()
    graph = port_record_graph(row, source_tokenizer=tokenizer,
        source_tokenizer_fingerprint="1" * 64, expanded_tokenizer_fingerprint="2" * 64,
        role_ids=(2, 3, 4, 5), corpus_id="search-owl-overcap", source_file_sha256="3" * 64,
        consumer_admission={"status": "approved", "id": row.record.id,
            "source_row_sha256": hashlib.sha256(row.raw_line).hexdigest(),
            "target_sha256": hashlib.sha256(json.dumps(row.raw["target"], ensure_ascii=False,
                sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "decision_id": hold["id"], "proof_sha256": "4" * 64},
        decision_hold_exclusions=(hold,))
    parse_graph(graph, tokenizer_fingerprint="2" * 64)
    consumer = graph["nodes"][1]
    count = graph["conversion"]["consumer_target_context_tokens"]
    assert graph["conversion"]["consumer_target_positive_tokens"] == 0
    assert graph["conversion"]["decision_hold_target_exclusions"] == [hold]
    assert all(weight == 0 for weight in consumer["target_weights"][-count:])


def test_exact_source_hold_routes_graph_to_held_and_exact_target_hold_masks(tmp_path):
    row = _source_row(tmp_path, label="gold")
    authorization = {"decision_id": "decision:target"}
    hold_details = {row.record.id: {"id": row.record.id, "reason": "whole source held"},
                    "decision:target": {"id": "decision:target", "reason": "target held"}}
    source_hits, exclusions, decision_ids, source_hash_ids = decision_hold_policy(
        row, authorization, set(hold_details), hold_details)
    assert source_hits == [row.record.id]
    assert [item["id"] for item in exclusions] == ["decision:target"]
    assert decision_ids == ["decision:target"]
    assert source_hash_ids == []
    # A source hold is routed to test by the builder, with the original graph
    # intact for evaluation/context instead of being dropped.
    graph = _graph(row)
    graph["split"] = "test"
    for node in graph["nodes"]:
        node["split"] = "test"
    parsed = parse_graph(graph, tokenizer_fingerprint="2" * 64)
    assert graph["split"] == "test"
    assert graph["nodes"][0]["producing"]
    assert graph["nodes"][1]["imports"]
    assert parsed.plan.token_count > 0


def test_hold_loader_binds_current_file_and_id_reason(tmp_path):
    path = tmp_path / "holds.jsonl"
    path.write_text(json.dumps({"id": "decision:one", "reason": "held"}) + "\n")
    ids, details, bindings = load_hold_details([path])
    assert ids == {"decision:one"}
    assert details["decision:one"]["reason"] == "held"
    assert bindings[0]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert _hold_references({"id": "source:one", "nested": {
        "teacher_trajectory_id": "trajectory:one", "decision_id": "decision:one"}}) == (
            {"source:one", "trajectory:one"}, {"decision:one"})
