import hashlib
import json

from natlang_neuralese.data.text_corpus import gold_text_rows


class _Backend:
    def to_str(self):
        return "{}"


class _Tokenizer:
    chat_template = "fixture-template"
    all_special_tokens = []
    backend_tokenizer = _Backend()

    def get_vocab(self):
        return {"x": 0}

    def apply_chat_template(self, messages, **_kwargs):
        return json.dumps(messages, ensure_ascii=False)

    def __call__(self, text, **_kwargs):
        return {"input_ids": [ord(char) for char in text]}


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_gold_text_hydrates_marker_output_writer_and_preserves_exact_code():
    block_id = "nz1_" + "a" * 52
    body = "Supported facts from this pass; unresolved fields remain pending."
    marker = f"<|neuralese|>{body}<|/neuralese|>"
    code = f"const notes: Neuralese<string> = {marker}; return notes;"
    raw = json.dumps({"code": code, "finish": True}, ensure_ascii=False, separators=(",", ":"))
    sidecar = {
        "schema": "natlang.neuralese-code/1", "mode": "marker-output", "code_sha256": _sha(code),
        "parts": [
            {"type": "text", "text": "const notes: Neuralese<string> = "},
            {"$write": {"name": f"soft-state:{block_id}", "type": "Neuralese<string>",
                        "source": body, "code_source": marker}},
            {"type": "text", "text": "; return notes;"},
        ],
        "sites": [{"name": f"soft-state:{block_id}", "purpose": "validated-runtime-soft-state-writer",
                    "block_id": block_id, "body_sha256": _sha(body),
                    "writer_call_id": "eval-1", "writer_node": "eval-1#1",
                    "writer_action": "eval-finish-true"}],
    }
    group = ["case:shared-source-group"]
    common = {"split": "test", "source_groups": group,
              "training_admission": {"approved": True},
              "decision": {"training_approved": True, "failed_action": False}}
    writer = {**common, "id": "writer", "source_ref": {"source_row_sha256": "1" * 64},
              "messages": [{"role": "user", "content": "Write the accumulated note."}],
              "target": {"role": "assistant", "tool_calls": [{"id": "eval-1",
                  "function": {"name": "eval", "arguments": raw}, "neuralese_code": sidecar}]}}
    reader = {**common, "id": "reader", "source_ref": {"source_row_sha256": "1" * 64},
              "messages": [{"role": "user", "content": [
                  {"type": "text", "text": "Prior notes: "},
                  {"type": "neuralese", "id": block_id},
              ]}],
              "target": {"role": "assistant", "content": "Continue from the prior notes."}}
    train = {**common, "id": "train-anchor", "split": "train", "source_groups": ["case:train-anchor"],
             "messages": [{"role": "user", "content": "Independent training anchor."}],
             "target": {"role": "assistant", "content": "Anchor answer."}}

    rows, receipt, omissions, provenance = gold_text_rows([writer, reader, train], [], tokenizer=_Tokenizer())

    assert not omissions
    assert len(rows) == 3
    writer_text = next(row["text"] for row in rows if row["id"] == "writer")
    reader_text = next(row["text"] for row in rows if row["id"] == "reader")
    assert code in writer_text
    assert f"<|neuralese|>{body}<|/neuralese|>" in reader_text
    reader_provenance = next(row for row in provenance if row["id"] == "reader")
    assert reader_provenance["neuralese_context_attestations"][0]["writer_record_id"] == "writer"
    assert receipt["task_or_trajectory_admission_granted"] is False


def test_provider_expanded_soft_read_is_crisp_and_provenance_bound_to_existing_writer():
    block_id = "nz1_" + "c" * 52
    name = f"soft-state:{block_id}"
    body = "Observed note from the successful earlier producer."
    group = ["case:provider-read"]
    common = {"split": "test", "source_groups": group,
              "training_admission": {"approved": True},
              "decision": {"training_approved": True, "failed_action": False}}
    writer = {**common, "id": "provider-writer", "source_ref": {"source_row_sha256": "1" * 64},
              "messages": [{"role": "user", "content": "Write a note."}],
              "target": {"role": "assistant", "tool_calls": [{"function": {"name": "return_result",
                  "arguments": json.dumps({"status": "success", "value": {"$write": {
                      "name": name, "type": "Neuralese<string>", "source": body}}})}}]}}
    reader = {**common, "id": "provider-reader", "source_ref": {
                  "source_row_sha256": "2" * 64, "invocation_id": "call-reader"},
              "neuralese_conversion": {"external_context_inputs": [{
                  "schema": "natlang.external-context-input/1", "origin": "same-run-producer",
                  "learner_representation": "typed-read-linked-to-existing-writer",
                  "block_id": block_id, "type": "Neuralese<string>",
                  "body_sha256": _sha(body), "invocation_id": "call-reader",
                  "source_row_sha256": "2" * 64, "read_node": "call-reader#7",
                  "model_turn_node": "call-reader#turn1", "producer_write_node": "call-writer#6",
                  "transport_provenance_sha256": "3" * 64, "learned_vectors": False,
                  "qualification_certificate": False, "training_admission": False}]},
              "messages": [{"role": "user", "content": [
                  {"type": "text", "text": "Prior note: "},
                  {"type": "read", "name": name, "source": body}]}],
              "target": {"role": "assistant", "content": "Continue."}}
    anchor = {**common, "id": "provider-train-anchor", "split": "train",
              "source_groups": ["case:independent-anchor"],
              "messages": [{"role": "user", "content": "Independent training anchor."}],
              "target": {"role": "assistant", "content": "Anchor answer."}}

    rows, receipt, omissions, provenance = gold_text_rows([writer, reader, anchor], [], tokenizer=_Tokenizer())

    assert not omissions
    rendered = next(row["text"] for row in rows if row["id"] == "provider-reader")
    assert "Prior note: " + body in rendered
    assert "<|neuralese|>" not in rendered
    reader_provenance = next(row for row in provenance if row["id"] == "provider-reader")
    attest = reader_provenance["neuralese_context_attestations"]
    assert len(attest) == 1
    assert attest[0]["source_kind"] == "provider-expanded-same-run-read"
    assert attest[0]["writer_record_id"] == "provider-writer"
    assert attest[0]["read_node"] == "call-reader#7"
    assert receipt["hash_bound_reader_context_blocks"] == 1

    corrupt = json.loads(json.dumps(reader))
    corrupt["messages"][0]["content"][1]["source"] = "different body"
    _, _, rejected, _ = gold_text_rows([writer, corrupt, anchor], [], tokenizer=_Tokenizer())
    assert any(item["id"] == "provider-reader" and "digest mismatch" in item.get("detail", "")
               for item in rejected)
