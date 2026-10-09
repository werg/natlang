import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_neuralese_counterfactual_reader_proposal.py"
SPEC = importlib.util.spec_from_file_location("counterfactual_reader", SCRIPT)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class CounterfactualRecurrenceRowsTests(unittest.TestCase):
    def test_projects_admitted_body_and_reader_port_without_admitting_r(self):
        body = "Observed typed body"
        block_id = "nz1_example"
        producer_id = "derived-writer:source:001"
        reader_id = "source:002"
        writer = {
            "id": producer_id,
            "derived_target": {"exact_body_sha256": builder.sha_bytes(body.encode())},
            "target": {"role": "assistant", "tool_calls": [{"function": {"name": "return_result",
                "arguments": '{"status":"success","value":"Observed typed body"}'}}]},
        }
        target = {"role": "assistant", "tool_calls": [{"function": {"name": "return_result",
            "arguments": '{"status":"success","value":"Observed typed body"}'}}]}
        read_receipt = {"block": {"id": block_id, "body": body},
                        "block_read": {"node": "reader#read"}, "model_turn": {"node": "reader#turn"},
                        "context_occurrences": 1}
        reader = {"id": reader_id, "target": target,
                  "messages": [{"role": "user", "content": [{"type": "neuralese", "id": block_id}]}],
                  "source_ref": {"provider_expanded_read_contexts": [read_receipt]},
                  "training_admission": {"approved": True}}
        edge = {"proposal_id": "counterfactual:p->r", "producer": {
                    "derived_native_record_id": producer_id, "writer_block_id": block_id,
                    "body_sha256": builder.sha_bytes(body.encode()),
                    "root_body_admission_receipt": {"path": "body-receipt.json", "sha256": "a"*64},
                    "target_role": "approved-derived-body-supervision-not-sampled-action"},
                "observed_reader": {"native_record_id": reader_id,
                    "root_action_disposition": {"status": "root-admitted-action"},
                    "target_sha256": builder.sha_json(target), "reader_node": "reader#read",
                    "model_turn_node": "reader#turn",
                    "observed_source_graph_edge": {"writer_node": "writer#write", "reader_node": "reader#read"},
                    "read_inputs": [{"node": "writer#write", "block": block_id}],
                    "model_turn_inputs": [{"node": "reader#read", "block": block_id}]}}

        rows = builder.materialize_counterfactual_recurrence_rows(
            [edge], {producer_id: writer}, {reader_id: reader})
        self.assertEqual(len(rows), 2)
        by_role = {row["counterfactual_recurrence_view"]["role"]: row for row in rows}
        writer_view = by_role["derived-body-as-proposed-learned-writer"]
        reader_view = by_role["observed-reader-with-proposed-learned-writer-input"]
        self.assertEqual(writer_view["target"]["tool_calls"][0]["function"]["arguments"],
                         '{"status":"success","value":{"$write":{"name":"soft-state:nz1_example","type":"Neuralese<string>","source":"Observed typed body"}}}')
        self.assertEqual(reader_view["target"], target)
        self.assertEqual(reader_view["messages"][0]["content"],
                         [{"type": "read", "name": "soft-state:nz1_example", "source": body}])
        self.assertIs(reader_view["training_admission"]["approved"], False)
        self.assertIs(writer_view["training_admission"]["approved"], False)


if __name__ == "__main__":
    unittest.main()
