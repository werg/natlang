from scripts.materialize_studio_teacher import materialize


def test_studio_teacher_projection_keeps_reasoning_and_normalized_choices():
    trace = [{"version": "reduction-trace/1", "seq": 0}]
    row = {"schema": "natlang.studio_teacher_trajectory/1", "id": "teacher:case:7",
        "case": {"id": "studio:case", "target": "studio:wiki", "split": "train",
                 "source_revision": "abc", "expected": {"ok": True}},
        "provenance": {"model": "teacher", "source_revision": "abc"},
        "outcome": {"accepted": True}, "runs": {"reducer": {"trace": trace}, "view": {"trace": trace}},
        "exchanges": [{"request": {"messages": [{"role": "user", "content": "do it"}],
                                     "tools": [{"function": {"parameters": {"x-private": [1]}}}]},
                       "assistant": {"content": "", "reasoning": "Use the supplied state.",
                                     "calls": [{"tool": "write", "arguments": {"path": "return", "value": 7}}]}}]}
    samples = materialize(row)
    assert samples[0]["teacher_reasoning"] == "Use the supplied state."
    assert samples[0]["target"]["tool_calls"][0]["function"]["name"] == "write"
    assert "x-private" not in samples[0]["tools"][0]["function"]["parameters"]
    assert samples[0]["trace_admission"]["kind"] == "exact-studio-state-oracle"
