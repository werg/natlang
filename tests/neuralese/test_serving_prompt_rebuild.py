"""The current ts-host serving request rebuilt for recorded tool-loop calls (data/serving_prompt.py over ts-host's code;
the rebuild itself is tested against live agent requests in ts-host/test/serving-request.test.mjs)."""
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training" / "neuralese"))
from natlang_neuralese.data.prompt_rendering import prompt_pieces, render_system_prompt  # noqa: E402
from natlang_neuralese.data.serving_prompt import serving_identity, serving_messages  # noqa: E402

PIECES = {p["id"]: p["text"] for p in prompt_pieces()}
OPENING = "You are inside this call: f(): number\n\nInstructions:\nReturn 1."


def test_recorded_runtime_prose_moves_into_the_call_opening():
    guidance = "Program guidance: prefer the newest record."
    system = PIECES["interpreter"] + PIECES["function-tools"] + "\n\n" + guidance
    base = PIECES["interpreter"] + PIECES["generation-guidance"]  # natlang run and collection send one system prompt
    for entry in ("collector", "runtime"):
        messages, tools, report = serving_messages([{"role": "system", "content": system},
                                                    {"role": "user", "content": OPENING}], [], entry=entry)
        assert report["mode"] == f"serving-{entry}", report
        assert report["call_surface"] == {"depthLimited": False, "ownCode": True, "folder": None, "fileTools": "all"}
        assert messages[0]["content"] == base
        assert messages[1]["content"] == OPENING + "\n\n" + guidance + "\n\n" + PIECES["function-tools"].strip()
        assert [t["function"]["name"] for t in tools] == ["eval", "read_page", "read_code", "edit_code", "diff_code",
                                                         "compact_history", "return_result"]
        assert report["ts_host"] == serving_identity()


def test_other_calls_keep_their_recording():
    messages = [{"role": "system", "content": "Judge whether the step made progress."}, {"role": "user", "content": "x"}]
    out, tools, report = serving_messages(messages, None, entry="runtime")
    assert (out, tools, report["mode"], report["reason"]) == (messages, None, "recorded", "not a tool-loop call")


def test_program_header_rendering_is_ts_hosts():
    header = render_system_prompt(PIECES["interpreter"] + "\n\nKeep answers short.", "program-header")
    assert header.endswith("Keep answers short.") and "return_result" not in header
    assert render_system_prompt("x", "full") == "x"


def test_zero_context_segmented_weights_keep_only_targets():
    import torch
    from natlang_neuralese.train.text_supervision import segmented_position_weights
    span = torch.ones((1, 12))
    weights = segmented_position_weights(span, [(3, 5), (8, 10)], context_weight=0.0, feedback_weight=0.0)[0]
    assert all(weights[i] == 0 for i in (0, 1, 2, 5, 6, 7, 10, 11))
    assert all(weights[i] > 0 for i in (3, 4, 8, 9))
    assert math.isclose(float(weights.sum()), 12.0, rel_tol=1e-6)
