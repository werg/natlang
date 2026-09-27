"""A conversation's turns merge into one sequence whose trained spans are exactly the turns' completions."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from merge_sft_chains import merge  # noqa: E402


def test_later_turns_extend_the_sequence_and_masked_reasoning_is_context():
    first = {"id": "a:0", "teacher_trajectory_id": "a", "prompt": "<u>Go", "completion": "note</think>call<e>",
             "completion_masked": len("note</think>")}
    second = {"id": "a:1", "teacher_trajectory_id": "a", "prompt": first["prompt"] + first["completion"] + "<t>1",
              "completion": "done<e>"}
    other = {"id": "a:2", "teacher_trajectory_id": "a", "prompt": "<u>Child", "completion": "x<e>"}
    merged = merge([second, other, first])
    assert [row["turns"] for row in merged] == [["a:0", "a:1"], ["a:2"]]
    assert merged[0]["segments"] == [["<u>Go", False], ["note</think>", False], ["call<e>", True], ["<t>1", False],
                                     ["done<e>", True]]
    assert "".join(text for text, _ in merged[0]["segments"]) == second["prompt"] + second["completion"]
