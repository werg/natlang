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


def test_merging_preserves_admission_and_does_not_cross_split_or_program():
    base={'id':'a','teacher_trajectory_id':'same','program_id':'p','split':'train',
          'source_groups':['g'],'prompt':'hello','completion':'go!',
          'training_admission':{'approved':True},'task_modality':'directory-reducer'}
    other={**base,'id':'b','prompt':'hellogo!next','completion':'done!'}
    merged=merge([base,other])
    assert len(merged)==1 and merged[0]['training_admission']['approved']
    assert merged[0]['task_modality']=='directory-reducer'
    assert merged[0]['chain_admission']['all_turns_approved']
    assert len(merge([base,{**other,'split':'test'}]))==2
    assert len(merge([base,{**other,'program_id':'q'}]))==2
