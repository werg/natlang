from array import array

from natlang_neuralese.train.text_supervision import context_only_mask


def test_context_only_mask_preserves_failed_feedback_and_declared_span_offsets():
    # [open, user, failed assistant, tool feedback, declared text, assistant, close]
    source = [90, 1, 2, 3, 4, 5, 6, 91]
    labels = [0, 2, 2, 5, 3, 2, 5, 0]
    result = context_only_mask(source, labels, [[4, 5]], lambda tokens: '<tool_response>rejected result')
    assert isinstance(result, array)
    assert list(result) == [0, 0, 0, 1, 0, 1, 0, 0]


def test_context_only_mask_supports_declared_spans_without_role_labels():
    result = context_only_mask([90, 1, 2, 3, 91], None, [[1, 3]], None)
    assert list(result) == [0, 0, 1, 1, 0]
