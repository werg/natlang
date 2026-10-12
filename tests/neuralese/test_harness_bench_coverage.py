"""Sampled harness decisions retain the exact complete-chain spine."""
import random

from natlang_neuralese.harness_bench.records import choose_targets


def test_sample_preserves_final_assistant_without_exceeding_budget():
    messages = [{'role': 'user', 'content': 'task'}]
    assistants = []
    for index in range(20):
        assistants.append(len(messages))
        messages.append({'role': 'assistant', 'content': str(index)})
        messages.append({'role': 'tool', 'content': [{'type': 'text', 'text': str(index)}]})
    for seed in range(32):
        selected = choose_targets(messages, assistants, 'sample', 8, random.Random(seed))
        assert len(selected) == len(set(selected)) == 8
        assert selected == sorted(selected)
        assert selected[-1] == assistants[-1]
        # The maximal selected record preserves every earlier source assistant verbatim.
        prefix_and_target = messages[:selected[-1] + 1]
        assert [m for m in prefix_and_target if m['role'] == 'assistant'] == [messages[i] for i in assistants]
        assert selected == choose_targets(messages, assistants, 'sample', 8, random.Random(seed))


def test_small_and_all_target_modes_preserve_exact_indices():
    messages = [{'role': 'assistant', 'content': str(i)} for i in range(3)]
    assistants = list(range(3))
    assert choose_targets(messages, assistants, 'sample', 1, random.Random(0)) == [2]
    assert choose_targets(messages, assistants, 'sample', 8, random.Random(0)) == assistants
    assert choose_targets(messages, assistants, 'all', 1, random.Random(0)) == assistants
    assert choose_targets(messages, [], 'sample', 8, random.Random(0)) == []
    assert choose_targets(messages, assistants, 'sample', 0, random.Random(0)) == []


def test_remaining_slots_prioritize_view_readers():
    messages = [
        {'role': 'user', 'content': 'task'},
        {'role': 'assistant', 'content': 'first'},
        {'role': 'tool', 'content': [{'type': 'text', 'text': 'view'}]},
        {'role': 'assistant', 'content': 'reader'},
        {'role': 'tool', 'content': 'ordinary result'},
        {'role': 'assistant', 'content': 'final'},
    ]
    assert choose_targets(messages, [1, 3, 5], 'sample', 2, random.Random(0)) == [3, 5]
