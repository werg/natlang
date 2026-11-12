"""serve/append_only.py: the actor keeps its own sampled ids for earlier replies without changing the prompt text."""
from natlang_neuralese.serve.append_only import TokenHistory, _bytes_to_unicode


class ByteTokenizer:
    """A byte-level BPE vocabulary: pieces in the GPT-2 alphabet, one added special token."""

    def __init__(self, pieces, added):
        alphabet = _bytes_to_unicode()
        self.pieces = [''.join(alphabet[b] for b in piece.encode()) for piece in pieces] + [None] * len(added)
        self.added_tokens_decoder = {len(pieces) + i: token for i, token in enumerate(added)}

    def __len__(self):
        return len(self.pieces)

    def convert_ids_to_tokens(self, ids):
        return [self.pieces[i] for i in ids]


A, B, AB, C, ABC, SPACE_A, END = range(7)


def history():
    return TokenHistory(ByteTokenizer(['a', 'b', 'ab', 'c', 'abc', ' a'], ['<|im_end|>']))


def text(h, ids):
    return b''.join(h.table[i] for i in ids)


def test_retokenized_reply_is_replaced_by_the_sampled_ids():
    h = history()
    h.store([C], [A, B, C, END])                  # the model sampled "abc" as three tokens
    rendered = [C, ABC, END, SPACE_A]             # the next prompt re-tokenizes it as one
    spliced = h.splice(rendered)
    assert spliced == [C, A, B, C, END, SPACE_A]
    assert text(h, spliced) == text(h, rendered)
    assert h.stats['append_only_spliced'] == 1


def test_already_append_only_prompt_is_returned_unchanged():
    h = history()
    h.store([C], [A, B, END])
    rendered = [C, A, B, END, SPACE_A]
    assert h.splice(rendered) is rendered


def test_changed_text_keeps_the_sampled_ids_up_to_the_last_shared_boundary():
    h = history()
    h.store([C], [A, B, C, END])                  # sampled "abc"
    rendered = [C, AB, END]                       # the harness kept only "ab"
    spliced = h.splice(rendered)
    assert spliced == [C, A, B, END]
    assert text(h, spliced) == text(h, rendered)


def test_unknown_prefix_and_non_byte_level_tokenizers_pass_through():
    h = history()
    h.store([A], [B, END])
    rendered = [C, ABC]
    assert h.splice(rendered) is rendered

    class Words:
        added_tokens_decoder = {}

        def __len__(self):
            return 2

        def convert_ids_to_tokens(self, ids):
            return ['▁hello' for _ in ids]

    plain = TokenHistory(Words())
    assert not plain.enabled and plain.splice([0, 1]) == [0, 1]


def test_longest_stored_prefix_wins_and_the_lru_is_bounded():
    h = TokenHistory(ByteTokenizer(['a', 'b', 'ab', 'c', 'abc', ' a'], ['<|im_end|>']), max_entries=2)
    h.store([C], [A, END])
    h.store([C, A, END, C], [A, B, C, END])
    assert h.splice([C, A, END, C, ABC, END]) == [C, A, END, C, A, B, C, END]
    h.store([B], [A])
    h.store([A], [B])
    assert len(h.entries) == 2 and h.splice([C, A, END, C, ABC, END]) == [C, A, END, C, ABC, END]


def test_splices_chain_across_an_episode():
    """Call 2's prompt re-tokenizes call 1's (spliced) prompt the plain way: it is still found, and the whole history
    keeps the sampled ids."""
    h = history()
    h.store([C], [A, B, C, END])                       # call 0: sampled "abc" as three tokens
    call1 = h.splice([C, ABC, END, SPACE_A])           # call 1 renders it as one token
    assert call1 == [C, A, B, C, END, SPACE_A]
    h.store(call1, [A, B, C, END])                     # call 1 samples "abc" again
    rendered = [C, ABC, END, SPACE_A, ABC, END, C]     # call 2: the plain tokenization of the same text
    call2 = h.splice(rendered)
    assert call2 == [C, A, B, C, END, SPACE_A, A, B, C, END, C]
    assert text(h, call2) == text(h, rendered)
