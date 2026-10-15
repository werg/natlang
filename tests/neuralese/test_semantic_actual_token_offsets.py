"""Real HF byte-level noncanonical and partial UTF-8 source token offsets."""
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteDecoder
from transformers import PreTrainedTokenizerFast

from natlang_neuralese.model.semantic_message import retained_source_offsets


def test_actual_noncanonical_bytelevel_and_utf8_fragments_retain_all_ids():
    vocab = {piece: index for index, piece in enumerate(sorted(ByteLevel.alphabet()))}
    for piece in ['he', 'hel', 'hell', 'hello']:
        vocab[piece] = len(vocab)
    backend = Tokenizer(BPE(vocab, [('h', 'e'), ('he', 'l'), ('hel', 'l'), ('hell', 'o')]))
    backend.pre_tokenizer = ByteLevel(add_prefix_space=False)
    backend.decoder = ByteDecoder()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend)
    text = 'hello €'
    canonical = tokenizer(text, add_special_tokens=False)['input_ids']
    assert canonical[0] == vocab['hello']
    ids = [vocab[letter] for letter in 'hello'] + canonical[1:]
    assert ids != canonical and tokenizer.decode(ids, clean_up_tokenization_spaces=False) == text
    original = list(ids)
    offsets, policy = retained_source_offsets(tokenizer, text, ids)
    assert ids == original and len(offsets) == len(ids)
    assert policy == 'actual-token-decoder-alignment'
    assert offsets[:5] == [[0,1], [1,2], [2,3], [3,4], [4,5]]
    assert offsets[-1][1] == len(text)
    assert sum(start == end for start, end in offsets) >= 2
    assert all(0 <= start <= end <= len(text) for start, end in offsets)
    assert all(one[1] == two[0] for one, two in zip(offsets, offsets[1:]))
    exact, policy = retained_source_offsets(tokenizer, text, canonical)
    assert policy == 'exact-tokenizer-offsets' and len(exact) == len(canonical)
