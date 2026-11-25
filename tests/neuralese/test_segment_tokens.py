"""Memoized tokenization by added-token pieces (natlang_neuralese/serve/segment_tokens.py) equals the tokenizer's own."""
import random

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import AddedToken, PreTrainedTokenizerFast

from natlang_neuralese.serve.segment_tokens import SegmentTokenCache

SPECIALS = ['<|im_start|>', '<|im_end|>', '<|endoftext|>']
PLAIN_ADDED = ['<think>', '</think>', '<tool_call>']
WORDS = ['assistant', 'user', 'system', 'the', 'tool', 'call', 'result', '{"a": 1}', 'x', '\n', '  ', 'think', '<', '>',
         'im_start', '|', 'émoji 🙂', '123456', "it's", 'return_result(', ')']


def tokenizer():
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    rng = random.Random(0)
    corpus = [' '.join(rng.choice(WORDS) for _ in range(30)) for _ in range(300)]
    backend.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=400, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=backend)
    tok.add_special_tokens({'additional_special_tokens': [AddedToken(t, normalized=False) for t in SPECIALS]})
    tok.add_tokens([AddedToken(t, normalized=False, special=False) for t in PLAIN_ADDED])
    return tok


def texts(n, seed=1):
    rng = random.Random(seed)
    pieces = WORDS + SPECIALS + PLAIN_ADDED + ['<|im_', 'end|>', '</think', ' <think>', '\n\n']
    history = ''
    for _ in range(n):    # a growing history, as each call of an episode renders it again
        history += ''.join(rng.choice(pieces) for _ in range(rng.randrange(1, 25)))
        yield history
        yield ''.join(rng.choice(pieces) for _ in range(rng.randrange(0, 40)))


def test_pieces_tokenize_as_the_whole_text_and_repeat_from_memory():
    tok = tokenizer()
    cache = SegmentTokenCache.for_tokenizer(tok, budget_chars=2000)
    assert cache is not None
    for text in texts(150):
        for escaped in (False, True):
            assert cache.tokens(text, escaped=escaped) == \
                tok(text, add_special_tokens=False, split_special_tokens=escaped)['input_ids']
    assert cache.hits > cache.misses and cache.chars <= 2000 + max(len(k) for k, _ in cache.memo)


def test_a_tokenizer_not_exact_by_pieces_gets_no_cache(monkeypatch):
    tok = tokenizer()
    tok.add_tokens([AddedToken('<strip>', lstrip=True, normalized=False)])
    assert SegmentTokenCache.for_tokenizer(tok) is None
    monkeypatch.setenv('NATLANG_SEGMENT_TOKEN_CACHE', '0')
    assert SegmentTokenCache.for_tokenizer(tokenizer()) is None
