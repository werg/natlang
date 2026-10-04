import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.student_guidance import guided_generate, load_guide, rollback_token


class Tokenizer:
    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def decode(self, ids, **kwargs):
        return ''.join(chr(i) for i in ids)


class Disconnect:
    disconnected = False
    def poll(self):
        return self.disconnected
    def __call__(self, ids, scores, **kwargs):
        return torch.tensor([self.disconnected])


class Model:
    def generate(self, input_ids, attention_mask, max_new_tokens, logits_processor, stopping_criteria, **kwargs):
        ids = input_ids.clone()
        for _ in range(max_new_tokens):
            # At the rejected call name, try the same wrong token until the ban
            # selects the valid alternative. Thereafter complete the good call.
            reply = Tokenizer().decode(ids[0, 1:].tolist())
            opening = '<|tool_call_start|>['
            if reply == opening:
                scores = torch.full((1, 256), -100.0)
                scores[0, ord('x')] = 2
                scores[0, ord('e')] = 1
                for proc in logits_processor:
                    scores = proc(ids, scores)
                token = scores.argmax().item()
            else:
                wanted = opening + ('xyz()' if reply.startswith(opening + 'x') else 'eval()') + ']<|tool_call_end|>' + chr(3)
                token = ord(wanted[len(reply)])
            ids = torch.cat((ids, torch.tensor([[token]])), dim=1)
            if any(bool(stop(ids, None)[0]) for stop in stopping_criteria) or token == 3:
                break
        return ids


def generate(retries=4, limit=128):
    module = load_guide()
    inputs = SimpleNamespace(input_ids=torch.tensor([[1]]), attention_mask=torch.ones((1, 1), dtype=torch.long))
    return guided_generate(Model(), Tokenizer(), inputs,
        module.Settings(require_call=True, tools=['eval'], syntax=False, retries=retries), module,
        output_limit=limit, sampling={'do_sample': False}, eos_token_id=3, pad_token_id=0, disconnect=Disconnect())


def test_backtracking_rebuilds_prefix_and_bans_wrong_token():
    output, receipt = generate()
    assert Tokenizer().decode(output[0, 1:].tolist()).startswith('<|tool_call_start|>[eval()')
    assert len(receipt['rejections']) == 1
    assert receipt['rejections'][0]['reason'] == 'unknown-tool'
    assert receipt['attempted_tokens'] > len(output[0]) - 1 - receipt['forced_prefix_tokens']
    assert not receipt['budget_exhausted']
    assert receipt['mh_proposal_supported'] is False


def test_retry_exhaustion_is_visible_and_preserves_model_choice():
    output, receipt = generate(retries=0)
    assert 'xyz()' in Tokenizer().decode(output[0, 1:].tolist())
    assert receipt['rejections'][0]['accepted']


def test_prefix_has_to_fit_output_budget():
    with pytest.raises(ValueError, match='allowance'):
        generate(limit=5)


def test_unicode_mapping_decodes_whole_prefixes():
    class ByteTokenizer:
        def decode(self, ids, **kwargs):
            return bytes(ids).decode('utf-8', errors='replace')
    assert rollback_token(ByteTokenizer(), list('aéx'.encode()), 1) == 1
