"""Maple port vs the official reference implementation on identical tokens.

Loads the official bf16 MapleForCausalLM (trust_remote_code) and our Neuralese port engine, and reports per-model
next-token CE and top-1 accuracy, logit agreement and max logit difference on (a) plain English text and (b) the
first held windows of a Maple gold-text packet. A strong base model with high CE on both means a broken setup;
reference-vs-port divergence localizes it to the port.

    python scripts/maple_reference_parity.py --heads H --reference DIR --text-data T [--windows 2 --tokens 1024]
"""
import argparse
import json

import torch
import torch.nn.functional as F

ENGLISH = (
    "The history of the printing press begins in the fifteenth century, when Johannes Gutenberg combined movable "
    "metal type, oil-based ink and a wooden press adapted from wine making. Within a few decades printing shops "
    "had spread to more than two hundred cities across Europe, and the price of books fell dramatically. Scholars "
    "argue that this sudden abundance of printed material accelerated the Reformation, the Scientific Revolution "
    "and the rise of vernacular literature, because ideas could now travel faster than the authorities who wished "
    "to suppress them. ")


def scores(logits, ids):
    logits = logits[0, :-1].float()
    target = ids[0, 1:]
    return {'ce': float(F.cross_entropy(logits, target)), 'top1': float((logits.argmax(-1) == target).float().mean()),
            'tokens': target.numel()}


@torch.no_grad()
def port_logits(engine, ids):
    out = engine.backbone.forward_embeds(engine.backbone.embed(ids), logits=False)
    return torch.cat([engine.backbone.logits(chunk) for chunk in out['h_final'].split(512, 1)], 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--heads', required=True)
    p.add_argument('--reference', required=True)
    p.add_argument('--text-data', required=True)
    p.add_argument('--windows', type=int, default=2)
    p.add_argument('--tokens', type=int, default=1024)
    a = p.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    def default_rope(config, device=None, seq_len=None):
        # transformers 4.57's default (removed in 5.x): partial rotary over head_dim * partial_rotary_factor.
        head_dim = getattr(config, 'head_dim', None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * getattr(config, 'partial_rotary_factor', 1.0))
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
        return inv_freq, 1.0
    ROPE_INIT_FUNCTIONS.setdefault('default', default_rope)
    from natlang_neuralese.serve import load_engine
    tokenizer = AutoTokenizer.from_pretrained(a.reference)
    cases = {'english': tokenizer(ENGLISH * 3, return_tensors='pt').input_ids[:, :a.tokens]}
    chat = tokenizer.apply_chat_template([{'role': 'user', 'content': 'What is the capital of France? Answer briefly.'}],
                                         tokenize=False, add_generation_prompt=True) + 'The capital of France is Paris.'
    cases['chat'] = tokenizer(chat, return_tensors='pt').input_ids
    held = [json.loads(line) for line in open(a.text_data)]
    held = [row for row in held if row['split'] == 'test'][:a.windows]
    for i, row in enumerate(held):
        ids = row.get('token_ids') or tokenizer(row['text']).input_ids
        cases[f'held{i}'] = torch.tensor([ids[:a.tokens]])
        cases[f'held{i}_retokenized'] = tokenizer(row['text'], return_tensors='pt').input_ids[:, :a.tokens]
    results = {name: {} for name in cases}
    reference = AutoModelForCausalLM.from_pretrained(a.reference, trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()
    ref_logits = {}
    with torch.no_grad():
        for name, ids in cases.items():
            ids = ids.cuda()
            ref_logits[name] = reference(input_ids=ids).logits.float().cpu()
            results[name]['reference'] = scores(ref_logits[name], ids.cpu())
    del reference; torch.cuda.empty_cache()
    engine = load_engine(heads_checkpoint=a.heads, device='cuda')
    for name, ids in cases.items():
        logits = port_logits(engine, ids.cuda()).float().cpu()
        results[name]['port'] = scores(logits, ids)
        r = ref_logits[name]
        results[name]['port_vs_reference'] = {
            'top1_agreement': float((logits.argmax(-1) == r.argmax(-1)).float().mean()),
            'max_abs_logit_diff': float((logits - r).abs().max()),
            'first_position_max_abs_diff': float((logits[0, 0] - r[0, 0]).abs().max()),
            'same_ids_as_tokenizer': None}
        results[name]['first_tokens'] = tokenizer.convert_ids_to_tokens(ids[0, :12].tolist())
    print(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
