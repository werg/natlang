"""Real-checkpoint parity of the Maple-family port with transformers on a Mellum (or Maple) checkpoint (GB10).

Loads transformers' reference, then our ``load_maple`` (one at a time, BF16), on the first ``--tokens`` tokens of a
text-corpus document; reports CE of each, argmax agreement and logit differences.

    python scripts/mellum_parity.py --model DIR --text-data T [--tokens 2048]
"""
import argparse
import gc
import json

import torch
import torch.nn.functional as F


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--text-data', required=True)
    p.add_argument('--tokens', type=int, default=2048)
    a = p.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from natlang_neuralese.maple.model import load_maple
    tokenizer = AutoTokenizer.from_pretrained(a.model)
    text = next(row['text'] for row in map(json.loads, open(a.text_data)) if len(row['text']) > 8 * a.tokens)
    ids = tokenizer(text, return_tensors='pt').input_ids[:, :a.tokens].cuda()

    def measure(logits):
        logits = logits.float()
        return logits, float(F.cross_entropy(logits[0, :-1], ids[0, 1:]))

    with torch.no_grad():
        reference = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map='cuda').eval()
        expected, expected_ce = measure(reference(input_ids=ids).logits)
        expected = expected.cpu()
        del reference
        gc.collect()
        torch.cuda.empty_cache()
        ours = load_maple(a.model, device='cuda', dtype=torch.bfloat16, ternary_attention=False).eval()
        actual, actual_ce = measure(ours(ids).logits)
        actual = actual.cpu()
    difference = (actual - expected).abs()
    print(json.dumps({
        'model': a.model, 'tokens': ids.shape[1], 'reference_ce': expected_ce, 'ours_ce': actual_ce,
        'argmax_agreement': float((actual.argmax(-1) == expected.argmax(-1)).float().mean()),
        'max_abs_logit_difference': float(difference.max()), 'mean_abs_logit_difference': float(difference.mean()),
        'first_position_max_abs': float(difference[0, 0].max())}, indent=1))


if __name__ == '__main__':
    main()
