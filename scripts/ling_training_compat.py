#!/usr/bin/env python3
"""Prepare a separate, source-pinned Ling training model; never edit HF cache.

Weight files are linked to their resolved originals, not duplicated. Model code,
config and tokenizer files are copied. Sorted MoE dispatch and padding changes
are intentionally absent until their independent numerical reviews pass.
"""
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'training' / 'neuralese'))
from natlang_neuralese.common.hashing import sha256_file_hex as sha256_file  # noqa: E402

REVISION = '9a98e35fe1c9ee255f78dd64771c7ae15a799481'
SOURCE_SHA256 = 'c2509bf7ac580c262e2581d34d6403aa21682d2e10beb9ad85ad8820a7e33a40'
MODEL_FILE = 'modeling_bailing_moe_v3.py'
CONFIG_SHA256 = '9750d847957913f665a13c0b5a6537199e33c6f3ec970d9fcb55a0e5076d4012'
CONFIGURATION_SHA256 = 'f2c048966aec8a2f042cfeb1351f74d51a28589b409c55baae7d24e841c1f6c4'


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError('Reviewed source anchor is missing or ambiguous')
    return source.replace(before, after, 1)


def patch_completion_forward(source):
    """Patch only the causal-LM method; reject unsupported sparse/MTP requests."""
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == 'BailingMoeV3ForCausalLM')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == 'forward')
    lines = source.splitlines(keepends=True)
    forward = ''.join(lines[method.lineno - 1:method.end_lineno])
    forward = replace_once(forward, '        **kwargs,\n    )',
                           '        logits_to_keep: int = 0,\n        **kwargs,\n    )')
    # Validate before invoking the expensive decoder. The reviewed full model
    # has zero MTP layers; a future MTP model needs a separate loss review.
    forward = replace_once(forward,
        '        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions',
        '        if isinstance(logits_to_keep, bool) or not isinstance(logits_to_keep, int):\n'
        '            raise TypeError("Ling completion projection requires an integer logits_to_keep")\n'
        '        if logits_to_keep < 0:\n'
        '            raise ValueError("logits_to_keep must be nonnegative")\n'
        '        if logits_to_keep and self.num_nextn_predict_layers:\n'
        '            raise ValueError("Completion projection with MTP has not been reviewed")\n'
        '        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions')
    forward = replace_once(forward,
        '        logits = self.lm_head(hidden_states)',
        '        if logits_to_keep > hidden_states.shape[1]:\n'
        '            raise ValueError("logits_to_keep exceeds the input sequence length")\n'
        '        projection_states = hidden_states[:, -logits_to_keep:, :] if logits_to_keep else hidden_states\n'
        '        logits = self.lm_head(projection_states)')
    result = ''.join(lines[:method.lineno - 1]) + forward + ''.join(lines[method.end_lineno:])
    ast.parse(result)
    return result


def patch_source(source):
    if hashlib.sha256(source.encode()).hexdigest() != SOURCE_SHA256:
        raise ValueError('Unreviewed Ling source; expected pinned upstream source SHA-256')
    result = replace_once(source,
        'from transformers.utils.import_utils import is_torch_fx_available',
        'try:\n    from transformers.utils.import_utils import is_torch_fx_available\n'
        'except ImportError:\n    is_torch_fx_available = lambda: hasattr(torch, "fx")')
    result = replace_once(result,
        '        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]',
        '        if self.rope_type in ROPE_INIT_FUNCTIONS:\n'
        '            self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]\n'
        '        elif self.rope_type == "default":\n'
        '            def default_rope_init(config, device=None):\n'
        '                positions = torch.arange(0, config.head_dim, 2, device=device, dtype=torch.float32)\n'
        '                return 1.0 / (config.rope_theta ** (positions / config.head_dim)), 1.0\n'
        '            self.rope_init_fn = default_rope_init\n'
        '        else:\n            raise ValueError(f"Unsupported rope type: {self.rope_type}")')
    result = replace_once(result,
        '            scaling_factor = self.config.rope_scaling["factor"]',
        '            scaling_factor = self.config.rope_scaling.get("factor", 1.0)')
    result = replace_once(result,
        "        mode = 'fused_recurrent' if q_len <= 64 else self.mode",
        "        mode = self.mode if self.training or q_len > 64 else 'fused_recurrent'")
    return patch_completion_forward(result)


def prepare(source, output):
    source, output = source.resolve(), output.absolute()
    if output.resolve().is_relative_to(source):
        raise ValueError('Output must be outside the original model snapshot')
    original_bytes = (source / MODEL_FILE).read_bytes()
    if hashlib.sha256(original_bytes).hexdigest() != SOURCE_SHA256:
        raise ValueError('Unreviewed Ling source bytes; expected pinned upstream SHA-256')
    original = original_bytes.decode('utf-8')
    patched = patch_source(original)
    if (sha256_file(source / 'config.json') != CONFIG_SHA256 or
            sha256_file(source / 'configuration_bailing_moe_v3.py') != CONFIGURATION_SHA256):
        raise ValueError('Unreviewed Ling configuration source or model config')
    config = json.loads((source / 'config.json').read_text())
    if config.get('architectures') != ['BailingMoeV3ForCausalLM'] or config.get('num_nextn_predict_layers') != 0:
        raise ValueError('Expected reviewed Ling causal model without MTP layers')
    if output.exists() or output.is_symlink():
        raise FileExistsError('Output must be a new directory; original models are never overwritten')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    try:
        copied, linked = [], []
        for item in sorted(source.iterdir()):
            if not item.is_file():
                continue
            # Snapshot files can themselves be HF cache symlinks. Resolve them
            # before linking weights; copy all code/tokenizer bytes physically.
            target = output / item.name
            if item.name.endswith('.safetensors'):
                original_weight = item.resolve(strict=True)
                target.symlink_to(original_weight)
                linked.append({'name': item.name, 'target': str(original_weight),
                               'bytes': original_weight.stat().st_size})
            elif item.suffix in ('.py', '.json', '.jinja', '.txt', '.model', '.tiktoken') or item.name.startswith(('LICENSE', 'README')):
                shutil.copyfile(item, target)
                copied.append({'name': item.name, 'sha256': sha256_file(target)})
        (output / MODEL_FILE).write_text(patched)
        index = source / 'model.safetensors.index.json'
        if index.exists():
            expected_weights = set(json.loads(index.read_text())['weight_map'].values())
            if expected_weights != {item['name'] for item in linked}:
                raise ValueError('Model weight index does not match available weight files')
        (output / 'ling-training-compat.diff').write_text(''.join(difflib.unified_diff(
            original.splitlines(True), patched.splitlines(True), fromfile=MODEL_FILE,
            tofile='training-compatible/' + MODEL_FILE)))
        receipt = {'version': 'natlang.ling_training_compat/1', 'reviewed_revision': REVISION,
                   'source': str(source), 'output': str(output),
                   'upstream_source_sha256': SOURCE_SHA256,
                   'patched_source_sha256': sha256_file(output / MODEL_FILE),
                   'copied_original_files': copied, 'weight_links': linked,
                   'changes': ['transformers_rope_and_fx_compatibility', 'short_training_chunk_mode',
                               'integer_completion_projection'],
                   'not_enabled': ['sorted_moe_dispatch', 'padding_mask_fix'],
                   'weight_integrity': 'Links preserve originals; verify the pinned download receipt separately.',
                   'readiness': 'Prepared code; full-model training throughput and padded kernel equivalence require review.'}
        (output / 'ling-training-compat.json').write_text(json.dumps(receipt, indent=2) + '\n')
        return receipt
    except BaseException:
        # Only this invocation's new directory is removed; source is untouched.
        shutil.rmtree(output)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), indent=2))


if __name__ == '__main__':
    main()
