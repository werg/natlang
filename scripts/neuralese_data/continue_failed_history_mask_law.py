"""Rebind masks only after proving their complete role/failure callable law equal.

An unrelated helper edit changes the historical whole-file mask identity. This
static derivative preserves every source, row, span, evidence and decoder byte;
it never qualifies a changed detector or invents newly computed masks.
"""
from pathlib import Path
import copy
import json
import os

from natlang_neuralese.common.hashing import canonical_json_sha256_hex as digest
from natlang_neuralese.common.jsonio import write_json_durable
from natlang_neuralese.train import text_supervision
from natlang_neuralese.train.native_failed_history import binding, checked, load_manifest, mask_law, SCHEMA
from scripts.neuralese_data.source_policy_identity import scoped_python_policy

ROOTS = ('chat_roles', 'failed_assistant_mask', 'failed_feedback')


def continue_law(reference, old_helper, out):
    source = checked(reference)
    original = json.loads(source.read_text())
    old_helper = Path(old_helper)
    old_binding = binding(old_helper)
    new_binding = binding(text_supervision.__file__)
    before = scoped_python_policy(old_helper, ROOTS, allow_imports=True)
    after = scoped_python_policy(text_supervision.__file__, ROOTS, allow_imports=True)
    if before != after:
        raise ValueError('failed-history role/failure callable law changed; recompute masks')
    # AST equality includes the regex definitions and their exact imports. Use
    # the now-identical actual patterns to reconstruct the old full-file law.
    old_law = digest({'helper_sha256': old_binding['sha256'],
        'failure_patterns': [(pattern.pattern, pattern.flags)
                             for pattern in text_supervision.FAILED_TOOL_FEEDBACK]})
    if original.get('schema') != SCHEMA or original.get('mask_law_sha256') != old_law:
        raise ValueError('historical mask law does not bind the supplied old helper')
    value = copy.deepcopy(original)
    value['mask_law_sha256'] = mask_law()
    value['mask_law_continuation'] = {'schema': 'natlang.failed-history-same-callable-law/1',
        'original_manifest': reference, 'old_helper': old_binding, 'new_helper': new_binding,
        'old_helper_source': old_helper.read_text(), 'callable_law': before,
        'mask_values_recomputed': False, 'row_source_span_evidence_changes': 0}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for asset in original['tokenizer']['files']:
        path = checked(asset, root=source.parent)
        destination = out / asset['path']
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            os.link(path, destination)  # Independently bound immutable decoder assets.
        checked({**asset, 'path': str(destination)})
    target = out / 'manifest.json'
    if target.exists():
        if json.loads(target.read_text()) != value:
            raise ValueError('continued mask destination differs')
    else:
        write_json_durable(target, value)
    result = binding(target)
    load_manifest(result)  # Current unmodified consumer, including decoder custody.
    if checked(reference) != source or binding(old_helper) != old_binding or binding(text_supervision.__file__) != new_binding:
        raise ValueError('failed-history law continuation changed during verification')
    return result
