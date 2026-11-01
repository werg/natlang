"""Authenticate ordinary-to-derived-native source corrections from exact graph pairs."""
from __future__ import annotations
import json
from pathlib import Path
from ..common.hashing import sha256_file_hex_cached
from ..common.jsonio import open_jsonl_text
from ..data.native_graph_schema import indexed_graph_metadata, communication_classification
from ..data.native_sources import _row_bindings
from ..data.static_native_context import verify_graphification
from .own_output_supervision import policy_objective

SCHEMA='natlang.static-native-context-source-correction/1'

def verify_static_context_correction(proof_path, declaration):
    if not isinstance(declaration,dict) or declaration.get('schema')!=SCHEMA:
        raise ValueError('static native context correction lacks typed evidence')
    def checked(ref):
        if not isinstance(ref,dict) or not isinstance(ref.get('path'),str) or not isinstance(ref.get('sha256'),str):
            raise ValueError('static context correction artifact lacks exact binding')
        path=Path(ref['path']);path=(path if path.is_absolute() else Path(proof_path).resolve().parent/path).resolve(strict=True)
        digest=sha256_file_hex_cached(path)
        if digest!=ref['sha256'] or 'bytes' in ref and path.stat().st_size!=ref['bytes']:
            raise ValueError('static context correction artifact changed')
        return path,{'path':str(path),'sha256':digest,'bytes':path.stat().st_size}
    pairs_path,pairs_ref=checked(declaration.get('graph_pairs'));policy_path,policy_ref=checked(declaration.get('ce_policy'))
    objective=policy_objective(json.loads(policy_path.read_text()));result={}
    with open_jsonl_text(pairs_path) as rows:
        for line in rows:
            if not line.strip():continue
            item=json.loads(line);parent=item['parent'];child=item['child'];ident=parent['id']
            if ident in result or child.get('id')!=ident:raise ValueError('static context graph pairs repeat or change row identity')
            verify_graphification(parent,child,objective)
            before=indexed_graph_metadata(parent,tokenizer_fingerprint=parent['tokenizer_fingerprint'])
            after=indexed_graph_metadata(child,tokenizer_fingerprint=parent['tokenizer_fingerprint'])
            old_class=communication_classification(before);new_class=communication_classification(after)
            if old_class['communication'] or not new_class['producer_consumer'] or new_class['external_state_consumption']:
                raise ValueError('static context correction does not change ordinary to local producer/consumer')
            result[ident]={'old_binding':_row_bindings([before])[ident], 'new_binding':_row_bindings([after])[ident]}
    if not result or len(result)!=declaration.get('rows'):raise ValueError('static context correction graph pair inventory differs')
    return result,[pairs_ref,policy_ref]
