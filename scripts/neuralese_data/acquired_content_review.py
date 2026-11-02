"""Bounded exact-token and role-structure review of one acquired graph."""
from __future__ import annotations
import hashlib
import json
import numpy as np


def content_review(row, objective):
    """Hash actual tokens/masks and record visible role boundaries, not semantics."""
    digest=hashlib.sha256();role_spec=objective.get('role_tokens') or {}
    start=role_spec.get('start');roles=role_spec.get('roles') or {};reverse={v:k for k,v in roles.items()}
    counts={k:0 for k in roles};unknown=0;transcript=[];node_shapes=[];total=0
    for node in row.get('nodes') or []:
        ids=np.asarray(node.get('token_ids') or [],dtype='<i8')
        weights=np.asarray(node.get('target_weights') or [],dtype='<f8')
        if ids.size!=weights.size:raise ValueError('source review saw unaligned token weights')
        digest.update(len(ids).to_bytes(8,'little'));digest.update(ids.tobytes());digest.update(weights.tobytes())
        total+=len(ids);positions=np.flatnonzero(ids==start) if isinstance(start,int) else []
        node_roles=[]
        for position in positions:
            idx=int(position)+1;role=reverse.get(int(ids[idx])) if idx<len(ids) else None
            if role is None:unknown+=1;node_roles.append('?')
            else:counts[role]+=1;node_roles.append(role)
        transcript.extend(node_roles)
        node_shapes.append({'imports':len(node.get('imports') or []),'producing':bool(node.get('producing')),
                            'roles':node_roles})
    shape=json.dumps(node_shapes,sort_keys=True,separators=(',',':')).encode()
    return {'id':row.get('id'),'exact_token_weight_sha256':digest.hexdigest(),
        'role_structure_sha256':hashlib.sha256(shape).hexdigest(),'role_counts':counts,
        'unknown_role_starts':unknown,'nodes':len(node_shapes),'forward_tokens':total,
        'scope':'exact token/raw-weight identity and visible role topology; no semantic or outcome claim'}
