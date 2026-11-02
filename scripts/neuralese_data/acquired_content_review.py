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
    sample=None
    if int(hashlib.sha256(str(row.get('id')).encode()).hexdigest()[:8],16) % 512 == 0:
        from natlang_neuralese.train.own_output_supervision import graph_weights
        effective,_=graph_weights(row,objective);offset=0;remaining=8192;segments=[]
        for node in row.get('nodes') or []:
            ids=node.get('token_ids') or [];weights=effective[offset:offset+len(ids)];offset+=len(ids)
            run=[]
            for token,weight in zip(ids,weights,strict=True):
                if weight>0 and remaining>0:run.append(token);remaining-=1
                elif run:segments.append(run);run=[]
                if remaining==0:break
            if run:segments.append(run)
            if remaining==0:break
        sample={'selection':'sha256(graph-id) modulo 512 equals zero',
                'effective_positive_token_segments':segments,'token_limit':8192,
                'possibly_truncated':remaining==0}
    return {'id':row.get('id'),'exact_token_weight_sha256':digest.hexdigest(),
        'sample':sample,
        'role_structure_sha256':hashlib.sha256(shape).hexdigest(),'role_counts':counts,
        'unknown_role_starts':unknown,'nodes':len(node_shapes),'forward_tokens':total,
        'scope':'exact token/raw-weight identity and visible role topology; no semantic or outcome claim'}
