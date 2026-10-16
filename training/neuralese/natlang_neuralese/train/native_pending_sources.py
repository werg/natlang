"""Byte-bound admitted sources that remain mandatory before capacity activation.

This module declares sources, never an optimizer or a second exposure history.
Native continuation frames remain the sole owner of required/earned visits.
"""
from __future__ import annotations
import json
from pathlib import Path
import re

from ..common.hashing import sha256_file_hex_cached

SCHEMA='natlang.native-pending-sources/1'


def bound(reference,parent):
    path=Path(reference['path'])
    if not path.is_absolute():path=Path(parent)/path
    path=path.resolve(strict=True)
    if sha256_file_hex_cached(path)!=reference.get('sha256'):
        raise ValueError('pending native source artifact differs from its immutable binding')
    return path


def pending_source_dependencies(path):
    """Use the existing graph-source dependency verifier, including full cohort."""
    from .recipe import native_source_dependencies
    path=Path(path).resolve(strict=True);value=json.loads(path.read_text())
    if value.get('schema')!=SCHEMA or not isinstance(value.get('sources'),list) or not value['sources']:
        raise ValueError('pending native sources need explicit complete-cohort declarations')
    dependencies={str(path):sha256_file_hex_cached(path)}
    for source in value['sources']:
        for role in ('full_cohort','pending_native_graphs'):
            target=bound(source[role],path.parent)
            dependencies[str(target)]=source[role]['sha256']
            for item in native_source_dependencies(target):dependencies[item['path']]=item['sha256']
        for reference in source.get('evidence') or []:
            target=bound(reference,path.parent);dependencies[str(target)]=reference['sha256']
    return [{'path':key,'sha256':digest} for key,digest in sorted(dependencies.items())]


def load_pending_sources(path,*,tokenizer_fingerprint):
    path=Path(path).resolve(strict=True);dependencies=pending_source_dependencies(path)
    value=json.loads(path.read_text())
    if value.get('tokenizer_fingerprint')!=tokenizer_fingerprint:
        raise ValueError('pending native source tokenizer differs from the actual student')
    rows={}
    for source in value['sources']:
        declared=source.get('rows')
        if not isinstance(declared,list) or not declared:
            raise ValueError('pending native source requires all immutable row bindings and quotas')
        for row in declared:
            key=row.get('id');binding=row.get('binding');required=row.get('required_visits')
            if (not isinstance(key,str) or not key or key in rows or not isinstance(binding,str)
                    or re.fullmatch('[0-9a-f]{64}',binding) is None
                    or type(required) is not int or required<1):
                raise ValueError('pending native source row identities/bindings/required visits are invalid or duplicated')
            rows[key]={'binding':binding,'required_visits':required}
    return {'binding':{'path':str(path),'sha256':sha256_file_hex_cached(path)},
        'sources':[{key:item for key,item in source.items() if key!='rows'} for source in value['sources']],'rows':rows,'dependencies':dependencies}


def declaration(full_cohort,pending_native_graphs,*,tokenizer_fingerprint,role_ids,required_visits=1,evidence=()):
    """Compile actual pending graph identities once using the shared parser/cache."""
    from ..data.native_sources import NativeGraphSource
    if type(required_visits) is not int or required_visits<1:
        raise ValueError('pending source required visits must be a positive integer')
    full_cohort,pending_native_graphs=Path(full_cohort).resolve(strict=True),Path(pending_native_graphs).resolve(strict=True)
    source=NativeGraphSource(pending_native_graphs,tokenizer_fingerprint,required_split='train',role_ids=role_ids)
    from .recipe import native_source_dependencies
    native_source_dependencies(full_cohort)
    native_source_dependencies(pending_native_graphs)
    return {'schema':SCHEMA,'tokenizer_fingerprint':tokenizer_fingerprint,'sources':[{
        'full_cohort':{'path':str(full_cohort),'sha256':sha256_file_hex_cached(full_cohort)},
        'pending_native_graphs':{'path':str(pending_native_graphs),'sha256':sha256_file_hex_cached(pending_native_graphs)},
        'evidence':list(evidence),'rows':[{'id':entry['id'],'binding':entry['binding'],'required_visits':required_visits}
            for entry in source.entries]}]}
