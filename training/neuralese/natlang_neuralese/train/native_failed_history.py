"""Source-bound failed-history masks, preserving raw row identities/exposure."""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import json
import re
from pathlib import Path

from ..common.hashing import canonical_json_sha256_hex as digest,sha256_file_hex_cached,sha256_hex
from . import text_supervision

SCHEMA='natlang.native-failed-assistant-history-mask/1'


def binding(path):
    path=Path(path)
    return {'path':str(path),'sha256':sha256_file_hex_cached(path),'bytes':path.stat().st_size}


def checked(reference, *, root=None):
    path=Path(reference['path'])
    if root is not None:
        if path.is_absolute():raise ValueError('failed-history decoder asset must be manifest-relative')
        path=(root/path).resolve(strict=True)
        if not path.is_relative_to(root.resolve()):raise ValueError('failed-history decoder asset escapes manifest')
    actual=binding(path)
    if any(actual[key]!=reference.get(key)for key in ('sha256','bytes')):
        raise ValueError('failed-history artifact binding mismatch')
    return path


def mask_law():
    # The exact shared helper file is already shipped in every learner package.
    # Bind it conservatively rather than adding a runtime dependency on a data CLI.
    return digest({'helper_sha256':sha256_file_hex_cached(text_supervision.__file__),
        'failure_patterns':[(pattern.pattern,pattern.flags)for pattern in text_supervision.FAILED_TOOL_FEEDBACK]})


def _mask(row,tokenizer):
    """Derive exact role/failure evidence from original immutable token rows."""
    conversion=row.get('conversion')or{}
    if (row.get('split')!='train' or len(row.get('nodes')or[])!=1
            or conversion.get('role_interpretation')not in ('ordinary-text','whole-trajectory-chat-roles')):
        raise ValueError('failed-history masks require an explicit ordinary/whole-trajectory TRAIN row')
    node=row['nodes'][0];ids=node['token_ids']
    start=tokenizer.convert_tokens_to_ids('<|im_start|>')
    roles={tokenizer.convert_tokens_to_ids(role):role for role in ('system','user','assistant','tool')}
    if (len(roles)!=4 or start in roles or start not in ids
            or any(type(token)is not int or token==getattr(tokenizer,'unk_token_id',None)for token in (start,*roles))):
        raise ValueError('failed-history tokenizer lacks exact native chat roles')
    prefixes=tuple(tuple(tokenizer(text,add_special_tokens=False)['input_ids'])
        for text in ('\n<tool_response>','\n\n<tool_response>'))
    labels=text_supervision.chat_roles(ids,start_id=start,role_ids=roles,tool_response_prefixes=prefixes)
    evidence=[]
    mask=text_supervision.failed_assistant_mask(ids,labels,
        lambda values:tokenizer.decode(values,skip_special_tokens=False),evidence=evidence)
    boundary=conversion.get('source_suffix_start') if conversion.get('role_interpretation')=='ordinary-text' else len(ids)
    if type(boundary)is not int or not 0<=boundary<=len(ids):
        raise ValueError('failed-history ordinary row needs exact original suffix coordinates')
    if (conversion.get('role_interpretation')=='ordinary-text'
            and any(label not in (text_supervision.ROLE_CODES.index('assistant_reply'),
                text_supervision.ROLE_CODES.index('assistant_reasoning'))for label in labels[boundary:])):
        raise ValueError('failed-history current suffix is not entirely the actual assistant target')
    if sha256_hex(tokenizer.decode(ids,skip_special_tokens=False).encode())!=conversion.get('source_text_sha256'):
        raise ValueError('failed-history row is not its exact authenticated rendered source text')
    positions=[index for index in range(boundary)if mask[index]and node['target_weights'][index]>0]
    if any(mask[boundary:]):raise ValueError('failed-history mask would remove the current assistant target')
    spans=[]
    for index in positions:
        if spans and spans[-1][1]==index:spans[-1][1]=index+1
        else:spans.append([index,index+1])
    return spans,evidence


def row_mask_record(row,tokenizer,*,source_sha256,source_evidence):
    from ..data.text_corpus import tokenizer_fingerprint
    if tokenizer_fingerprint(tokenizer)!=row.get('tokenizer_fingerprint'):
        raise ValueError('failed-history decoder differs from actual row tokenizer')
    spans,evidence=_mask(row,tokenizer)
    if not spans:return None
    return mask_record(row,source_sha256=source_sha256,source_evidence=source_evidence,
                       zero_spans=spans,failure_evidence=evidence)


def mask_record(row,*,source_sha256,source_evidence,zero_spans,failure_evidence):
    """Pure serialization; the consumer independently verifies every claimed span."""
    return {'source_sha256':source_sha256,'id':row['id'],'split':row['split'],
        'source_groups':row['source_groups'],'tokenizer_fingerprint':row['tokenizer_fingerprint'],
        'raw_row_sha256':digest(row),'zero_spans':zero_spans,'failure_evidence':failure_evidence,
        'source_evidence':source_evidence}


def validate_source_evidence(row,evidence):
    conversion=row.get('conversion')or{}
    if conversion.get('role_interpretation')=='ordinary-text':
        teacher=conversion.get('teacher_source')or{}
        guard=evidence.get('structured_input_guard')or{}
        if (evidence.get('kind')!='verified-teacher-capture'
                or evidence.get('teacher_source_sha256')!=digest(teacher)
                or not all(re.fullmatch('[0-9a-f]{64}',str(teacher.get(key,'')))
                    for key in ('source_row_sha256','source_artifact_sha256'))
                or guard.get('schema')!='natlang.reviewed-chat-input-guard/1'
                or guard.get('policy')!='no-embedded-renderer-chat-boundaries'
                or guard.get('forbidden_markers')!=['<|im_start|>','<|im_sep|>','<|im_end|>']
                or not re.fullmatch('[0-9a-f]{64}',str(guard.get('input_sha256','')))):
            raise ValueError('failed-history ordinary source lacks structured capture custody')
    else:
        intent=conversion.get('supervision_policy')or{}
        if (evidence.get('kind')!='declared-whole-trajectory'
                or intent.get('failed_assistant_weight')!=0
                or evidence.get('supervision_policy_sha256')!=digest(intent)
                or not re.fullmatch('[0-9a-f]{64}',str(conversion.get('source_text_sha256','')))
                or evidence.get('source_text_sha256')!=conversion.get('source_text_sha256')):
            raise ValueError('failed-history whole trajectory lacks declared zero-failure source intent')


def load_manifest(reference):
    path=checked(reference)
    value=json.loads(path.read_text())
    if (value.get('schema')!=SCHEMA or value.get('mask_law_sha256')!=mask_law()
            or not isinstance(value.get('rows'),list)or not value['rows']):
        raise ValueError('failed-history manifest needs exact current role/failure law and affected rows')
    decoder=value.get('tokenizer')or{}
    files=decoder.get('files')or[]
    if not files or len({item['path']for item in files})!=len(files):
        raise ValueError('failed-history decoder inventory is empty/ambiguous')
    dependencies=[binding(path)]
    for item in files:dependencies.append(binding(checked(item,root=path.parent)))
    # The tokenizer directory itself is relative; only declared files are used.
    directory=Path(decoder.get('directory',''))
    if directory.is_absolute()or directory==Path('.')or '..'in directory.parts:
        raise ValueError('failed-history decoder directory is not manifest-relative')
    for item in files:
        if not Path(item['path']).is_relative_to(directory):
            raise ValueError('failed-history decoder asset has the wrong directory')
    actual_files={str(p.relative_to(path.parent))for p in (path.parent/directory).iterdir()if p.is_file()}
    if actual_files!={item['path']for item in files}:raise ValueError('failed-history decoder has unbound files')
    rows={}
    sources=value.get('sources')
    if (not isinstance(sources,list)or not sources or any(
            not isinstance(item,dict)or item.get('split')!='train'
            or not re.fullmatch('[0-9a-f]{64}',str(item.get('sha256','')))for item in sources)
            or len({item['sha256']for item in sources})!=len(sources)):
        raise ValueError('failed-history sources require unique exact TRAIN hashes')
    sources={item['sha256']for item in sources}
    for row in value['rows']:
        key=(row.get('source_sha256'),row.get('id'))
        if (key in rows or key[0]not in sources or not isinstance(key[1],str)
                or row.get('split')!='train' or row.get('tokenizer_fingerprint')!=decoder.get('fingerprint')
                or not isinstance(row.get('source_groups'),list)or not row['source_groups']
                or not re.fullmatch('[0-9a-f]{64}',str(row.get('raw_row_sha256','')))
                or not isinstance(row.get('failure_evidence'),list)or not row['failure_evidence']
                or not isinstance(row.get('zero_spans'),list)or not row['zero_spans']):
            raise ValueError('failed-history manifest row source/role/split is invalid')
        rows[key]=row
    if checked(reference)!=path:raise ValueError('failed-history manifest changed while reading')
    return value,rows,dependencies


@lru_cache(maxsize=2)
def _decoder(directory,inventory,fingerprint):
    from transformers import AutoTokenizer
    from ..data.text_corpus import tokenizer_fingerprint
    tokenizer=AutoTokenizer.from_pretrained(directory,local_files_only=True,trust_remote_code=False)
    if tokenizer_fingerprint(tokenizer)!=fingerprint:
        raise ValueError('failed-history decoder fingerprint mismatch')
    return tokenizer


def prepare_masks(policy):
    """Load only small immutable mask/decoder metadata; no intrinsic index change."""
    rows={};dependencies=[];decoders={}
    references=policy.get('failed_history_masks',[])
    if not isinstance(references,list):raise ValueError('failed-history mask inputs must be a list')
    for reference in references:
        manifest,entries,items=load_manifest(reference);dependencies.extend(items)
        decoder=manifest['tokenizer']
        directory=str((Path(reference['path']).parent/decoder['directory']).resolve())
        inventory=tuple((item['path'],item['sha256'],item['bytes'])for item in decoder['files'])
        # Every manifest's assets were checked above. Identical verified content
        # can share the existing decoder cache despite separate snapshot paths.
        identity=(inventory,decoder['fingerprint'])
        decoder_binding=decoders.setdefault(identity,(directory,inventory,decoder['fingerprint']))
        for key,row in entries.items():
            source_rows=rows.setdefault(key[0],{})
            if key[1]in source_rows:raise ValueError('failed-history policies duplicate one original row')
            source_rows[key[1]]={'row':row,'decoder':decoder_binding}
    if rows:policy['_failed_history_rows']=rows
    return dependencies


def apply_masks(example,row,*,source_sha256,policy):
    """Zero only verified failed positions, retaining original row/source identity."""
    entry=((policy.get('_failed_history_rows')or{}).get(source_sha256)or{}).get(example.id)
    if entry is None:return example,row
    declared=entry['row']
    if (example.split!='train' or declared['raw_row_sha256']!=digest(row)
            or declared['source_groups']!=row['source_groups']
            or declared['tokenizer_fingerprint']!=example.tokenizer_fingerprint):
        raise ValueError('failed-history mask differs from the original TRAIN row binding')
    validate_source_evidence(row,declared.get('source_evidence')or{})
    tokenizer=_decoder(*entry['decoder'])
    spans,evidence=_mask(row,tokenizer)
    if spans!=declared['zero_spans']or evidence!=declared.get('failure_evidence'):
        raise ValueError('failed-history spans or actual failure evidence mismatch')
    weights=list(example.target_weights)
    for start,stop in spans:
        for index in range(start,stop):weights[index]=0.
    node={**row['nodes'][0],'target_weights':weights}
    return replace(example,target_weights=tuple(weights)),{**row,'nodes':[node]}
