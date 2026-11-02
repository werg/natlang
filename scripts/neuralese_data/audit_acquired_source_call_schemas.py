#!/usr/bin/env python3
"""Check original recorded tool arguments against their declared JSON schemas."""
import argparse
from collections import Counter
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import jsonschema.validators
from natlang_neuralese.common.hashing import sha256_file_hex
from natlang_neuralese.common.jsonio import write_json_durable


def bind(path):
    path=Path(path).resolve(strict=True);before=path.stat();digest=sha256_file_hex(path);after=path.stat()
    if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(
            after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):
        raise ValueError('bound source changed during read')
    return {'path':str(path),'sha256':digest,'bytes':after.st_size}


def reject_remote_refs(value):
    if isinstance(value,dict):
        for key,item in value.items():
            if key in ('$ref','$dynamicRef') and (not isinstance(item,str) or not item.startswith('#')):
                raise ValueError('external schema references cannot be verified offline')
            reject_remote_refs(item)
    elif isinstance(value,list):
        for item in value:reject_remote_refs(item)


def inspect(fixture):
    if fixture.get('schema')!='natlang.acquired-static-service-feasibility/1' or fixture.get('training_admission') is not False:
        raise ValueError('requires source observation custody')
    schemas={};failures=[]
    for item in fixture['tool_declarations']:
        fn=item.get('function') or {};name=fn.get('name');schema=fn.get('parameters')
        if item.get('type')!='function' or not isinstance(name,str) or name in schemas or not isinstance(schema,dict):
            raise ValueError('missing or repeated tool schema')
        reject_remote_refs(schema);Draft202012Validator.check_schema(schema)
        schemas[name]=Draft202012Validator(schema)
    counts=Counter()
    for observation in fixture['observations']:
        name=observation['name'];validator=schemas.get(name);counts[name]+=1
        if validator is None:
            failures.append({'call_id':observation['call_id'],'issue':'tool not declared'});continue
        for error in validator.iter_errors(observation['arguments']):
            failures.append({'call_id':observation['call_id'],'issue':error.message,
                'path':[str(part) for part in error.absolute_path]})
    return {'declared_tools':len(schemas),'observed_calls':sum(counts.values()),'calls_by_tool':dict(counts),
        'argument_schema_failures':failures,'schema_dialect':'JSON Schema 2020-12',
        'scope':'Source-declared argument schemas only; no tool existence, execution or task outcome verified.'}


def audit(fixture_path,expected_sha,output):
    fixture_ref=bind(fixture_path)
    if fixture_ref['sha256']!=expected_sha:raise ValueError('source fixture SHA mismatch')
    result=inspect(json.loads(Path(fixture_path).read_text()))
    result.update({'schema':'natlang.acquired-source-call-schema-review/1','training_admission':False,
        'fixture':fixture_ref,'validator':bind(jsonschema.validators.__file__),'builder':bind(__file__)})
    if bind(fixture_path)!=fixture_ref:raise ValueError('source fixture changed during schema review')
    output=Path(output)
    if output.exists():raise ValueError('review artifact already exists')
    write_json_durable(output,result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fixture',required=True)
    p.add_argument('--fixture-sha256',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();r=audit(a.fixture,a.fixture_sha256,a.out)
    print(json.dumps({'observed_calls':r['observed_calls'],'schema_failures':len(r['argument_schema_failures']),
        'calls_by_tool':r['calls_by_tool'],'training_admission':False}))
if __name__=='__main__':main()
