#!/usr/bin/env python3
"""Prepare pinned KernelBench source tasks without executing imported code or claiming timings."""
import argparse, ast, hashlib, json, subprocess
from pathlib import Path


def sha(value):
    return hashlib.sha256(value if isinstance(value,bytes) else value.encode()).hexdigest()


class FamilyTree(ast.NodeTransformer):
    def visit_Constant(self,node):
        return ast.copy_location(ast.Constant(value=type(node.value).__name__),node)


def prepare(source,levels):
    source=source.resolve(strict=True)
    revision=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip():
        raise ValueError('source snapshot must be clean')
    license_bytes=(source/'LICENSE').read_bytes()
    if b'MIT License' not in license_bytes:raise ValueError('review the changed upstream license')
    tasks,held=[],[]
    for level in levels:
        for path in sorted((source/'KernelBench'/level).glob('*.py')):
            relative=path.relative_to(source).as_posix();raw=path.read_bytes()
            try:
                if len(raw)>300_000:raise ValueError('oversized source')
                module=ast.parse(raw.decode())
                model=next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name=='Model')
                functions={n.name:n for n in module.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
                if not {'get_inputs','get_init_inputs'}<=functions.keys():raise ValueError('missing input contract')
                if not any(isinstance(n,ast.FunctionDef) and n.name=='forward' for n in model.body):raise ValueError('missing forward contract')
                forward=next(n for n in model.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
                family=sha(ast.dump(FamilyTree().visit(ast.parse(ast.unparse(forward))),include_attributes=False))
                # Group structural forward aliases across levels before partitioning.
                group='kernelbench/forward-family/'+family
                split='validation' if int(family[:8],16)%5==0 else 'train'
                imports=sorted({n.module.split('.')[0] for n in ast.walk(module) if isinstance(n,ast.ImportFrom) and n.module}|{a.name.split('.')[0] for n in ast.walk(module) if isinstance(n,ast.Import) for a in n.names})
                tasks.append({'schema':'natlang.optimization-source-task/1','id':'kernelbench/'+relative,
                    'family':'gpu-kernel-'+level,'group':group,'split':split,
                    'source':{'url':'https://github.com/ScalingIntelligence/KernelBench','revision':revision,'file':relative,'sha256':sha(raw),'license':'MIT','license_sha256':sha(license_bytes)},
                    'input':{'reference_python':raw.decode(),'contract':'Implement equivalent ModelNew initialization and forward behavior for fresh host-generated inputs; use the provided reference only as a behavior specification.'},
                    'objective':{'correctness':'reference equivalence; per-task numerical contract required','quality':'same-device repeated latency relative to the reference','environment':'pinned isolated GPU/compiler/runtime profile'},
                    'imports':imports,'status':'prepared_requires_gpu_executor',
                    'publication':'Task source only; no generated trajectory, timing, correctness or admission claimed'})
            except (SyntaxError,StopIteration,ValueError,UnicodeError) as error:
                held.append({'file':relative,'sha256':sha(raw),'reason':str(error) or 'unsupported source contract'})
    if not tasks:raise ValueError('no supported source tasks')
    return revision,license_bytes,tasks,held


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--levels',default='level1,level2');args=parser.parse_args()
    levels=args.levels.split(',')
    if not levels or len(set(levels))!=len(levels) or any(level not in ['level1','level2','level3','level4'] for level in levels):parser.error('invalid levels')
    revision,license_bytes,tasks,held=prepare(args.source,levels);args.out.mkdir(parents=True,exist_ok=True)
    body=''.join(json.dumps(task,sort_keys=True)+'\n' for task in tasks)
    with (args.out/'kernel-source-tasks.jsonl').open('x') as f:f.write(body)
    with (args.out/'UPSTREAM_LICENSE.txt').open('xb') as f:f.write(license_bytes)
    manifest={'schema':'natlang.optimization-source-corpus/1','source_revision':revision,'generator_sha256':sha(Path(__file__).read_bytes()),'tasks':len(tasks),'groups':len({t['group'] for t in tasks}),'splits':{s:sum(t['split']==s for t in tasks) for s in ['train','validation']},'held':held,'sha256':sha(body),'provider_calls':0,'gpu_executions':0,'status':'prepared_requires_gpu_executor','levels':levels}
    with (args.out/'manifest.json').open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
    print(json.dumps(manifest))


if __name__=='__main__':main()
