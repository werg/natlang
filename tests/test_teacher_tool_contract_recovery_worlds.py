"""Fresh recovery worlds retain old data and execute their actual service contracts."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from scripts.neuralese_data.build_fresh_recovery_teacher_worlds import (
    worlds, terminal_policy_worlds, tool_contract_recovery_worlds)


def test_existing_worlds_remain_byte_identical():
    for rows, expected in [
        (list(worlds()), '6b844c06fde5431bbc7f9505a9529f48e3602a75df96f46560a91b350393cafa'),
        (list(terminal_policy_worlds(2)), '8922bd4d949f4b0a2d70fc10eb84607cd4fc182fa412da130d9b78fd4e8eeba2')]:
        raw = json.dumps(rows, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()
        assert hashlib.sha256(raw).hexdigest() == expected


def test_fresh_complete_oracles_and_recovery_variants():
    rows = list(tool_contract_recovery_worlds())
    assert len(rows) == len({r['id'] for r in rows}) == 24
    assert rows == list(tool_contract_recovery_worlds())
    old_groups = {g for r in list(worlds()) + list(terminal_policy_worlds(2)) for g in r['source_groups']}
    assert not old_groups & {g for r in rows for g in r['source_groups']}
    assert len({g for r in rows for g in r['source_groups']}) == 8
    assert {r['curriculum']['variant'] for r in rows} == {
        'reserved-service-binding', 'page-range-recovery', 'typed-result-recovery'}
    for row in rows:
        s = row['semantics']; namespace, service = next(iter(s['services'].items()))
        records = json.loads(service.split('const ROWS:Row[] = ', 1)[1].split(';\n', 1)[0])
        policy = s['inputs']['policy']
        selected = [r for r in records if r['category'] == policy['category'] and not r['blocked']
                    and (r['waiver'] or (r['active'] and r['value'] >= policy['minimum']))]
        expected = {'ids': sorted(r['id'] for r in selected), 'total': sum(r['value'] for r in selected), 'count': len(selected)}
        assert s['expected'] == json.loads(s['expected_files']['summary.json']) == expected
        assert row['curriculum']['reference']['root'][-1][1]['value'] == expected
        verdicts = []
        for record, child in zip(records, row['curriculum']['reference']['children'], strict=True):
            assert child['match'] == ['select_record(', record['id']]
            assert child['calls'][-1] == ['return_result', {'status': 'success', 'value': record in selected}]
            verdicts.append(record in selected)
        assert verdicts[:4] == [False, True, True, False]
        assert row['split'] == 'train' and row['generation']['provider_calls'] == 0
        assert 'Return the resulting boolean directly.' in s['files']['summarize/select_record.nl']
        assert 'an empty page follows' not in service
        if row['curriculum']['variant'] == 'page-range-recovery':
            assert s['failure_seed']['code'] == f'return {namespace}.page(0);'


def test_actual_typescript_page_service_contract():
    # Execute transpiled authored service code rather than restating its bounds in Python.
    root = Path(__file__).resolve().parents[1]
    ts = root / 'node_modules/typescript'
    assert ts.is_dir(), 'use the existing dependency overlay for this worktree'
    node = shutil.which('node'); assert node
    sources = [{'service':next(iter(r['semantics']['services'].values())),
                'seed':r['semantics']['failure_seed']['code'], 'variant':r['curriculum']['variant']}
               for r in tool_contract_recovery_worlds()]
    script = r'''
const ts = require(process.argv[1]);
const vm = require('node:vm');
let data = ''; process.stdin.on('data', x => data += x);
process.stdin.on('end', () => {
 for (const entry of JSON.parse(data)) {
  const source=entry.service;
  if(entry.variant==='typed-result-recovery') {
   const file='/authored-contract-case.ts', text=source+'\n'+entry.seed;
   const options={noEmit:true,strict:true,target:ts.ScriptTarget.ES2022};
   const host=ts.createCompilerHost(options), get=host.getSourceFile.bind(host);
   host.getSourceFile=(path,...args)=>path===file?ts.createSourceFile(path,text,ts.ScriptTarget.ES2022,true):get(path,...args);
   const program=ts.createProgram([file],options,host);
   if(!program.getSemanticDiagnostics().some(d=>d.code===2322 && ts.flattenDiagnosticMessageText(d.messageText,' ').includes("'number'"))) throw Error('typed seed no longer produces a recoverable declared-type error');
  }
  const js = ts.transpileModule(source, {compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.CommonJS}}).outputText;
  const context = {exports:{}}; vm.runInNewContext(js, context);
  const {page,page_count} = context.exports; const count = page_count();
  const all = Array.from({length:count},(_,i)=>page(i+1)).flat();
  if (!all.length || new Set(all.map(r=>r.id)).size !== all.length) throw Error('lost records');
  const raw = JSON.parse(source.split('const ROWS:Row[] = ')[1].split(';\n')[0]);
  if (JSON.stringify(all) !== JSON.stringify(raw)) throw Error('page partition differs');
  for (const n of [0,-1,1.5,count+1,NaN,Infinity]) {
   let rejected=false; try { page(n); } catch(e) { rejected=String(e.message).includes(`integer from 1 through ${count}`); }
   if (!rejected) throw Error('invalid page was accepted: '+n);
  }
 }
 console.log(JSON.stringify({passed:true,services:JSON.parse(data).length}));
});
'''
    result = subprocess.run([node, '-e', script, str(ts)], input=json.dumps(sources), text=True,
                            capture_output=True, check=True)
    assert json.loads(result.stdout) == {'passed': True, 'services': 24}
