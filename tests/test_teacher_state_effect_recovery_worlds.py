"""Execute fresh reference effects; no model calls or fabricated teacher targets."""
import json
from pathlib import Path
import shutil
import subprocess

from scripts.neuralese_data.build_fresh_recovery_teacher_worlds import (
    state_effect_recovery_worlds, worlds, terminal_policy_worlds, tool_contract_recovery_worlds)


def test_fresh_state_worlds_are_deterministic_train_only_and_independent():
    rows = list(state_effect_recovery_worlds())
    assert rows == list(state_effect_recovery_worlds())
    assert len(rows) == len({r['id'] for r in rows}) == 24
    old = list(worlds()) + list(terminal_policy_worlds()) + list(tool_contract_recovery_worlds())
    assert not {r['id'] for r in rows} & {r['id'] for r in old}
    assert not {g for r in rows for g in r['source_groups']} & {g for r in old for g in r['source_groups']}
    assert len({g for r in rows for g in r['source_groups']}) == 8
    assert all(r['split'] == 'train' and r['generation']['provider_calls'] == 0 for r in rows)
    for row in rows:
        s = row['semantics']
        assert '\nargs:\n' in s['files']['summarize.nl']
        assert json.loads(s['folder_files']['cache.json'])['summary'] != s['expected']
        assert set(s['folder_files']) == set(s['expected_files'])
        assert json.loads(s['expected_files']['cache.json'])['summary'] == s['expected']
        inbox = json.loads(s['folder_files']['inbox.json'])
        assert len(inbox) == 4 and inbox[0] == inbox[-1]
        assert len(json.loads(s['folder_files']['schedule.json'])) == 2
        assert s['folder_files']['notes.txt'] == s['expected_files']['notes.txt']


def test_actual_reference_executes_every_effect_and_is_idempotent():
    node = shutil.which('node'); assert node
    rows = list(state_effect_recovery_worlds())
    script = r'''
let input='';process.stdin.on('data',x=>input+=x);process.stdin.on('end',async()=>{
const AsyncFunction=Object.getPrototypeOf(async function(){}).constructor;
for(const row of JSON.parse(input)) {
 const s=row.semantics, files=new Map(Object.entries(s.folder_files));
 const reference=new AsyncFunction('folder','policy','select_record',row.curriculum.reference.root[0][1].code);
 for(let pass=0;pass<2;pass++) {
  const reads=[],writes=[],asked=[];
  const folder={file(name){return {async readText(){reads.push(name);if(!files.has(name))throw Error('missing '+name);return files.get(name);},async writeText(text){writes.push(name);files.set(name,text);}};}};
  const select_record=async(record,policy)=>{asked.push(record.id);return record.category===policy.category && !record.blocked && (record.waiver || (record.active && record.value>=policy.minimum));};
  const result=await reference(folder,s.inputs.policy,select_record);
  if(JSON.stringify(result)!==JSON.stringify(s.expected))throw Error('answer differs');
  for(const [name,text]of Object.entries(s.expected_files)) {
   const actual=files.get(name);
   if(name.endsWith('.json')) {if(JSON.stringify(JSON.parse(actual))!==JSON.stringify(JSON.parse(text)))throw Error('file differs: '+name);}
   else if(actual!==text)throw Error('unrelated file changed');
  }
  const final=JSON.parse(files.get('records.json'));
  if(JSON.stringify(asked)!==JSON.stringify(final.map(r=>r.id)))throw Error('child obligations lost');
  if(writes.length!==5 || new Set(writes).size!==5 || writes.some(n=>!reads.includes(n)))throw Error('complete effects/readback missing');
 }
}
console.log(JSON.stringify({passed:true,cases:JSON.parse(input).length,passes:2}));
}).on('error',e=>{throw e;});
'''
    result = subprocess.run([node, '-e', script], input=json.dumps(rows), text=True,
                            capture_output=True, check=True)
    assert json.loads(result.stdout) == {'passed': True, 'cases': 24, 'passes': 2}


def test_oracle_requires_archive_inbox_and_schedule_not_only_answer():
    for row in state_effect_recovery_worlds():
        s = row['semantics']
        for name in ('inbox.json', 'archive.json', 'schedule.json', 'cache.json', 'records.json'):
            assert json.loads(s['folder_files'][name]) != json.loads(s['expected_files'][name])
        archive = json.loads(s['expected_files']['archive.json'])
        assert len(archive) == len({event['id'] for event in archive}) == 4
        schedule = json.loads(s['expected_files']['schedule.json'])
        assert len(schedule) == len({entry['id'] for entry in schedule}) == s['expected']['count']
        assert [entry['id'] for entry in schedule] == s['expected']['ids']
